"""Measured finite arm moves over the shared native STM32 ROS transport.

No port ownership, automatic session opening, homing, calibration fabrication,
or recovery from an out-of-work pose. The existing owner opens the session.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

from arm_commissioning import stationary_status
from controller_feedback import load_calibration


def measured_reference(state, calibration, now_ns, *, expected_boot=None, expected_session=None):
    stamp = state.get('monotonic_ns')
    if not isinstance(stamp, int) or not 0 <= now_ns-stamp <= 350_000_000 or not state.get('telemetry_fresh'):
        raise ValueError('Нет свежей телеметрии STM32')
    boot = (state.get('identity') or {}).get('boot')
    session = (state.get('controller') or {}).get('session')
    if not boot or expected_boot is not None and boot != expected_boot:
        raise ValueError('STM32 перезапустилась; движение отменено')
    if expected_session is not None and session != expected_session:
        raise ValueError('Сессия STM32 изменилась; движение отменено')
    samples = (state.get('arm') or {}).get('joints', [])
    if len(samples) != 6:
        raise ValueError('Нужны измерения всех шести приводов')
    positions, physical, ticks, acquired, outside = [], [], [], [], []
    for i, (sample, cal) in enumerate(zip(samples, calibration)):
        at = sample.get('acquired_monotonic_ns')
        if (sample.get('joint') != i+1 or not sample.get('position_valid') or
            not sample.get('raw_valid') or sample.get('error') != 0 or sample.get('device_error') != 0 or
            not isinstance(at, int) or not 0 <= now_ns-at <= 250_000_000):
            raise ValueError('Нет достоверного свежего положения сустава '+str(i+1))
        observed = cal.observe(sample.get('raw_ticks'))
        reported = sample.get('position_rad')
        if not isinstance(reported, (float, int)) or not math.isfinite(reported) or abs(reported-observed['position_rad']) > 1e-6:
            raise ValueError('Калибровка драйвера не совпадает с калибровкой руки')
        positions.append(observed['position_rad']); physical.append(observed['physical_deg'])
        ticks.append(observed['raw_ticks']); acquired.append(at/1e9)
        if observed['outside_soft_limit']:
            outside.append(i+1)
    return dict(at=time.time(), boot_id=boot, session=session, servo_deg=physical,
                position_rad=positions, raw_ticks=ticks, acquired_monotonic=acquired,
                outside_soft_limits=outside, measured=True, attained=False, source='stm32_uart3_readback')


class LegacyNativeManualArm:
    native = True

    def __init__(self, root, node, model, message_type=None):
        if message_type is None:
            from std_msgs.msg import String
            message_type = String
        self.root, self.model, self.message_type = Path(root), model, message_type
        self.profile = json.loads((self.root/'config/controller-profile.json').read_text())
        path = self.root/'config/controller-calibration.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.profile['calibration_sha256']:
            raise ValueError('Калибровка руки изменилась относительно профиля STM32')
        self.calibration = load_calibration(path)
        self.pub = node.create_publisher(message_type, '/explorer/controller_request', 10)
        self.subscription = node.create_subscription(message_type, '/explorer/controller_result', self._result, 100)
        self.lock, self.condition = threading.Lock(), threading.Condition()
        self.cancelled = threading.Event()
        self.ready, self.error, self.stop_revision = False, None, 0
        self.gamepad_permit = lambda: False
        self.source_id, self.sequence = 'native_arm_'+uuid.uuid4().hex, 0
        self.pending, self.results = {}, {}

    def _state(self):
        return json.loads((self.root/'data/controller-state.json').read_text())

    def _refresh_profile(self):
        profile=json.loads((self.root/'config/controller-profile.json').read_text())
        if (profile.get('transport')!='controller_v1' or
            profile.get('calibration_sha256')!=self.profile.get('calibration_sha256')):
            raise ValueError('Калибровка/транспорт изменены; требуется перезапуск исполнителя руки')
        self.profile=profile

    def reference(self, expected_boot=None, expected_session=None):
        self._refresh_profile()
        state=self._state()
        if (state.get('identity') or {}).get('source_sha256')!=self.profile['firmware_source_sha256']:
            raise ValueError('Ожидаю телеметрию выбранной прошивки STM32')
        return measured_reference(state, self.calibration, time.monotonic_ns(),
                                  expected_boot=expected_boot, expected_session=expected_session)

    def _transport(self):
        self._refresh_profile()
        state = self._state()
        if self.profile.get('telemetry_only', True) or state.get('telemetry_only', True):
            raise ValueError(self.profile.get('blocking_reason_ru') or 'Профиль STM32 разрешает только телеметрию')
        if state.get('session_state') != 'active' or state.get('controller', {}).get('mode') != 1:
            raise ValueError('Сначала явно откройте рабочую сессию STM32')
        self.reference()
        return state

    def prepare_geometry(self):
        self.model()
        self.ready = True
        return self.status()

    def status(self):
        current, blocked = {}, None
        try:
            current = self.reference()
            self._transport()
            if current['outside_soft_limits']:
                raise ValueError('Нужно отдельное восстановление суставов: '+str(current['outside_soft_limits']))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blocked = str(exc)
        return dict(ready=self.ready and blocked is None, geometry_ready=self.ready,
                    blocked_by=blocked, servo_deg=current.get('servo_deg'),
                    raw_ticks=current.get('raw_ticks'), position_rad=current.get('position_rad'),
                    estimated_only=False, measured=bool(current), busy=self.lock.locked(), error=self.error,
                    step_deg=None, runtime_ms=None, continuous_motion=blocked is None,
                    continuous_executor_implemented=True, hardware_accepted=self.profile.get('hardware_accepted') is True,
                    physical_stop_latency_verified=False)

    def _result(self, message):
        try:
            result = json.loads(message.data)
            if result.get('source_id') != self.source_id:
                return
            sequence = result.get('source_sequence')
            with self.condition:
                if sequence in self.pending:
                    self.results[sequence] = result
                    self.condition.notify_all()
        except (ValueError, TypeError):
            return

    def _send(self, operation, *, wait=False, permit=None, source_ns=None, expires_ns=None, **fields):
        now_ns = time.monotonic_ns()
        source_ns = now_ns if source_ns is None else source_ns
        expires_ns = source_ns+150_000_000 if expires_ns is None else expires_ns
        if not 0 <= now_ns-source_ns < 150_000_000 or not now_ns < expires_ns <= source_ns+150_000_000:
            raise ValueError('Срок команды руки истёк до отправки')
        with self.condition:
            if len(self.pending) >= 256:
                oldest = next(iter(self.pending))
                self.pending.pop(oldest); self.results.pop(oldest, None)
            self.sequence += 1
            sequence = self.sequence
            self.pending[sequence] = (now_ns, operation)
        try:
            self.pub.publish(self.message_type(data=json.dumps(dict(operation=operation,
                source_id=self.source_id, source_sequence=sequence,
                source_monotonic_ns=source_ns, expires_monotonic_ns=expires_ns, **fields))))
        except Exception:
            with self.condition:
                self.pending.pop(sequence, None)
            raise
        if wait:
            end = time.monotonic()+.15
            while True:
                if permit is not None:
                    permit()
                with self.condition:
                    if sequence in self.results:
                        result = self.results.pop(sequence)
                        self.pending.pop(sequence, None)
                        if result.get('accepted') is not True:
                            raise ValueError('STM32 отклонила '+operation+': '+str(result.get('reason', result.get('error'))))
                        return result
                    if time.monotonic() >= end:
                        self.pending.pop(sequence, None)
                        raise ValueError('Нет подтверждения STM32 для '+operation)
                    self.condition.wait(.01)
        return sequence

    def _check_results(self):
        with self.condition:
            for sequence, (sent, operation) in list(self.pending.items()):
                result = self.results.pop(sequence, None)
                if result is not None:
                    del self.pending[sequence]
                    if result.get('accepted') is not True:
                        raise ValueError('STM32 отклонила '+operation+': '+str(result.get('reason', result.get('error'))))
                elif time.monotonic_ns()-sent > 150_000_000:
                    del self.pending[sequence]
                    raise ValueError('Нет подтверждения STM32 для '+operation)

    def stop(self):
        self.stop_revision += 1
        self.cancelled.set()
        try:
            self._send('ARM_CANCEL')
            requested, reason = True, None
        except (OSError, ValueError, RuntimeError) as exc:
            requested, reason = False, str(exc)
        return dict(no_further_steps=True, cancel_requested=requested, reason=reason,
                    physical_stop_latency_verified=False, hardware_emergency_stop=False)

    def _positions(self, physical):
        if len(physical) != 6 or any(type(x) not in (int, float) or not math.isfinite(x) for x in physical):
            raise ValueError('Нужны шесть конечных углов')
        positions = []
        for c, angle in zip(self.calibration, physical):
            raw = (angle-c.physical_degrees_at_raw_zero)/c.physical_degrees_per_tick
            positions.append(c.target_position(raw*c.radians_per_tick+c.radians_at_raw_zero))
        return positions

    def _physical(self, positions):
        return [((q-c.radians_at_raw_zero)/c.radians_per_tick)*c.physical_degrees_per_tick+
                c.physical_degrees_at_raw_zero for q, c in zip(positions, self.calibration)]

    def _path(self, current, goal, trajectory=None):
        import numpy as np
        from timed_trajectory import Limits, TimedPath
        start = current['position_rad']
        target = self._positions(goal)
        limits = Limits(np.array([c.lower for c in self.calibration]),
                        np.array([c.upper for c in self.calibration]),
                        np.full(6, math.radians(10)), np.full(6, math.radians(30)),
                        np.full(6, math.radians(120)))
        model = self.model()
        def collision_free(a, b):
            start_deg, goal_deg = self._physical(a), self._physical(b)
            return all(model.path(start_deg[:5], goal_deg[:5], shape)['valid'] is True
                       for shape in (0., -.25, -.5, -.75, -1., -1.25, -1.54))
        if trajectory is None:
            duration = max(.5, max(abs(a-b) for a, b in zip(start, target))/math.radians(8))
            times, positions, velocities, accelerations = [0., duration], [start, target], [[0.]*6]*2, [[0.]*6]*2
        else:
            expected = ['arm'+str(i)+'_Joint' for i in range(1, 6)]
            names = trajectory['names']
            if len(names) != 5 or set(names) != set(expected):
                raise ValueError('MoveIt должен передать пять однозначно названных суставов')
            order = [names.index(name) for name in expected]
            times = list(trajectory['times'])
            arrays = []
            for field in ('positions', 'velocities', 'accelerations'):
                rows = trajectory[field]
                if len(rows) != len(times) or any(len(row) != 5 for row in rows):
                    raise ValueError('Нужны полные положения, скорости и ускорения MoveIt')
                arrays.append([[row[index] for index in order]+[start[5] if field == 'positions' else 0.] for row in rows])
            positions, velocities, accelerations = arrays
            if len(times) < 2 or not np.allclose(positions[0], start, atol=math.radians(.5), rtol=0):
                raise ValueError('Начало пути MoveIt не совпадает с текущей опорной позой')
            if not np.allclose(positions[-1][:5], target[:5], atol=math.radians(.1), rtol=0):
                raise ValueError('Конец пути MoveIt не совпадает с запрошенной позой')
            if abs(target[5]-start[5]) > 1e-6:
                if max(map(abs, velocities[-1])) > 1e-5 or max(map(abs, accelerations[-1])) > 1e-4:
                    raise ValueError('Перед движением захвата путь MoveIt должен завершиться в покое')
                duration = max(.5, abs(target[5]-start[5])/math.radians(8))
                times.append(times[-1]+duration)
                positions.append(positions[-1][:5]+[target[5]])
                velocities.append([0.]*6); accelerations.append([0.]*6)
        path = TimedPath(['actuator'+str(i) for i in range(1, 7)], times,
                         positions, velocities, accelerations, limits, collision_free)
        if path.duration > 60:
            raise ValueError('Конечное движение дольше 60 секунд')
        return path

    def _persist(self, reference, phase, **fields):
        value = dict(reference)
        value.update(phase=phase, controller_boot_id=reference.get('boot_id'),
                     boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        value.update(fields)
        target = self.root/'data/arm-state.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value)); temporary.replace(target)
        return value

    def _wait_cancel_complete(self, boot, session, permit):
        deadline=time.monotonic()+.35
        while time.monotonic()<deadline:
            permit()
            state=self._state()
            measured_reference(state,self.calibration,time.monotonic_ns(),
                               expected_boot=boot,expected_session=session)
            controller=state.get('controller',{})
            if controller.get('arm_enabled') is False and controller.get('arm_cancel_pending') is False:
                return
            time.sleep(.005)
        raise ValueError('STM32 приняла отмену, но ещё не подтвердила завершение удержания')

    def move(self, start, goal, deadline=None, expected_stop_revision=None, execution_permit=None, source='operator', trajectory=None,speed='normal'):
        from timed_trajectory import Execution
        if source not in ('operator', 'supervised_policy', 'supervised_trajectory', 'local_mission'):
            raise ValueError('Неверный источник команды')
        if source == 'local_mission' and not callable(execution_permit):
            raise ValueError('Нет разрешения локальной миссии')
        if not self.ready:
            raise ValueError('Сначала подготовьте геометрию руки')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Рука уже выполняет движение')
        enabled = False
        record = None
        current = None
        activity_started = False
        try:
            revision = self.stop_revision if expected_stop_revision is None else expected_stop_revision
            if revision != self.stop_revision or deadline is not None and time.monotonic() > deadline:
                raise ValueError('Команда отменена или устарела')
            self.cancelled.clear()
            with (self.root/'data/arm-commissioning.lock').open('w') as file_lock:
                fcntl.flock(file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._transport()
                current = self.reference()
                if current['outside_soft_limits']:
                    raise ValueError('Исходная поза вне рабочего диапазона; требуется отдельное восстановление')
                planned_start = self._positions(start)
                if max(abs(a-b) for a, b in zip(planned_start, current['position_rad'])) > math.radians(.5):
                    raise ValueError('Измеренная исходная поза изменилась')
                path = self._path(current, goal, trajectory)
                boot, session = current['boot_id'], current['session']
                def permit():
                    if self.cancelled.is_set() or revision != self.stop_revision:
                        raise ValueError('Движение отменено')
                    if deadline is not None and not self.gamepad_permit():
                        raise ValueError('Кнопка разрешения отпущена')
                    if execution_permit is not None:
                        execution_permit()
                    self._transport()
                    base_status=json.loads((self.root/'data/status.json').read_text())
                    stationary_status(base_status, time.time())
                    if source!='local_mission' and base_status.get('mission'):
                        raise ValueError('Сначала отмените автономную миссию для ручного управления рукой')
                    self.reference(boot, session)
                permit()
                # A stale gamepad decision must not begin after geometry checks.
                if deadline is not None and time.monotonic() > deadline:
                    raise ValueError('Команда устарела во время проверки геометрии')
                with self.condition:
                    self.pending.clear(); self.results.clear()
                self._send('CALIBRATION', wait=True, permit=permit)
                # Cancellation is required even if enable arrived at the MCU
                # but its result was lost on the return path.
                self._persist(current, 'command_in_progress', at=time.time(),
                              ends_monotonic=time.monotonic()+path.duration+3., command_source=source)
                activity_started = True
                enabled = True
                self._send('ARM_ENABLE', wait=True, permit=permit)
                def feedback():
                    measured = self.reference(boot, session)
                    return dict(position=measured['position_rad'], acquired_monotonic=measured['acquired_monotonic'], valid=[True]*6)
                execution = Execution(path,
                    lambda q, now, expires, generation: self._send('ARM', position_rad=q,
                        source_ns=int(now*1e9), expires_ns=min(int(expires*1e9), int(now*1e9)+150_000_000)),
                    lambda: self._send('ARM_CANCEL'), feedback,
                    period_s=.05, tolerance=math.radians(.5), settle_s=.25)
                while execution.state not in ('reached', 'fault', 'cancelled'):
                    permit(); self._check_results()
                    execution.tick(time.monotonic())
                    time.sleep(.005)
                self._check_results()
                if execution.state != 'reached':
                    raise ValueError(execution.reason)
                # Drain every outstanding result; ACK is necessary but the
                # execution's measured settle test is what proves attainment.
                while self.pending:
                    permit(); self._check_results(); time.sleep(.005)
                self._send('ARM_CANCEL', wait=True)
                self._wait_cancel_complete(boot,session,permit)
                enabled = False
                record = dict(self.reference(boot, session), attained=True, phase='measured_reached',
                              requested_servo_deg=list(goal), command_source=source,
                              duration_s=path.duration, trajectory_sha256=path.source_sha256)
                self.error = None
                return record
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.error = str(exc)
            raise ValueError(self.error) from exc
        finally:
            if enabled:
                try:
                    self._send('ARM_CANCEL')
                except (OSError, ValueError, RuntimeError):
                    pass
            try:
                if record is not None:
                    self._persist(record, 'measured_reached', ends_monotonic=time.monotonic())
                elif activity_started:
                    # An interrupted command is not a measured stationary pose.
                    self._persist(current, 'interrupted', at=time.time(), attained=False, measured=False,
                                  reason=self.error, ends_monotonic=time.monotonic())
            finally:
                self.lock.release()


from manual_reference_arm import ReferenceArmMixin

class NativeManualArm(ReferenceArmMixin, LegacyNativeManualArm):
    pass

ManualArm = NativeManualArm
