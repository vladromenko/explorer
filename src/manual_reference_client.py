"""Operator client through the existing ROS UART owner, never opens serial."""
import fcntl
import json
import math
from pathlib import Path
import time
import uuid


class ManualClient:
    def __init__(self, root='/home/vlad/Explorer'):
        import rclpy
        from std_msgs.msg import String
        self.root = Path(root)
        self.rclpy, self.message_type = rclpy, String
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = rclpy.create_node('manual_reference_operator_'+uuid.uuid4().hex[:8])
        self.pub = self.node.create_publisher(String, '/explorer/controller_request', 10)
        self.sub = self.node.create_subscription(String, '/explorer/controller_result', self._result, 100)
        self.source = 'manual_'+uuid.uuid4().hex
        self.seq, self.results = 0, {}
        self._wait_transport()
        self._probe_transport()

    def _wait_transport(self):
        end = time.monotonic()+5
        # Request discovery can finish before the result subscription matches.
        # Sending then loses the first ACK and lets a motion lease expire.
        while (self.pub.get_subscription_count() < 1 or self.sub.get_publisher_count() < 1) and time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=.05)
        if self.pub.get_subscription_count() != 1 or self.sub.get_publisher_count() != 1:
            self.close()
            raise ValueError('Нужен один controller_driver с обнаруженными каналами команд и ответов')

    def _probe_transport(self):
        # In CommandOnly 0.3.1 this operation returns EX_NOT_READY without
        # reading or writing the servo bus. A matching rejection also proves
        # both ROS data paths, unlike discovery counts alone. Only this
        # non-actuating probe is retried; physical commands are never retried.
        for attempt in range(3):
            result = self._exchange('MR_DIAGNOSTIC')
            if result is not None:
                return
        self.close()
        raise ValueError('Нет подтверждённого обмена с драйвером; команды приводам не отправлены')

    def close(self):
        self.node.destroy_node()
        if self.owns_context:
            self.rclpy.try_shutdown()

    def _result(self, message):
        value = json.loads(message.data)
        if value.get('source_id') == self.source:
            self.results[value.get('source_sequence')] = value

    def state(self):
        state = json.loads((self.root/'data/controller-state.json').read_text())
        profile = json.loads((self.root/'config/controller-profile.json').read_text())
        if (profile.get('manual_reference_version') != 1 or
            (state.get('identity') or {}).get('source_sha256') != profile.get('firmware_source_sha256') or
            not state.get('telemetry_fresh') or
            not 0 <= time.monotonic_ns()-state.get('monotonic_ns', 0) < 300_000_000):
            raise ValueError('Нет свежей идентичности установленной manual-reference прошивки')
        return state

    def _exchange(self, operation, **fields):
        self.seq += 1
        now = time.monotonic_ns()
        request = dict(operation=operation, source_id=self.source, source_sequence=self.seq,
            source_monotonic_ns=now, expires_monotonic_ns=now+200_000_000, **fields)
        self.pub.publish(self.message_type(data=json.dumps(request, allow_nan=False)))
        end = time.monotonic()+.3
        while self.seq not in self.results and time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=.005)
        return self.results.pop(self.seq, None)

    def send(self, operation, **fields):
        result = self._exchange(operation, **fields)
        if not result or result.get('accepted') is not True:
            raise ValueError(operation+': '+str((result or {}).get('reason', (result or {}).get('error', 'нет ACK'))))
        return result

    def open(self):
        state = self.state()
        if state.get('session_state') == 'active':
            return
        if state.get('session_state') == 'fault':
            self.send('CLEAR')
            self.wait_session('disarmed')
        self.send('OPEN')
        self.wait_session('active')

    def wait_session(self, desired):
        end = time.monotonic()+1
        while time.monotonic() < end:
            if self.state().get('session_state') == desired:
                return
            self.rclpy.spin_once(self.node, timeout_sec=.02)
        raise ValueError('Нет ожидаемого состояния сессии '+desired)

    def wait_phase(self, desired, timeout=5, heartbeat=True, after_ns=0, generation=None):
        end, next_keep = time.monotonic()+timeout, time.monotonic()
        while time.monotonic() < end:
            state = self.state()
            ref = state.get('manual_reference') or {}
            fresh = ref.get('received_monotonic_ns', 0) > after_ns
            if fresh and ref.get('phase') == 'FAULT':
                raise ValueError('Калибровка/движение: '+str(ref.get('error_name',ref.get('error')))+'; привод '+str(ref.get('failed_id'))+'; register='+hex(ref.get('last_register',0)))
            matches_generation = generation is None or ref.get('generation') == generation
            if fresh and matches_generation and ref.get('phase') == desired:
                return ref
            if heartbeat and time.monotonic() >= next_keep:
                self.send('MR_KEEPALIVE')
                next_keep = time.monotonic()+.07
            self.rclpy.spin_once(self.node, timeout_sec=.005)
        raise ValueError('Не достигнуто состояние '+desired+' за '+str(timeout)+' с')

    def begin(self, supported=False):
        if supported is not True:
            raise ValueError('Сначала поддержите руку; подтвердите это флагом --supported')
        self.open()
        self.send('MR_ABORT')
        self.send('CALIBRATION')
        at = time.monotonic_ns()
        self.send('MR_BEGIN', operator_supported=True)
        return self.wait_phase('MANUAL_SETUP', after_ns=at)

    def capture(self, supported=False):
        if supported is not True:
            raise ValueError('Рука должна оставаться поддержанной до READY; нужен --supported')
        state = self.state()
        if (state.get('manual_reference') or {}).get('phase') != 'MANUAL_SETUP':
            raise ValueError('Сначала выполните calibrate begin и дождитесь MANUAL_SETUP')
        at = time.monotonic_ns()
        self.send('MR_CAPTURE', operator_supported=True)
        ref = self.wait_phase('READY', after_ns=at)
        from controller_feedback import load_calibration
        from manual_reference_host import effective_calibration
        cal = effective_calibration(load_calibration(self.root/'config/controller-calibration.json'), ref)
        reference_degrees = [c.physical_degrees_at_raw_zero+c.physical_degrees_per_tick*raw
                             for c,raw in zip(cal,ref['raw_reference'])]
        record = dict(ref, operator_reference_servo_deg=reference_degrees,
            at=time.time(), firmware_source_sha256=state['identity']['source_sha256'],
            calibration_persistent=False, active_after_reboot=False,
            note='Capture is volatile in MCU; this file records evidence, not an automatic boot seed.')
        path = self.root/'data/manual-reference.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2)); tmp.replace(path)
        return record

    def move(self, servo_degrees, duration=1):
        from controller_feedback import load_calibration
        from manual_reference_host import effective_calibration
        state = self.state(); ref = state.get('manual_reference') or {}
        if ref.get('phase') != 'READY' or not ref.get('reference_valid'):
            raise ValueError('Сначала закончите ручную установку')
        cal = effective_calibration(load_calibration(self.root/'config/controller-calibration.json'), ref)
        if len(servo_degrees) != 6 or not all(math.isfinite(x) for x in servo_degrees):
            raise ValueError('Нужны шесть конечных углов заводского интерфейса')
        positions = []
        for c, x in zip(cal, servo_degrees):
            raw = (x-c.physical_degrees_at_raw_zero)/c.physical_degrees_per_tick
            q = raw*c.radians_per_tick+c.radians_at_raw_zero
            c.target_raw(q)
            positions.append(q)
        at = time.monotonic_ns()
        self.send('MR_MOVE', position_rad=positions, duration_s=duration)
        # Wait for a NEW state from this motion. Very short moves still have a
        # >=20ms sampling phase; a stale READY cannot masquerade as completion.
        time.sleep(.025)
        final = self.wait_phase('READY', timeout=65, after_ns=at, generation=(ref['generation']+1)&0xffffffff)
        if any(abs(a-b) > .002 for a, b in zip(final['position_rad'], positions)):
            raise ValueError('Траектория отменена или конечная команда не подтверждена')
        return dict(command_completed=True, measured_reached=False, reference=final)

    def jog(self, joint, degrees, duration=1):
        from controller_feedback import load_calibration
        from manual_reference_host import reference_from_state
        ref = reference_from_state(self.state(), load_calibration(self.root/'config/controller-calibration.json'), time.monotonic_ns())
        if not 1 <= joint <= 6 or not math.isfinite(degrees) or abs(degrees)>10:
            raise ValueError('Для jog: joint 1..6, |delta-deg| <= 10')
        target = ref['servo_deg']; target[joint-1] += degrees
        return self.move(target, duration)

    def stop(self):
        self.send('MR_STOP')
        return dict(command_stop_requested=True, physical_stop_verified=False)
