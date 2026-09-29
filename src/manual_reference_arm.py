"""Conditional adapter for the existing NativeManualArm; legacy stays intact."""
import json
import fcntl
import math
from pathlib import Path
import time
from manual_reference_host import reference_from_state


class ReferenceArmMixin:
    def _is_manual_reference(self):
        return self.profile.get('manual_reference_version') == 1

    def reference(self, expected_boot=None, expected_session=None):
        self._refresh_profile()
        if not self._is_manual_reference():
            return super().reference(expected_boot, expected_session)
        live=self._state()
        if (live.get('identity') or {}).get('source_sha256')!=self.profile.get('firmware_source_sha256'):
            raise ValueError('Ожидаю выбранную прошивку CommandOnly')
        ref=live.get('manual_reference') or {}
        if not ref.get('reference_valid'):
            reason=ref.get('error_name')
            raise ValueError('Нужна ручная исходная поза'+
                             (' · STM32 '+reason if reason and reason!='EX_OK' else ''))
        if not hasattr(self, '_reference_nominal'):
            self._reference_nominal=list(self.calibration)
        result=reference_from_state(live,self._reference_nominal,time.monotonic_ns(),expected_boot,expected_session)
        self.calibration=result.pop('calibration')
        return result

    def _transport(self):
        self._refresh_profile()
        if not self._is_manual_reference(): return super()._transport()
        state=self._state()
        if state.get('session_state')!='active' or state.get('controller',{}).get('mode')!=1:
            raise ValueError('Сначала explorer arm calibrate begin')
        self.reference()
        return state

    def status(self):
        if not self._is_manual_reference(): return super().status()
        try:
            ref=self.reference(); self._transport()
            blocked=None
        except (OSError,ValueError,KeyError,TypeError) as exc:
            ref={};blocked=str(exc)
        from command_arm_state import describe
        contract=describe(self._state(),self.calibration,time.monotonic_ns(),self.profile)
        return dict(ready=self.ready and blocked is None,geometry_ready=self.ready,blocked_by=blocked,contract=contract,
            servo_deg=ref.get('servo_deg'),raw_ticks=ref.get('raw_ticks'),position_rad=ref.get('position_rad'),
            estimated_only=True,measured=False,busy=self.lock.locked(),error=self.error,
            continuous_motion=blocked is None,continuous_executor_implemented=True,
            hardware_accepted=False,physical_stop_latency_verified=False,
            reference_source=contract['reference_source'],state_source=contract['state_source'])

    def stop(self):
        result=super().stop()
        if self._is_manual_reference():result.update(stop_type='commanded_hold',measured=False)
        return result

    def _wait_cancel_complete(self,boot,session,permit):
        if not self._is_manual_reference(): return super()._wait_cancel_complete(boot,session,permit)
        end=time.monotonic()+.5
        while time.monotonic()<end:
            state=self._state(); ref=state.get('manual_reference') or {}
            if ref.get('boot')!=boot or ref.get('session')!=session:
                raise ValueError('Сессия изменилась во время отмены')
            if ref.get('phase')=='READY':
                self.reference(boot,session)
                return
            if ref.get('phase') not in ('EXECUTING','CANCEL_PENDING'):
                raise ValueError('Нет достоверного завершения отмены: '+str(ref.get('phase')))
            time.sleep(.005)
        raise ValueError('Нет подтверждения прекращения командной траектории')

    def move(self,start,goal,deadline=None,expected_stop_revision=None,execution_permit=None,source='operator',trajectory=None):
        if not self._is_manual_reference():
            return super().move(start,goal,deadline,expected_stop_revision,execution_permit,source,trajectory)
        if source not in ('operator','supervised_policy','supervised_trajectory','local_mission'):
            raise ValueError('Неверный источник команды')
        if not self.ready: raise ValueError('Подготовьте существующую геометрию руки')
        if not self.lock.acquire(blocking=False): raise ValueError('Рука уже выполняет команду')
        enabled=False;current=None;owner=None
        try:
            owner=(self.root/'data/manual-reference-operator.lock').open('a')
            try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Ручная калибровка или другой исполнитель уже управляет рукой')
            revision=self.stop_revision if expected_stop_revision is None else expected_stop_revision
            if deadline is not None and time.monotonic()>deadline:raise ValueError('Команда джойстика устарела до запуска')
            self.cancelled.clear(); current=self.reference();self._transport()
            boot,session=current['boot_id'],current['session']
            path=self._path(current,goal,trajectory)
            curve=dict(times=path.times.tolist(),coefficients=[c.tolist() for c in path.coefficients],
                       units='ROS radians',interpolation='quintic',duration=path.duration)
            def permit():
                if self.cancelled.is_set() or self.stop_revision!=revision: raise ValueError('Движение отменено')
                if deadline is not None and not self.gamepad_permit(): raise ValueError('Управление джойстиком прекращено')
                if execution_permit is not None: execution_permit()
                self.reference(boot,session)
            permit()
            if deadline is not None and time.monotonic()>deadline:raise ValueError('Команда устарела во время проверки пути')
            self._persist(current,'command_in_progress',q_goal=self._positions(goal),trajectory=curve,
                          trajectory_time=0.,command_generation=current['command_generation'],
                          ends_monotonic=time.monotonic()+path.duration+2)
            enabled=True;self._send('ARM_ENABLE',wait=True,permit=permit)
            started=time.monotonic();next_sample=started
            while True:
                permit()
                now=time.monotonic()
                if now>=next_sample:
                    elapsed=min(path.duration,now-started)
                    sample=path.sample(elapsed)
                    q=sample.position.tolist() if hasattr(sample,'position') else sample['position']
                    self._send('ARM',position_rad=list(q),wait=True,permit=permit)
                    # Only the subsequent MCU raw_sent state advances the estimate.
                    sent=self.reference(boot,session)
                    self._persist(sent,'command_in_progress',q_goal=self._positions(goal),trajectory=curve,
                                  trajectory_time=elapsed,command_generation=sent['command_generation'])
                    next_sample=now+.05
                    if elapsed>=path.duration: break
                time.sleep(.005)
            # Command-state settling is not measured attainment.
            end=time.monotonic()+.2
            command_matched=False
            while time.monotonic()<end:
                permit()
                ref=self.reference(boot,session)
                if all(abs(a-b)<.002 for a,b in zip(ref['position_rad'],q)):
                    command_matched=True
                    break
                self._send('MR_KEEPALIVE',wait=True,permit=permit)
                time.sleep(.01)
            if not command_matched:
                raise ValueError('Последняя точка не подтверждена командным состоянием STM32')
            self._send('ARM_CANCEL',wait=True); enabled=False
            self._wait_cancel_complete(boot,session,lambda:None)
            record=dict(self.reference(boot,session),attained=False,measured=False,
                command_completed=True,phase='command_completed_unverified',requested_servo_deg=list(goal))
            self._persist(record,record['phase'],ends_monotonic=time.monotonic())
            return record
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            self.error=str(exc);raise ValueError(self.error) from exc
        finally:
            if enabled:
                try: self._send('ARM_CANCEL')
                except (OSError,ValueError,RuntimeError): pass
            if owner is not None:owner.close()
            self.lock.release()
