"""Persistent publisher for finite, CAD-checked operator position steps.

Servo state remains an estimate. No velocity stream, autonomous policy input or
automatic boot homing. An accepted 150 ms servo command cannot be recalled.
"""
import fcntl
import json
from pathlib import Path
import threading
import time
from arm_msgs.msg import ArmJoints
from rclpy.duration import Duration
from arm_commissioning import HARD_LIMITS,stationary_status

class ManualArm:
    def __init__(self,root,node,model):
        self.root=Path(root);self.model=model;self.pub=node.create_publisher(ArmJoints,'/arm6_joints',1)
        self.lock=threading.Lock();self.cancelled=threading.Event();self.ready=False;self.error=None
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.gamepad_permit=lambda:False
        self.stop_revision=0

    def reference(self):
        stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time())
        state=json.loads((self.root/'data/arm-state.json').read_text())
        if state.get('boot_id')!=self.boot or state.get('phase')!='command_elapsed_observation_required':
            raise ValueError('После включения сначала подготовьте руку в свободном пространстве')
        # Any observed MCU link fault invalidates the estimate until a new homing command.
        fault=self.root/'data/arm-telemetry-fault.json'
        if fault.exists() and json.loads(fault.read_text()).get('at',0)>state['at']:
            raise ValueError('Связь с контроллером прерывалась; нужно заново подготовить руку')
        return state

    def prepare_geometry(self):
        self.model();self.ready=True
        return self.status()

    def status(self):
        try:state=self.reference();blocked=None
        except (OSError,KeyError,ValueError) as exc:state={};blocked=str(exc)
        return dict(ready=self.ready,blocked_by=blocked,servo_deg=state.get('servo_deg'),
                    estimated_only=True,busy=self.lock.locked(),error=self.error,
                    step_deg=2,runtime_ms=150,continuous_motion=False)

    def write(self,record,phase):
        path=self.root/'data/arm-state.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(record,phase=phase,updated_at=time.time())));tmp.replace(path)

    def stop(self):
        self.stop_revision+=1
        self.cancelled.set()
        return dict(no_further_steps=True,commanded_step_duration_ms=150,physical_stop_latency_verified=False,
                    hardware_emergency_stop=False)

    def move(self,start,goal,deadline=None,expected_stop_revision=None,execution_permit=None,source="operator"):
        if source not in ('operator','supervised_policy'):raise ValueError('Неверный источник команды')
        if not self.ready:raise ValueError('Сначала дождитесь подготовки геометрии руки')
        if len(start)!=6 or len(goal)!=6 or any(type(v) is not int for v in start+goal):raise ValueError('Нужны шесть целых углов')
        if not any(a!=b for a,b in zip(start,goal)):raise ValueError('Нулевой шаг')
        for a,b,(lo,hi) in zip(start,goal,HARD_LIMITS):
            if not lo<=a<=hi or not lo<=b<=hi or abs(a-b)>2:raise ValueError('Шаг превышает 2° или предел сустава')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        sent=False;record=None
        try:
            if expected_stop_revision is not None and expected_stop_revision!=self.stop_revision:
                raise ValueError('Команда отменена во время планирования')
            self.cancelled.clear()
            with (self.root/'data/arm-commissioning.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                current=self.reference()
                if current['servo_deg']!=start:raise ValueError('Исходное положение команды изменилось')
                for shape in (0.,-.2,-.4,-.6,-.8):
                    if not self.model().path(start[:5],goal[:5],shape)['valid']:raise ValueError('MoveIt: столкновение с роботом или полом')
                self.reference()
                if self.cancelled.is_set() or (expected_stop_revision is not None and expected_stop_revision!=self.stop_revision) or deadline is not None and time.monotonic()>deadline:
                    raise ValueError('Команда отменена или истекла до отправки')
                if deadline is not None and not self.gamepad_permit():
                    raise ValueError('Кнопка разрешения отпущена или панель отключена')
                if execution_permit is not None:execution_permit()
                if not self.pub.get_subscription_count():raise ValueError('Контроллер руки не подключён')
                record=dict(at=time.time(),servo_deg=goal,runtime_ms=150,boot_id=self.boot,
                            ends_monotonic=time.monotonic()+.15,source='commanded_only',measured=False,
                            attained=False,observed_clear=True,publish_count=1,operator_step=source=="operator",command_source=source)
                msg=ArmJoints(time=150)
                for i,value in enumerate(goal,1):setattr(msg,'joint'+str(i),value)
                self.write(record,'command_in_progress');self.pub.publish(msg);sent=True
                if not self.pub.wait_for_all_acked(Duration(seconds=1)):
                    raise ValueError('Нет подтверждения доставки; положение неизвестно')
                end=record['ends_monotonic']+.15
                while time.monotonic()<end:
                    stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time())
                    time.sleep(.025)
                record['dds_acknowledged']=True
                self.write(record,'command_elapsed_observation_required')
                with (self.root/'data/arm-commissioning.jsonl').open('a') as log:
                    log.write(json.dumps(dict(record,event='observation_due'))+'\n')
                self.error=None
                return record
        except (OSError,ValueError,KeyError) as exc:
            self.error=str(exc)
            if sent:self.write(record,'monitor_failed_state_unknown')
            raise ValueError(self.error)
        finally:self.lock.release()
