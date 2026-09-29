"""Persistent publisher for finite, CAD-checked operator position steps.

Servo state remains an estimate. No velocity stream or
automatic boot homing. An accepted finite servo command cannot be recalled.
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
    factory_timed=True
    def __init__(self,root,node,model):
        self.root=Path(root);self.model=model;self.pub=node.create_publisher(ArmJoints,'/arm6_joints',1)
        self.lock=threading.Lock();self.cancelled=threading.Event();self.ready=False;self.error=None
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.gamepad_permit=lambda:False
        self.stop_revision=0
        self.active_runtime_ms=0
        try:self.startup_config=json.loads((self.root/'config/factory-arm-startup.json').read_text())
        except (OSError,ValueError):self.startup_config={}
        if self.startup_config.get('enabled') is True:
            threading.Thread(target=self._automatic_startup_home,daemon=True).start()

    def _automatic_startup_home(self):
        """Establish a known factory pose once near a full Jetson boot.

        A later web-service restart must not move the arm unexpectedly.
        """
        try:
            delay=float(self.startup_config.get('delay_after_web_start_s',8))
            window=float(self.startup_config.get('startup_window_s',180))
            time.sleep(max(1,min(delay,30)))
            uptime=float(Path('/proc/uptime').read_text().split()[0])
            if not 0<=uptime<=window:return
            deadline=time.monotonic()+30
            while not self.pub.get_subscription_count() and time.monotonic()<deadline:time.sleep(.25)
            self.home_reference(False,'automatic_factory_startup')
        except Exception as exc:
            self.error='Автоподготовка руки не выполнена: '+str(exc)

    def home_reference(self,observing,source='operator_factory_home'):
        """Command the known factory 90-degree pose; never claim measurement."""
        if source!='automatic_factory_startup' and observing is not True:
            raise ValueError('Подтвердите наблюдение и свободное пространство вокруг руки')
        pose=self.startup_config.get('pose_deg',[90]*6)
        runtime_ms=self.startup_config.get('motion_time_ms',4000)
        if (len(pose)!=6 or any(type(v) is not int or not lo<=v<=hi for v,(lo,hi) in zip(pose,HARD_LIMITS))
                or type(runtime_ms) is not int or not 1000<=runtime_ms<=5000):
            raise ValueError('Неверная настройка исходной позы robotio')
        if not self.lock.acquire(False):raise ValueError('Рука занята')
        record=None
        try:
            state=json.loads((self.root/'data/status.json').read_text())
            stationary_status(state,time.time())
            if source=='automatic_factory_startup' and self.startup_config.get('requires_latched_stop',True) and state.get('stop_latched') is not True:
                raise ValueError('Для автоподготовки нужен включённый STOP')
            if not self.pub.get_subscription_count():raise ValueError('Контроллер руки не подключён')
            for shape in (0.,-.2,-.4,-.6,-.8):
                if not self.model().path(pose[:5],pose[:5],shape)['valid']:
                    raise ValueError('Исходная поза пересекает модель робота или пол')
            message=ArmJoints(time=runtime_ms)
            for index,value in enumerate(pose,1):setattr(message,'joint'+str(index),value)
            record=dict(at=time.time(),servo_deg=list(pose),boot_id=self.boot,ends_monotonic=time.monotonic()+runtime_ms/1000,
                        runtime_ms=runtime_ms,source=source,measured=False,attained=False,publish_count=1,
                        startup_reference=True,command_source='automatic' if source=='automatic_factory_startup' else 'operator')
            self.write(record,'command_in_progress');self.pub.publish(message)
            if not self.pub.wait_for_all_acked(Duration(seconds=1)):
                raise ValueError('Нет подтверждения доставки исходной команды')
            while time.monotonic()<record['ends_monotonic']+.15:
                stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time());time.sleep(.025)
            record.update(dds_acknowledged=True,command_completed=True)
            self.write(record,'command_elapsed_observation_required');self.error=None
            result=dict(record,motion_sent=True,reference_source=source,attainment_measured=False)
            path=self.root/'data/arm-startup.json';temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(result));temporary.replace(path)
            return result
        except Exception:
            if record:self.write(record,'monitor_failed_state_unknown')
            raise
        finally:self.lock.release()

    def reference(self):
        stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time())
        state=json.loads((self.root/'data/arm-state.json').read_text())
        if state.get('boot_id')!=self.boot or state.get('phase')!='command_elapsed_observation_required':
            raise ValueError('После включения сначала подготовьте руку в свободном пространстве')
        # Any observed MCU link fault invalidates the estimate until a new observed reference.
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
                    step_deg=10,runtime_ms=None,continuous_motion=True,timed_path_supported=True,
                    controller_protocol='factory_micro_ros',reference_source=state.get('source'),
                    automatic_startup_home=self.startup_config.get('enabled') is True,
                    startup_pose_deg=self.startup_config.get('pose_deg',[90]*6),
                    torque_disable_available=False)

    def accept_reference(self,pose,observed):
        """Register a physically confirmed factory pose without moving servos."""
        if observed is not True:raise ValueError('Подтвердите фактическую исходную позу')
        if len(pose)!=6 or any(type(v) is not int or not lo<=v<=hi for v,(lo,hi) in zip(pose,HARD_LIMITS)):
            raise ValueError('Нужны шесть углов в диапазоне приводов')
        if not self.lock.acquire(False):raise ValueError('Рука движется')
        try:
            stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time())
            if not self.pub.get_subscription_count():raise ValueError('Контроллер руки не подключён')
            for shape in (0.,-.2,-.4,-.6,-.8):
                if not self.model().path(pose[:5],pose[:5],shape)['valid']:raise ValueError('Исходная поза пересекает робота или пол')
            record=dict(at=time.time(),servo_deg=list(pose),boot_id=self.boot,
                        source='operator_observed_reference',measured=False,attained=False,
                        ends_monotonic=time.monotonic(),runtime_ms=0,publish_count=0)
            self.write(record,'command_elapsed_observation_required')
            self.error=None
            return dict(record,motion_sent=False)
        finally:self.lock.release()

    def write(self,record,phase):
        path=self.root/'data/arm-state.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(record,phase=phase,updated_at=time.time())));tmp.replace(path)

    def stop(self):
        self.stop_revision+=1
        self.cancelled.set()
        return dict(no_further_steps=True,commanded_step_duration_ms=self.active_runtime_ms,physical_stop_latency_verified=False,
                    hardware_emergency_stop=False)

    def move(self,start,goal,deadline=None,expected_stop_revision=None,execution_permit=None,source="operator"):
        if source not in ('operator','supervised_policy','supervised_trajectory','local_mission'):raise ValueError('Неверный источник команды')
        if source=='local_mission' and not callable(execution_permit):raise ValueError('Нет разрешения локальной миссии')
        if not self.ready:raise ValueError('Сначала дождитесь подготовки геометрии руки')
        if len(start)!=6 or len(goal)!=6 or any(type(v) is not int for v in start+goal):raise ValueError('Нужны шесть целых углов')
        if not any(a!=b for a,b in zip(start,goal)):raise ValueError('Нулевой шаг')
        max_step=10 if source=='operator' else 2
        for a,b,(lo,hi) in zip(start,goal,HARD_LIMITS):
            if not lo<=a<=hi or not lo<=b<=hi or abs(a-b)>max_step:raise ValueError('Шаг превышает допустимый размер или предел сустава')
        runtime_ms=max(150,round(max(abs(a-b) for a,b in zip(start,goal))/10*1000))
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
                record=dict(at=time.time(),servo_deg=goal,runtime_ms=runtime_ms,boot_id=self.boot,
                            ends_monotonic=time.monotonic()+runtime_ms/1000,source='commanded_only',measured=False,
                            attained=False,observed_clear=True,publish_count=1,operator_step=source=="operator",command_source=source)
                msg=ArmJoints(time=runtime_ms)
                for i,value in enumerate(goal,1):setattr(msg,'joint'+str(i),value)
                self.write(record,'command_in_progress');self.pub.publish(msg);sent=True
                self.active_runtime_ms=runtime_ms
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
        finally:
            self.active_runtime_ms=0
            self.lock.release()

    def execute_path(self,start,goal,trajectory,permit,revision):
        from factory_trajectory import compile_path
        if not self.lock.acquire(False):raise ValueError('Рука занята')
        record=None;last_end=0.;sent=0;process_lock=None
        try:
            process_lock=(self.root/'data/arm-commissioning.lock').open('a')
            fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if self.reference()['servo_deg']!=start:raise ValueError('Исходная поза изменилась')
            path=compile_path(start,goal,trajectory,self.model())
            if self.stop_revision!=revision:raise ValueError('Путь отменён во время расчёта')
            self.cancelled.clear();permit()
            begun=time.monotonic()
            for item in path['commands']:
                target=begun+item['at']
                while time.monotonic()<target:
                    permit()
                    stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time())
                    time.sleep(min(.02,max(0.,target-time.monotonic())))
                permit()
                if self.cancelled.is_set() or self.stop_revision!=revision:raise ValueError('Путь отменён')
                if time.monotonic()-target>.15:raise ValueError('Сбой расписания траектории; догоняющие команды отменены')
                stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time())
                if not self.pub.get_subscription_count():raise ValueError('Нет контроллера руки')
                message=ArmJoints(time=item['runtime_ms'])
                for i,value in enumerate(item['pose'],1):setattr(message,'joint'+str(i),value)
                last_end=time.monotonic()+item['runtime_ms']/1000
                record=dict(at=time.time(),servo_deg=item['pose'],boot_id=self.boot,ends_monotonic=last_end,
                            runtime_ms=item['runtime_ms'],source='timed_factory_command_estimate',measured=False,attained=False,
                            trajectory_sha256=path['source_sha256'],full_moveit_path_retained=path['full_moveit_path_retained'])
                self.write(record,'command_in_progress');self.pub.publish(message);sent+=1
                self.active_runtime_ms=item['runtime_ms']
            while time.monotonic()<last_end+.05:
                permit();stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time());time.sleep(.01)
            if not self.pub.wait_for_all_acked(Duration(seconds=1)):raise ValueError('Нет DDS подтверждения последней команды')
            permit();record.update(command_completed=True,publish_count=sent,duration_s=path['duration'])
            self.write(record,'command_elapsed_observation_required');self.error=None
            return record
        except Exception as exc:
            self.error=str(exc)
            if record:
                # A factory servo finishes the already accepted finite segment.
                # Do not mistake cancellation for a measured reached position.
                remaining=last_end+.1-time.monotonic()
                if remaining>0:time.sleep(remaining)
                try:
                    stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time())
                    acknowledged=self.pub.wait_for_all_acked(Duration(seconds=.3))
                except Exception:acknowledged=False
                record.update(cancelled=True,command_completed=False,error=str(exc),publish_count=sent)
                self.write(record,'command_elapsed_observation_required' if acknowledged else 'monitor_failed_state_unknown')
            raise
        finally:
            if process_lock is not None:process_lock.close()
            self.active_runtime_ms=0
            self.lock.release()
