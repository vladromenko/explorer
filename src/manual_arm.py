"""Single persistent publisher for finite paths and bounded velocity intents.

Servo state remains a command estimate. Already accepted robotio endpoints
finish independently of the host; cancellation prevents further publications.
"""
import fcntl
import json
import math
from pathlib import Path
import threading
import time
from arm_msgs.msg import ArmJoints
from rclpy.duration import Duration
from arm_commissioning import HARD_LIMITS,stationary_status,coordinated_status,coordinated_policy_status

class StreamCancelled(ValueError):
    """Expected local cancellation, distinct from a planner/transport fault."""

class ManualArm:
    factory_timed=True
    supports_velocity_stream=True
    def __init__(self,root,node,model):
        self.root=Path(root);self.model=model;self.pub=node.create_publisher(ArmJoints,'/arm6_joints',1)
        self.lock=threading.Lock();self.cancelled=threading.Event();self.ready=False;self.error=None
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.gamepad_permit=lambda:False
        self.stop_revision=0
        self.active_runtime_ms=0
        self.active_path_timing=None
        self.publication_lock=threading.Lock()
        self.stream_lock=threading.Lock()
        self.stream_intent=None
        self.stream_thread=None
        self.streaming_active=False
        self.stream_generation=None
        try:self.startup_config=json.loads((self.root/'config/factory-arm-startup.json').read_text())
        except (OSError,ValueError):self.startup_config={}
        try:self.motion_config=json.loads((self.root/'config/arm-motion.json').read_text())
        except (OSError,ValueError):self.motion_config={}
        if self.startup_config.get('enabled') is True:
            threading.Thread(target=self._automatic_startup_home,daemon=True).start()

    def _uptime(self):
        return float(Path('/proc/uptime').read_text().split()[0])

    def _automatic_startup_home(self):
        """Establish a known factory pose once near a full Jetson boot.

        A later web-service restart must not move the arm unexpectedly.
        """
        try:
            delay=float(self.startup_config.get('delay_after_web_start_s',8))
            window=float(self.startup_config.get('startup_window_s',180))
            time.sleep(max(1,min(delay,30)))
            # Power and MCU telemetry commonly become valid several seconds
            # after the web service.  Wait for the complete precondition set
            # instead of spending the single startup attempt on UNKNOWN power.
            while True:
                uptime=self._uptime()
                if not 0<=uptime<=window:return
                try:
                    state=json.loads((self.root/'data/status.json').read_text())
                    stationary_status(state,time.time())
                    if self.startup_config.get('requires_latched_stop',True) and state.get('stop_latched') is not True:
                        raise ValueError('Для автоподготовки нужен включённый STOP')
                    if not self.pub.get_subscription_count():raise ValueError('Контроллер руки ещё запускается')
                except (OSError,ValueError,KeyError) as exc:
                    self.error='Автоподготовка ожидает готовности: '+str(exc)
                    time.sleep(.5);continue
                self.home_reference(False,'automatic_factory_startup')
                return
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

    def reference(self,status_check=None):
        status_check=status_check or stationary_status
        status_check(json.loads((self.root/'data/status.json').read_text()),time.time())
        state=json.loads((self.root/'data/arm-state.json').read_text())
        live_stream=(self.streaming_active and state.get("phase")=="command_in_progress"
            and state.get("motion_profile")=="ruckig_community_velocity"
            and state.get("command_generation")==self.stream_generation)
        if state.get("boot_id")!=self.boot or (state.get("phase")!="command_elapsed_observation_required" and not live_stream):
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
        with self.stream_lock:
            stream_status_check=coordinated_status if self.streaming_active and (self.stream_intent or {}).get("coordinated") else stationary_status
        try:state=self.reference(stream_status_check);blocked=None
        except (OSError,KeyError,ValueError) as exc:state={};blocked=str(exc)
        return dict(ready=self.ready,blocked_by=blocked,servo_deg=state.get('servo_deg'),
                    estimated_only=True,busy=self.lock.locked(),error=self.error,
                    step_deg=10,runtime_ms=None,continuous_motion=True,timed_path_supported=True,
                    controller_protocol='factory_micro_ros',reference_source=state.get('source'),
                    motion_profile=self.motion_config.get('profile','coordinated_quintic_lookahead'),
                    velocity_stream_supported=True,velocity_stream_active=self.streaming_active,
                    velocity_stream_backend="ruckig_community_velocity",
                    velocity_stream_limits=self.motion_config.get("streaming",{}),
                    active_path_timing=self.active_path_timing,
                    automatic_startup_home=self.startup_config.get('enabled') is True,
                    startup_pose_deg=self.startup_config.get('pose_deg',[90]*6),
                    torque_disable_available=False)

    def _publish_command(self,pose,runtime_ms,record,phase="command_in_progress",revision=None):
        """One transport boundary for finite paths and live velocity streams."""
        if len(pose)!=6 or any(type(value) is not int or not low<=value<=high
                for value,(low,high) in zip(pose,HARD_LIMITS)):
            raise ValueError("Invalid quantized arm endpoint")
        if type(runtime_ms) is not int or not 20<=runtime_ms<=5000:
            raise ValueError("Invalid arm endpoint duration")
        with self.publication_lock:
            if revision is not None and (self.stop_revision!=revision or self.cancelled.is_set()):
                raise ValueError("Arm command was cancelled before publication")
            if not self.pub.get_subscription_count():
                raise ValueError("Контроллер руки не подключён")
            message=ArmJoints(time=runtime_ms)
            for index,value in enumerate(pose,1):setattr(message,"joint"+str(index),value)
            stamp=time.time()
            record.update(at=stamp,command_sent_at=stamp,servo_deg=list(pose),q_commanded_deg=list(pose),
                runtime_ms=runtime_ms,boot_id=self.boot,measured=False,attained=False)
            self.write(record,phase)
            self.pub.publish(message)
            self.active_runtime_ms=runtime_ms

    def stream_velocity(self,joint_velocity,xyz_velocity,generation,valid_until,permit,owner="operator",coordinated=True):
        """Replace the latest intent immediately; never queue position deltas."""
        from arm_velocity import finite_vector
        joints=finite_vector(joint_velocity,6,"joint intent").tolist()
        xyz=finite_vector(xyz_velocity,3,"Cartesian intent").tolist()
        if (not callable(permit) or type(generation) is not int or not math.isfinite(valid_until)
                or not isinstance(owner,str) or not owner or len(owner)>80 or type(coordinated) is not bool):
            raise ValueError("Invalid live arm lease")
        if not self.ready:raise ValueError("Сначала дождитесь подготовки геометрии руки")
        now=time.monotonic()
        if valid_until<=now:raise ValueError("Arm intent expired")
        moving=any(abs(value)>1e-8 for value in joints+xyz)
        with self.stream_lock:
            previous=self.stream_intent
            if self.streaming_active and previous and previous["owner"]!=owner:
                if moving:raise ValueError("Рука занята другим владельцем скоростного управления")
                return dict(accepted=False,streaming=True,owner=previous["owner"],neutral_ignored=True)
            pulse=(previous or {}).get("pulse") if previous and previous["generation"]==generation else None
            previously_moving=bool(previous and any(abs(value)>1e-8 for value in previous["joints"]+previous["xyz"]))
            if moving and (not previously_moving or pulse is not None):
                # A not-yet-consumed brief press follows the latest direction;
                # a reversal must never replay the first direction on release.
                pulse=dict(joints=joints,xyz=xyz,at=now)
            self.stream_intent=dict(joints=joints,xyz=xyz,generation=generation,revision=self.stop_revision,
                valid_until=valid_until,permit=permit,owner=owner,at=now,received_at=time.time(),
                pulse=pulse,coordinated=bool(coordinated))
            if moving and not self.streaming_active:
                self.streaming_active=True
                self.stream_generation=generation
                self.stream_thread=threading.Thread(target=self._velocity_worker,args=(generation,self.stop_revision),daemon=True)
                self.stream_thread.start()
        return dict(accepted=True,streaming=self.streaming_active,generation=generation,
            latest_intent=True,measured=False)

    def stop_stream(self,owner,generation):
        with self.stream_lock:
            intent=self.stream_intent
            active=bool(self.streaming_active and intent and intent["owner"]==owner
                and intent["generation"]==generation)
            return self.stop() if active else dict(no_further_steps=True,owner_was_active=False)

    def _velocity_worker(self,generation,revision):
        from arm_velocity import VelocityGenerator,cartesian_joint_velocity
        process_lock=None;record=None;sent=0;last_end=0.0;acquired=False;failed=False;cancel_reason=None
        with self.stream_lock:last_intent_at=(self.stream_intent or {}).get("at",-math.inf)
        try:
            if not self.lock.acquire(False):raise ValueError("Рука занята конечной траекторией")
            acquired=True
            process_lock=(self.root/"data/arm-commissioning.lock").open("a")
            fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.stream_lock:
                starting=dict(self.stream_intent or {})
            status_check=coordinated_status if starting.get("coordinated") else stationary_status
            current=self.reference(status_check)
            config=self.motion_config.get("streaming",{})
            generator=VelocityGenerator(current["servo_deg"],config)
            self.cancelled.clear()
            next_sample=time.monotonic()
            previous=list(current["servo_deg"])
            while True:
                with self.stream_lock:
                    intent=dict(self.stream_intent or {})
                    last_intent_at=intent.get("at",-math.inf)
                    pulse=intent.get("pulse")
                    if pulse is not None:self.stream_intent["pulse"]=None
                now=time.monotonic()
                if (self.cancelled.is_set() or self.stop_revision!=revision or not intent
                        or intent["generation"]!=generation or intent["revision"]!=revision
                        or now>=intent["valid_until"] or intent["permit"]() is False):
                    raise StreamCancelled("Arm velocity stream cancelled or lease expired")
                status_check(json.loads((self.root/"data/status.json").read_text()),time.time())
                if now-next_sample>max(0.10,2*generator.period):
                    raise ValueError("Arm stream missed its schedule; stale intent discarded")
                active=intent
                if pulse is not None and not any(abs(value)>1e-8 for value in intent["joints"]+intent["xyz"]):
                    if now-pulse["at"]<=min(0.15,intent["valid_until"]-pulse["at"]):
                        active=dict(intent,**pulse)
                velocity=active["joints"]
                if any(abs(value)>1e-8 for value in active["xyz"]):
                    velocity=cartesian_joint_velocity(self.model(),generator.position,active["xyz"],
                        velocity,generator.maximum)
                sample=generator.step(velocity)
                if sample["changed"]:
                    pose=sample["pose"]
                    for shape in (0.0,-0.2,-0.4,-0.6,-0.8):
                        if not self.model().path(previous[:5],pose[:5],shape)["valid"]:
                            raise ValueError("Arm streaming path collides with robot or floor")
                    if intent["permit"]() is False or time.monotonic()>=intent["valid_until"]:
                        raise StreamCancelled("Arm intent expired during geometry check")
                    last_end=time.monotonic()+sample["runtime_ms"]/1000.0
                    record=dict(sample,ends_monotonic=last_end,source="timed_factory_command_estimate",
                        motion_profile="ruckig_community_velocity",command_source="coordinated_operator" if intent["owner"] in ("keyboard","gamepad") else intent["owner"],
                        owner=intent["owner"],input_source=intent["owner"],command_generation=generation,
                        input_received_at=intent["received_at"],intent_to_publish_s=time.monotonic()-intent["at"],
                        compute_s=time.monotonic()-now,
                        publish_count=sent+1,command_completed=False,operator_step=True,
                        endpoint_replacement_physically_verified=config.get("endpoint_replacement_physically_verified") is True)
                    self._publish_command(pose,sample["runtime_ms"],record,revision=revision)
                    previous=list(pose);sent+=1
                if sample["settled"] and not any(abs(value)>1e-8 for value in active["joints"]+active["xyz"]):
                    break
                next_sample+=generator.period
                remaining=next_sample-time.monotonic()
                if remaining>0:self.cancelled.wait(remaining)
            self.error=None
        except StreamCancelled as exc:
            failed=True;cancel_reason=str(exc);self.error=None
        except Exception as exc:
            failed=True;self.error=str(exc)
        finally:
            # STOP adds no pose, HOME, torque or gripper commands. Only the
            # already accepted bounded endpoint may still finish on robotio.
            if record is not None:
                remaining=last_end+0.02-time.monotonic()
                if remaining>0:time.sleep(remaining)
                try:acknowledged=self.pub.wait_for_all_acked(Duration(seconds=0.2))
                except Exception:acknowledged=False
                record.update(command_completed=not failed,cancelled=failed,publish_count=sent,
                    dds_acknowledged=bool(acknowledged),measured=False,attained=False,
                    error=self.error if failed else None,cancel_reason=cancel_reason)
                self.write(record,"command_elapsed_observation_required" if acknowledged else "monitor_failed_state_unknown")
            if process_lock is not None:process_lock.close()
            with self.stream_lock:
                if acquired:self.lock.release()
                self.active_runtime_ms=0
                self.streaming_active=False;self.stream_generation=None
                pending=self.stream_intent or {}
                # Input can arrive while the final bounded endpoint/ACK is
                # draining. Transfer only a newer live intent, never the old
                # generation cancelled by STOP or a stale held command.
                if (pending.get("at",-math.inf)>last_intent_at
                        and pending.get("revision") == self.stop_revision
                        and time.monotonic()<pending.get("valid_until",0.0)
                        and (pending.get("pulse") is not None
                            or any(abs(value)>1e-8 for value in pending.get("joints",[])+pending.get("xyz",[])))):
                    self.streaming_active=True
                    self.stream_generation=pending["generation"]
                    self.stream_thread=threading.Thread(target=self._velocity_worker,
                        args=(pending["generation"],self.stop_revision),daemon=True)
                    self.stream_thread.start()

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
        with self.publication_lock:
            self.stop_revision+=1
            self.cancelled.set()
        return dict(no_further_steps=True,commanded_step_duration_ms=self.active_runtime_ms,physical_stop_latency_verified=False,
                    hardware_emergency_stop=False)

    def move(self,start,goal,deadline=None,expected_stop_revision=None,execution_permit=None,source="operator",speed='normal'):
        if source not in ('operator','coordinated_operator','supervised_policy','supervised_mobile_policy','supervised_trajectory','local_mission'):raise ValueError('Неверный источник команды')
        if source=='local_mission' and not callable(execution_permit):raise ValueError('Нет разрешения локальной миссии')
        if speed not in ('precision','normal','teleop','teleop_fast','fast'):raise ValueError('Неизвестная скорость руки')
        if not self.ready:raise ValueError('Сначала дождитесь подготовки геометрии руки')
        if len(start)!=6 or len(goal)!=6 or any(type(v) is not int for v in start+goal):raise ValueError('Нужны шесть целых углов')
        if not any(a!=b for a,b in zip(start,goal)):raise ValueError('Нулевой шаг')
        max_step=10 if source in ('operator','coordinated_operator') else 2
        for a,b,(lo,hi) in zip(start,goal,HARD_LIMITS):
            if not lo<=a<=hi or not lo<=b<=hi or abs(a-b)>max_step:raise ValueError('Шаг превышает допустимый размер или предел сустава')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        sent=0;record=None;last_end=0.
        status_check=coordinated_status if source=='coordinated_operator' else coordinated_policy_status if source=='supervised_mobile_policy' else stationary_status
        try:
            if expected_stop_revision is not None and expected_stop_revision!=self.stop_revision:
                raise ValueError('Команда отменена во время планирования')
            publication_revision=self.stop_revision
            self.cancelled.clear()
            with (self.root/'data/arm-commissioning.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                current=self.reference(status_check)
                if current['servo_deg']!=start:raise ValueError('Исходное положение команды изменилось')
                for shape in (0.,-.2,-.4,-.6,-.8):
                    if not self.model().path(start[:5],goal[:5],shape)['valid']:raise ValueError('MoveIt: столкновение с роботом или полом')
                from factory_trajectory import compile_path,motion_profile
                motion=motion_profile(self.motion_config)
                teleop=self.motion_config.get('teleop',{})
                factors={'precision':(.65,.65,.65),'normal':(1.,1.,1.),
                         'teleop':(float(teleop.get('velocity_scale',4.)),float(teleop.get('acceleration_scale',8.)),float(teleop.get('jerk_scale',16.))),
                         'teleop_fast':(float(teleop.get('velocity_scale',4.))*2,float(teleop.get('acceleration_scale',8.))*2,float(teleop.get('jerk_scale',16.))*2),
                         'fast':(32.,256.,2048.)}
                for key,factor in zip(('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3'),factors[speed]):
                    motion[key]=[float(v)*factor for v in motion[key]]
                if speed=='fast':
                    motion['min_duration_s']=float(motion['min_duration_s'])/8
                    motion['publish_period_s']=max(.04,float(motion['publish_period_s'])/2)
                    motion['lookahead_s']=max(motion['publish_period_s'],float(motion['lookahead_s'])*.2)
                if speed in ('teleop','teleop_fast'):
                    motion['min_duration_s']=max(.06,float(teleop.get('min_duration_s',.12))*(.67 if speed=='teleop_fast' else 1.))
                    motion['publish_period_s']=float(teleop.get('publish_period_s',.04))
                    motion['lookahead_s']=float(teleop.get('lookahead_s',.08))
                path=compile_path(start,goal,None,self.model(),motion)
                self.reference(status_check)
                if self.cancelled.is_set() or (expected_stop_revision is not None and expected_stop_revision!=self.stop_revision) or deadline is not None and time.monotonic()>deadline:
                    raise ValueError('Команда отменена или истекла до отправки')
                if deadline is not None and not self.gamepad_permit():
                    raise ValueError('Кнопка разрешения отпущена или панель отключена')
                if execution_permit is not None:execution_permit()
                if not self.pub.get_subscription_count():raise ValueError('Контроллер руки не подключён')
                begun=time.monotonic()
                for item in path['commands']:
                    target=begun+item['at']
                    while time.monotonic()<target:
                        if self.cancelled.is_set():raise ValueError('Команда отменена')
                        status_check(json.loads((self.root/'data/status.json').read_text()),time.time())
                        time.sleep(min(.02,max(0.,target-time.monotonic())))
                    if self.cancelled.is_set() or (expected_stop_revision is not None and expected_stop_revision!=self.stop_revision):
                        raise ValueError('Команда отменена')
                    if deadline is not None and not self.gamepad_permit():
                        raise ValueError('Кнопка разрешения отпущена или панель отключена')
                    if execution_permit is not None:execution_permit()
                    if time.monotonic()-target>.15:raise ValueError('Сбой расписания плавного движения')
                    last_end=time.monotonic()+item['runtime_ms']/1000
                    record=dict(at=time.time(),servo_deg=item['pose'],runtime_ms=item['runtime_ms'],boot_id=self.boot,
                                ends_monotonic=last_end,source='timed_factory_command_estimate',measured=False,
                                attained=False,observed_clear=True,publish_count=sent+1,operator_step=source in ("operator","coordinated_operator"),
                                command_source=source,trajectory_sha256=path['source_sha256'],motion_profile=path['profile'],speed=speed)
                    self._publish_command(item["pose"],item["runtime_ms"],record,revision=publication_revision);sent+=1
                if not self.pub.wait_for_all_acked(Duration(seconds=1)):
                    raise ValueError('Нет подтверждения доставки; положение неизвестно')
                while time.monotonic()<last_end+.05:
                    status_check(json.loads((self.root/'data/status.json').read_text()),time.time())
                    time.sleep(.025)
                record.update(dds_acknowledged=True,command_completed=True,publish_count=sent,duration_s=path['duration'])
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
        self.active_path_timing=None
        try:
            process_lock=(self.root/'data/arm-commissioning.lock').open('a')
            fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if self.reference()['servo_deg']!=start:raise ValueError('Исходная поза изменилась')
            compile_started=time.monotonic()
            path=compile_path(start,goal,trajectory,self.model(),self.motion_config)
            self.active_path_timing=dict(compiled_duration_s=path["duration"],
                source_duration_s=path.get("source_duration_s"),
                time_scale=path.get("time_scale"),
                timing_limiter=path.get("timing_limiter"),
                source_waypoint_count=path.get("source_waypoint_count"),
                retiming_algorithm=path.get("retiming_algorithm"),
                original_moveit_sha256=path.get("original_moveit_sha256"),
                compile_duration_s=time.monotonic()-compile_started,
                generated_targets=len(path["commands"]), started_at=time.time(),
                execution_elapsed_s=0.0, running=True)
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
                last_end=time.monotonic()+item['runtime_ms']/1000
                record=dict(at=time.time(),servo_deg=item['pose'],boot_id=self.boot,ends_monotonic=last_end,
                            runtime_ms=item['runtime_ms'],source='timed_factory_command_estimate',measured=False,attained=False,
                            trajectory_sha256=path['source_sha256'],full_moveit_path_retained=path['full_moveit_path_retained'])
                self.active_path_timing["execution_elapsed_s"]=time.monotonic()-begun
                record["path_timing"]=dict(self.active_path_timing)
                self._publish_command(item["pose"],item["runtime_ms"],record,revision=revision);sent+=1
            while time.monotonic()<last_end+.05:
                permit();stationary_status(json.loads((self.root/'data/status.json').read_text()),time.time());time.sleep(.01)
            if not self.pub.wait_for_all_acked(Duration(seconds=1)):raise ValueError('Нет DDS подтверждения последней команды')
            permit();record.update(command_completed=True,publish_count=sent,duration_s=path['duration'])
            self.active_path_timing.update(running=False,execution_elapsed_s=time.monotonic()-begun)
            record["path_timing"]=dict(self.active_path_timing)
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
                if self.active_path_timing:
                    self.active_path_timing.update(running=False,error=str(exc),
                        execution_elapsed_s=time.monotonic()-begun)
                    record["path_timing"]=dict(self.active_path_timing)
                self.write(record,'command_elapsed_observation_required' if acknowledged else 'monitor_failed_state_unknown')
            raise
        finally:
            if process_lock is not None:process_lock.close()
            if self.active_path_timing:self.active_path_timing["running"]=False
            self.active_runtime_ms=0
            self.lock.release()
