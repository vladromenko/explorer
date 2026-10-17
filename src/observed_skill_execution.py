"""Explicit observed execution of an independently qualified geometric skill.

Uses the existing mission, trajectory, camera and STOP owners. Construction,
candidate inspection and preflight never actuate or clear a stopped robot.
"""
import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np


class ObservedSkillExecution:
    def __init__(self,root,robot,emit,scans,scan_transform):
        self.root=Path(root);self.robot=copy.copy(robot);self.emit=emit;self.scans=scans;self.scan_transform=scan_transform
        self.lock=threading.Lock();self.cancelled=threading.Event();self.session=None;self.lease=0.
        self.target=np.zeros(3);self.target_at=0.;self.state=dict(phase='idle',physical_ready=False)
        self.model_root=self.root/'data/observed-models'

    def candidate(self,identifier):
        if not isinstance(identifier,str) or len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid observed skill candidate')
        path=self.root/'data/observed-skill-candidates'/identifier
        from simulator.observed_transfer import validate
        return path,validate(path/'package')

    def preflight(self,identifier):
        reasons=[];checked=None
        try:
            _,checked=self.candidate(identifier)
            from simulator.skill_workshop import execution_fingerprint
            if checked['policy']['controller_sha256']!=execution_fingerprint()['sha256']:
                reasons.append('Установленный исполнитель отличается от проверенного в симуляции')
            for relative,digest in checked['qualification']['controller']['files'].items():
                if relative.startswith('config/'):
                    path=self.root/relative
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                        reasons.append('Не совпадает геометрия или калибровка: '+relative)
            # Contact settings on the installed hardware are currently accepted
            # for a sock. A rigid cookie/cube cannot inherit that permission.
            if checked['policy']['task_id']!='sock_basket':reasons.append('На роботе ещё не принят контактный профиль этого предмета')
        except (OSError,ValueError,KeyError,TypeError) as error:reasons.append(str(error))
        for module in ('mujoco','gymnasium','scipy'):
            if importlib.util.find_spec(module) is None:reasons.append('Не установлена среда нового исполнителя: '+module)
        try:
            reasons+=self.robot.blockers()
            settings=self.robot._settings()
            plane=np.asarray(settings.get('floor_plane_base'),float)
            if plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:2])>.03 or plane[2]<.999:
                reasons.append('Нет принятой горизонтальной плоскости пола')
            if not 30<=settings['open_deg']<settings['close_deg']<=170:
                reasons.append('Принятое раскрытие захвата не совпадает с моделью 30–170 градусов')
            # A different aperture mapping changes contact geometry. It must
            # be calibrated and revalidated rather than silently normalized.
            if abs(settings['open_deg']-30)>.5 or abs(settings['close_deg']-170)>.5:
                reasons.append('Модель захвата требует отдельного сопоставления с измеренным раскрытием')
            status=json.loads((self.root/'data/status.json').read_text())
            if status.get('stop_latched') is not False or status.get('mode')!='AUTONOMOUS':
                reasons.append('Перед наблюдаемой попыткой выберите автономный режим и явно снимите STOP')
            if self.robot.missions.status().get('active'):reasons.append('Уже выполняется другая миссия')
            if self.robot.arm.status().get('busy') or self.robot.trajectory.status().get('busy'):reasons.append('Рука занята')
            scan=self.scans()
            if len(scan)!=2 or any(time.monotonic()-s['acquired_monotonic']>.6 for s in scan):reasons.append('Нет двух свежих полных сканов лидара')
        except (OSError,ValueError,KeyError,TypeError) as error:reasons.append(str(error))
        try:
            lock=json.loads((self.root/'config/grounding-model.json').read_text())
            for relative,digest in lock['files'].items():
                p=self.model_root/'models/grounding-dino-tiny'/relative
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Не установлена проверенная модель зрения')
            if not (self.model_root.parent/'env-vision/bin/python').is_file():reasons.append('Не установлена среда модели зрения')
        except (OSError,ValueError,KeyError) as error:reasons.append(str(error))
        return dict(ready=not reasons,blocked_by=list(dict.fromkeys(reasons)),candidate=identifier,
            simulation_ready=bool(checked),physical_ready=False,observed_trial_required=True)

    def status(self):
        return dict(self.state,busy=self.lock.locked(),session=self.session,
            observed_lease_s=max(0.,self.lease-time.monotonic()),physical_ready=False)

    def heartbeat(self,session,observing):
        if session!=self.session or not self.lock.locked():raise ValueError('Observed skill session not found')
        if observing is not True:return self.stop()
        self.lease=time.monotonic()+2.;return self.status()

    def start(self,identifier,observing):
        if observing is not True:raise ValueError('Нужен наблюдающий оператор')
        check=self.preflight(identifier)
        if not check['ready']:raise ValueError('; '.join(check['blocked_by']))
        if not self.lock.acquire(False):raise ValueError('Попытка уже выполняется')
        self.session=uuid.uuid4().hex;self.cancelled.clear();self.lease=time.monotonic()+2.
        self.state=dict(phase='starting',candidate=identifier,physical_ready=False)
        threading.Thread(target=self._run,args=(identifier,),daemon=True,name='observed-skill').start()
        return self.status()

    def _lease(self):
        if self.cancelled.is_set() or time.monotonic()>self.lease:raise ValueError('Наблюдение прекращено или нажата STOP')

    def stop(self):
        self.cancelled.set();self.target=np.zeros(3);self.target_at=0.
        errors=[]
        if self.lock.locked():
            for action in (lambda:self.emit('policy_drive',mission=self.session,velocity=[0.,0.,0.]),
                           self.robot.trajectory.stop,self.robot.arm.stop):
                try:action()
                except Exception as error:errors.append(str(error))
        return dict(stopping=True,session=self.session,stop_errors=errors)

    def _drive(self):
        while not self.cancelled.is_set():
            try:
                self._lease();self.robot.permit(self.session)
                velocity=self.target if time.monotonic()-self.target_at<.15 else np.zeros(3)
                self.emit('autonomy_lease',mission=self.session)
                self.emit('policy_drive',mission=self.session,velocity=velocity.tolist())
            except Exception:self.cancelled.set()
            time.sleep(.05)

    def _run(self,identifier):
        from dataclasses import asdict
        folder=self.root/'data/observed-skill-runs'/self.session;folder.mkdir(parents=True)
        try:
            self._lease();path,checked=self.candidate(identifier)
            self.robot.settings=self.robot._settings();self.robot.owner_check=self._lease
            self.robot.mid=self.session;self.robot.boot=self.robot.arm.reference()['boot_id'];self.robot.tracking=False
            self.robot.missions.begin_compound(self.session,'mobile_policy',900,self._lease)
            self.robot.hold();threading.Thread(target=self._drive,daemon=True).start()
            from simulator.observed_robot import run_candidate
            transport=FactoryObservedTransport(self)
            result=run_candidate(path/'package',transport,folder/'execution',model_root=self.model_root,
                mesh_dir=path/'package/assets/meshes',progress=lambda row:self.state.update(phase=row.get('phase','running')))
            self.state.update(phase='completed' if result.state=='succeeded' else 'stopped',
                result=asdict(result),visual_outcome_confirmed=result.state=='succeeded',
                physical_success_verified=False,physical_ready=False)
        except Exception as error:self.state.update(phase='stopped',reason=str(error),physical_ready=False)
        finally:
            self.stop()
            try:
                if self.robot.mid==self.session:
                    self.robot.missions.finish(self.session,'succeeded' if self.state.get('phase')=='completed' else 'interrupted',dict(reason=self.state.get('reason')))
                    self.robot.mid=None
            finally:
                try:(folder/'result.json').write_text(json.dumps(self.state,ensure_ascii=False,indent=2))
                finally:self.lock.release()


class FactoryObservedTransport:
    def __init__(self,owner):self.owner=owner;self.robot=owner.robot;self.origin=None

    def cancelled(self):return self.owner.cancelled.is_set()
    def permit(self):self.owner._lease();self.robot.permit(self.owner.session)
    def preflight(self,checked):self.permit()
    def stop(self):
        self.owner.target=np.zeros(3);self.owner.target_at=0.
        self.owner.emit('policy_drive',mission=self.owner.session,velocity=[0.,0.,0.])

    def _base(self,status):
        pose=status['raw_pose'];yaw=float(pose['yaw']);c,s=np.cos(yaw),np.sin(yaw)
        result=np.eye(4);result[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        plane=self.robot.settings['floor_plane_base'];result[:3,3]=[pose['x'],pose['y'],plane[3]/plane[2]]
        if self.origin is None:
            anchor=result.copy();anchor[2,3]=0.;self.origin=np.linalg.inv(anchor)
        return self.origin@result

    def state(self):
        self.permit();status=json.loads((self.owner.root/'data/status.json').read_text())
        from arm_commissioning import coordinated_policy_status
        coordinated_policy_status(status,time.time())
        arm=(self.robot.arm.reference(coordinated_policy_status) if getattr(self.robot.arm,'factory_timed',False)
             else self.robot.arm.reference(expected_boot=self.robot.boot))
        digest=hashlib.sha256(json.dumps(self.robot.settings,sort_keys=True).encode()).hexdigest()
        acquired=time.monotonic()-(time.time()-status['at'])
        return dict(acquired_monotonic=acquired,servo_deg=arm['servo_deg'],body_velocity=status['odom_velocity'],
            boot_id=self.robot.boot,map_epoch=self.robot.missions.maps.epoch(),calibration_sha256=digest,
            joint_state_source='measured' if arm.get('measured') is True else 'accepted_command_estimate',
            floor_transform_validated=True,base_to_world=self._base(status).tolist())

    def apply(self,action,*,reference,period):
        self.permit();action=np.asarray(action,float)
        if np.any(abs(action[6:])>np.array([.12,.12,.35])):raise ValueError('Скорость не входит в принятый диапазон наблюдаемой попытки')
        if np.max(abs(action[:6]-reference[:6]))>.3:
            if np.any(abs(action[6:])>.001):raise ValueError('Изменение руки и перемещение базы должны быть разделены')
            self.stop();self.robot._move(action[:6].tolist())
        self.owner.target=action[6:].copy();self.owner.target_at=time.monotonic()
        if np.any(abs(action[6:])>.001):self.robot.missions.emit('resume_base',mission=self.owner.session)
        time.sleep(period);self.permit();return self.state()

    def capture(self,after_frame=None):
        self.permit();sample=self.robot.vision.wait_snapshot(self.permit,after_stamp=time.time())
        transform,_,angles=self.robot.vision.geometry(sample,self.robot.settings)
        state=self.state();exposure=float(sample['stamp']);stamp=time.monotonic()-(time.time()-exposure)
        if abs(state['acquired_monotonic']-stamp)>.075:raise ValueError('Положение базы и RGB-D не синхронизированы')
        state['servo_deg']=list(angles)
        k=np.asarray(sample['k']);h,w=sample['depth'].shape
        if np.max(abs(np.asarray(sample['d'])))>1e-6:raise ValueError('Новому исполнителю нужен выпрямленный зарегистрированный RGB-D')
        return dict(frame_id=hashlib.sha256(str(exposure).encode()).hexdigest()[:32],registered=True,arm_stationary=True,depth_unit='metre',
            acquired_monotonic=stamp,robot_state=state,rgb=sample['rgb'],depth=sample['depth'],
            optical_to_world=(np.asarray(state['base_to_world'])@transform).tolist(),
            intrinsics=dict(width=w,height=h,fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2])))

    def scan(self,scanner):
        self.permit();scan=dict(self.owner.scans()[scanner]);state=self.state()
        from scipy.spatial.transform import Rotation
        translation,quaternion=self.owner.scan_transform(scan['frame'])
        sensor=np.eye(4);sensor[:3,3]=translation;sensor[:3,:3]=Rotation.from_quat(quaternion).as_matrix()
        scan['sensor_to_world']=(np.asarray(state['base_to_world'])@sensor).tolist()
        return scan


class FullScanCache:
    """Navigation uses complete measured scans, never downsampled display data."""
    def __init__(self,clock=time.monotonic):self.lock=threading.Lock();self.values={};self.clock=clock

    def receive(self,index,message,source_now):
        stamp=message.header.stamp.sec+message.header.stamp.nanosec/1e9
        if not -.1<=source_now-stamp<.6:return
        ranges=[float(v) if np.isfinite(v) and message.range_min<float(v)<message.range_max else None for v in message.ranges]
        value=dict(frame=message.header.frame_id,ranges=ranges,angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),acquired_monotonic=self.clock()-max(0.,source_now-stamp))
        with self.lock:self.values[index]=value

    def snapshot(self):
        with self.lock:return [dict(self.values[i]) for i in (0,1)] if len(self.values)==2 else []
