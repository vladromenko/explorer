"""Record a whole operator-driven journey, with explicit task-stage labels.

State/action provenance stays commanded-only. This recorder cannot drive a robot
and its episodes are not silently mixed into the six-joint ACT training set.
"""
import json
import threading
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from lerobot_bridge import write_json

STAGES=('travel_to_object','grasp','carry','place')


def sample(root,now):
    state=json.loads((root/'data/status.json').read_text())
    if not 0<=now-state.get('at',0)<1:raise ValueError('Нет свежего состояния шасси')
    if state.get('mode')!='MANUAL':raise ValueError('Показ записывается только при ручном управлении')
    if state.get('power',{}).get('state','UNKNOWN') in ('LOW_POWER','CRITICAL','UNKNOWN','CHARGING'):
        raise ValueError('Запись остановлена из-за питания')
    contract=state.get('arm_state') or {}
    command_mode=contract.get('estimated_only') is True
    arm=contract if command_mode else state.get('arm_command_state',{})
    angles=arm.get('servo_deg')
    if (command_mode and not arm.get('reference_valid') or not command_mode and
        arm.get('phase') not in ('command_in_progress','command_elapsed_observation_required')):
        raise ValueError('Нет достоверной истории команд руки')
    fault=root/'data/arm-telemetry-fault.json'
    if not command_mode and fault.exists() and json.loads(fault.read_text()).get('at',0)>arm.get('at',0):
        raise ValueError('После потери связи заново подготовьте руку')
    action=np.asarray([*(angles or []),*state.get('velocity',[])],dtype=float)
    if action.shape!=(9,) or not np.isfinite(action).all():raise ValueError('Неполное состояние для записи')
    if any(not 0<=state.get('sensor_age',{}).get(k,1e9)<=ttl for k,ttl in [('odom',.5),('battery',2),('scan0',.6),('scan1',.6)]):
        raise ValueError('Датчики устарели')
    with np.load(root/'data/rgbd-snapshot.npz',allow_pickle=False) as frame:
        stamp=float(frame['stamp'])
        if not 0<=now-stamp<2.:raise ValueError('Нет свежего кадра камеры')
        camera_state_offset=state['at']-stamp
        # The on-board perception process publishes an acquisition timestamp
        # after a bounded processing delay.  Preserve that delay in every row
        # instead of pretending the two independently written files are
        # simultaneous.  Larger gaps are rejected rather than relabelled.
        if abs(camera_state_offset)>1.5:raise ValueError('Камера и состояние слишком далеко разнесены по времени')
        image=frame['rgb'].copy()
    return dict(at=now,image_stamp=stamp,state_stamp=state['at'],command=action.tolist(),
                camera_state_offset_s=camera_state_offset,
                arm_in_progress=arm['phase'] in ('command_in_progress','EXECUTING'),raw_odometry_pose=state.get('raw_pose'),
                q_estimated=arm.get('q_estimated'),state_source=arm.get('state_source','command_estimate'),
                lidar=state.get('lidar'),sensor_age=state.get('sensor_age'),
                measured_arm_angles=False,velocity_source='commanded_body_velocity'),image


class MobileDemonstrations:
    def __init__(self,root):
        self.root=Path(root);self.folder=self.root/'data/mobile-demonstrations';self.folder.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.active=None;self.last=None;self.lease=0
        for path in self.folder.glob('*/episode.json'):
            record=json.loads(path.read_text())
            if record['state']=='recording':
                record.update(state='interrupted',outcome='unknown');write_json(path,record)

    def status(self):
        episodes=[]
        for path in sorted(self.folder.glob('*/episode.json')):
            try:episodes.append(json.loads(path.read_text()))
            except (OSError,ValueError,TypeError):pass
        eligible=[e for e in episodes if e.get('state')=='complete' and e.get('outcome')=='success' and
                  e.get('label_source')=='operator' and set(e.get('stages',[]))==set(STAGES) and e.get('samples',0)>=20]
        skills={name:sum(e.get('name')==name for e in eligible) for name in {e.get('name') for e in episodes if e.get('name')}}
        with self.lock:return dict(active=self.active,last=self.last,automatic_replay=False,
            format='explorer_mobile_episode_v2',trainable=True,successful=len(eligible),skills=skills,
            training_required_successful_demonstrations=10,recommended_demonstrations='30–100',
            observations=['wrist_rgb','arm_command_estimate','body_velocity_command','odometry','dual_lidar_summary','stage'],
            policy='LeRobot ACT 9-DoF candidate; offline validation before supervised execution')

    def save(self):write_json(self.folder/self.active['id']/'episode.json',self.active)

    def start(self,name,observing,object_label='',object_class='unknown',size_class='medium',destination=''):
        if observing is not True:raise ValueError('Показ требует наблюдателя')
        if object_class not in ('soft_cloth','rigid','fragile','slippery','deformable','unknown'):
            raise ValueError('Неизвестный физический класс предмета')
        if size_class not in ('small','medium','large'):raise ValueError('Неизвестный размер предмета')
        with self.lock:
            if self.active:raise ValueError('Показ поездки уже записывается')
            sample(self.root,time.time())
            ident=uuid.uuid4().hex;(self.folder/ident).mkdir()
            self.active=dict(id=ident,name=name,state='recording',started=time.time(),outcome='unknown',
                stage='travel_to_object',stages=['travel_to_object'],samples=0,
                source='operator_mobile_demonstration',joint_state_source='commanded_not_measured',
                physical_sample_rate_hz=2,automatic_replay_allowed=False,dataset_kind='mobile_manipulation_9dof',
                object_label=object_label[:80],object_class=object_class,size_class=size_class,destination=destination[:80],
                transfer_context=dict(grasp_family='learned_from_operator',contact_feedback='visual_only'))
            self.lease=time.monotonic()+2;self.save()
            threading.Thread(target=self.run,args=(ident,),daemon=True).start()
            return self.status()

    def heartbeat(self,ident,observing):
        with self.lock:
            if not self.active or self.active['id']!=ident:raise ValueError('Запись не найдена')
            self.lease=time.monotonic()+2 if observing is True else 0
            return self.status()

    def stage(self,value):
        if value not in STAGES:raise ValueError('Неизвестный этап')
        with self.lock:
            if not self.active:raise ValueError('Запись не идёт')
            self.active['stage']=value
            if value not in self.active['stages']:self.active['stages'].append(value)
            self.save();return self.status()

    def finish(self,outcome,reason=None):
        if outcome not in ('success','failure','unknown'):raise ValueError('Неизвестный результат')
        with self.lock:
            if not self.active:return self.status()
            if outcome=='success' and (self.active['samples']<20 or set(self.active['stages'])!=set(STAGES)):
                raise ValueError('Для полного успешного показа запишите поездку, захват, перевозку и размещение')
            self.active.update(state='complete' if reason is None else 'interrupted',outcome=outcome,
                               label_source='operator' if reason is None else 'recorder',ended=time.time(),reason=reason)
            self.save();self.last=self.active;self.active=None
            return self.status()

    def run(self,ident):
        previous=-1
        try:
            while True:
                with self.lock:
                    if not self.active or self.active['id']!=ident:return
                    if time.monotonic()>self.lease:raise ValueError('Панель записи отключена')
                    if time.time()-self.active['started']>900:raise ValueError('Достигнут предел 15 минут')
                    record,image=sample(self.root,time.time())
                    if record['image_stamp']>previous:
                        index=self.active['samples'];folder=self.folder/ident
                        filename=f'{index:06d}.jpg'
                        if not cv2.imwrite(str(folder/filename),image):raise ValueError('Не удалось записать изображение')
                        record.update(image=filename,stage=self.active['stage'])
                        with (folder/'samples.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                        self.active['samples']+=1;self.save();previous=record['image_stamp']
                time.sleep(.2)
        except (OSError,ValueError,KeyError) as exc:
            with self.lock:
                if self.active and self.active['id']==ident:self.finish('unknown',str(exc))
