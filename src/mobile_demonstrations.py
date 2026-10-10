"""Record a whole operator-driven journey without interrupting teleoperation.

State/action provenance stays commanded-only. This recorder cannot drive a robot
and its episodes are not silently mixed into the six-joint ACT training set.
"""
import json
import shutil
import threading
import time
import uuid
from stored_records import records
from pathlib import Path
import cv2
import numpy as np
from lerobot_bridge import write_json
from mobile_policy_contract import SAMPLE_CONTRACT,CONTRACT_SHA256

STAGES=("travel_to_object", "grasp", "carry", "place")

def required_stages(name):
    return set()


def episode_quality(folder,full_task=False):
    """Evaluate recorded evidence, never infer an operator outcome from it."""
    path=Path(folder)/"samples.jsonl"
    if not path.exists():
        return {"usable":False,"reason":"Нет синхронизированных кадров и команд","samples":0}
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)<2:
        return {"usable":False,"reason":"Слишком короткая запись","samples":len(rows)}
    stamps=[float(row["image_stamp"]) for row in rows]
    gaps=[b-a for a,b in zip(stamps,stamps[1:])]
    commands=[tuple(float(value) for value in row["command"]) for row in rows]
    duration=stamps[-1]-stamps[0]
    rate=(len(rows)-1)/duration if duration>0 else 0.0
    base_changes=sum(any(abs(value)>0.025 for value in command[6:]) for command in commands)
    arm_changes=len({command[:6] for command in commands})
    reasons=[]
    if any(not np.isfinite(stamp) for stamp in stamps) or any(gap<=0 for gap in gaps):
        reasons.append("Времена кадров не возрастают")
    if any(not np.isfinite(command).all() for command in commands):reasons.append("В командах есть нечисловые значения")
    if any(row.get("sample_contract_sha256") not in (None,CONTRACT_SHA256) for row in rows):
        reasons.append("Несовместимая семантика команд")
    if any(row.get("executed_action_source") not in (None,"operator_manual_control") for row in rows):
        reasons.append("Самостоятельные команды модели не являются показом оператора")
    if len(rows)<12 or duration<5:reasons.append("Нужен более длинный показ")
    if len(set(commands))<5:reasons.append("В записи почти нет разных команд")
    if max(gaps)>1.5:reasons.append("Есть длинный пропуск кадров")
    if rate<1.5:reasons.append("Камера слишком медленная для этого показа")
    if any(len(command)!=9 for command in commands):reasons.append("Неполные команды шасси или руки")
    gripper_targets=len({command[5] for command in commands if len(command)>5})
    if full_task and base_changes<2:reasons.append("Не записана поездка шасси")
    if full_task and arm_changes<3:reasons.append("Не записано движение руки")
    if full_task and gripper_targets<2:reasons.append("Не записано управление захватом")
    return {"usable":not reasons,"reason":"; ".join(reasons),"samples":len(rows),
            "duration_s":round(duration,3),"observed_fps":round(rate,3),
            "largest_frame_gap_s":round(max(gaps),3),"distinct_commands":len(set(commands)),
            "base_motion_samples":base_changes,"distinct_arm_targets":arm_changes,
            "distinct_gripper_targets":gripper_targets,
            "arm_state_source":"command_estimate"}


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
    try:manual=json.loads((root/'data/manual-teleop.json').read_text())
    except (OSError,ValueError,TypeError):manual={}
    if now-manual.get('at',0)>1:manual={}
    issued_source=arm.get("command_source",arm.get("source"))
    if arm.get("phase") in ("command_in_progress","EXECUTING") and issued_source in (
        "supervised_mobile_policy","local_mission","autonomous","learned_policy"):
        raise ValueError("Самостоятельное движение не записывается как показ оператора")
    estimated=arm.get("q_estimated_deg") or angles
    observation=np.asarray([*(estimated or []),*state.get("velocity",[])],dtype=float)
    if observation.shape!=(9,) or not np.isfinite(observation).all():raise ValueError("Неполная оценка для записи")
    return dict(at=now,image_stamp=stamp,state_stamp=state['at'],command=action.tolist(),
                schema="explorer_mobile_sample_v3",sample_contract_sha256=CONTRACT_SHA256,
                observation_state=observation.tolist(),applied_action=action.tolist(),
                executed_action_source="operator_manual_control",sample_contract=SAMPLE_CONTRACT,
                image_stamp_source="onboard_camera_acquisition_estimate",command_stamp=arm.get("at"),
                arm_command_sent_at=arm.get("command_sent_at",arm.get("at")),
                arm_velocity_deg_s=arm.get("velocity_deg_s"),arm_acceleration_deg_s2=arm.get("acceleration_deg_s2"),
                command_generation=arm.get("command_generation"),arm_owner=arm.get("owner",arm.get("input_source")),
                applied_command_source=issued_source,
                observation_arm_source="command_trajectory_estimate" if arm.get("q_estimated_deg") else "command_estimate",
                proposed_manual_action=manual.get("normalized_actions"),
                issued_body_velocity=state.get("velocity"),observed_body_velocity=state.get("odom_velocity"),
                camera_state_offset_s=camera_state_offset,
                arm_in_progress=arm['phase'] in ('command_in_progress','EXECUTING'),raw_odometry_pose=state.get('raw_pose'),
                q_estimated=arm.get('q_estimated'),state_source=arm.get('state_source','command_estimate'),
                lidar=state.get('lidar'),sensor_age=state.get('sensor_age'),
                measured_arm_angles=False,velocity_source='commanded_body_velocity',
                manual_control=manual or None),image


class MobileDemonstrations:
    def __init__(self,root):
        self.root=Path(root);self.folder=self.root/'data/mobile-demonstrations';self.folder.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.active=None;self.last=None;self.lease=0
        self.on_complete=None
        rows,self.record_errors=records(self.folder.glob("*/episode.json"),("id","state","started"))
        for path,record in rows:
            if record['state']=='recording':
                record.update(state='interrupted',outcome='unknown');write_json(path,record)

    def status(self):
        rows,self.record_errors=records(self.folder.glob("*/episode.json"),("id","state","started"))
        episodes=[row for _,row in rows]
        episodes.sort(key=lambda item:item.get('started',0),reverse=True)
        eligible=[e for e in episodes if e.get("state")=="complete" and e.get("outcome")=="success" and
                  e.get("label_source")=="operator" and e.get("quality",{}).get("usable") is True]
        skills={name:sum(e.get('name')==name for e in eligible) for name in {e.get('name') for e in episodes if e.get('name')}}
        with self.lock:return dict(active=self.active,last=self.last or (episodes[0] if episodes else None),
            record_errors=self.record_errors,recent=episodes[:10],storage_root=str(self.folder),automatic_replay=False,
            format='explorer_mobile_episode_v2',trainable=True,successful=len(eligible),skills=skills,
            training_required_successful_demonstrations=1,recommended_demonstrations='5–100; больше разнообразия обычно лучше',
            observations=['wrist_rgb','arm_command_estimate','body_velocity_command','odometry','dual_lidar_summary','stage'],
            policy='LeRobot ACT 9-DoF candidate; offline validation before supervised execution')

    def save(self):write_json(self.folder/self.active['id']/'episode.json',self.active)

    def start(self,name,observing,object_label='',object_class='unknown',size_class='medium',destination='',workflow_id='',skill_id=''):
        if observing is not True:raise ValueError('Показ требует наблюдателя')
        if object_class not in ('soft_cloth','rigid','fragile','slippery','deformable','unknown'):
            raise ValueError('Неизвестный физический класс предмета')
        if size_class not in ('small','medium','large'):raise ValueError('Неизвестный размер предмета')
        with self.lock:
            if self.active:raise ValueError('Показ поездки уже записывается')
            if shutil.disk_usage(self.root).free<2*1024**3:raise ValueError("На диске осталось меньше 2 ГБ")
            sample(self.root,time.time())
            ident=uuid.uuid4().hex;(self.folder/ident).mkdir()
            self.active=dict(id=ident,name=name,state='recording',started=time.time(),outcome='unknown',
                stage='travel_to_object',stages=['travel_to_object'],samples=0,
                source='operator_mobile_demonstration',joint_state_source='commanded_not_measured',
                physical_sample_rate_hz=2,automatic_replay_allowed=False,dataset_kind='mobile_manipulation_9dof',
                control_metadata_schema='explorer-manual-v2',legacy_action_vector_unchanged=True,
                object_label=object_label[:80],object_class=object_class,size_class=size_class,destination=destination[:80],
                workflow_id=workflow_id,skill_id=skill_id,
                transfer_context=dict(grasp_family='learned_from_operator',contact_feedback='visual_only'))
            self.active.update(sample_contract=SAMPLE_CONTRACT,sample_contract_sha256=CONTRACT_SHA256,
                format="explorer_mobile_episode_v3",synchronization="timestamped_bounded_offset_no_measured_joint_ground_truth")
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

    def finish(self,outcome,reason=None,expected_id=None):
        if outcome not in ('success','failure','unknown'):raise ValueError('Неизвестный результат')
        with self.lock:
            if not self.active:
                if expected_id and self.last and self.last["id"]!=expected_id:raise ValueError("Другой эпизод уже завершён")
                return self.status()
            if expected_id and self.active["id"]!=expected_id:raise ValueError("Завершается другой эпизод")
            full_task=bool(self.active.get("object_label") and self.active.get("destination"))
            quality=episode_quality(self.folder/self.active["id"],full_task=full_task)
            self.active.update(state='complete' if reason is None else 'interrupted',outcome=outcome,
                               label_source='operator' if reason is None else 'recorder',ended=time.time(),reason=reason,
                               quality=quality,physical_sample_rate_hz=quality.get("observed_fps"))
            self.save();self.last=self.active;completed=self.active;self.active=None
            if self.on_complete is not None:
                try:self.on_complete(completed,self.folder/completed['id']/'episode.json')
                except (OSError,ValueError,KeyError) as exc:
                    completed["postprocess_error"]=str(exc)
                    write_json(self.folder/completed["id"]/"episode.json",completed)
            return self.status()

    def run(self,ident):
        previous=-1;last_good=time.monotonic()
        try:
            while True:
                with self.lock:
                    if not self.active or self.active['id']!=ident:return
                    if time.monotonic()>self.lease:raise ValueError('Панель записи отключена')
                    if time.time()-self.active['started']>900:raise ValueError('Достигнут предел 15 минут')
                    record=image=None
                    try:record,image=sample(self.root,time.time())
                    except ValueError as exc:
                        transient=str(exc) in ('Камера и состояние слишком далеко разнесены по времени','Нет свежего кадра камеры')
                        if not transient or time.monotonic()-last_good>3:raise
                        self.active['sampling_warning']=str(exc);self.save()
                    if record is not None and record['image_stamp']>previous:
                        index=self.active['samples'];folder=self.folder/ident
                        filename=f'{index:06d}.jpg'
                        if not cv2.imwrite(str(folder/filename),image):raise ValueError('Не удалось записать изображение')
                        record.update(image=filename,stage=self.active['stage'])
                        with (folder/'samples.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                        self.active['samples']+=1;self.active.pop('sampling_warning',None)
                        self.save();previous=record['image_stamp'];last_good=time.monotonic()
                time.sleep(.2)
        except (OSError,ValueError,KeyError) as exc:
            with self.lock:
                if self.active and self.active['id']==ident:self.finish('unknown',str(exc))
