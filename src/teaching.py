"""Supervised demonstrations for LeRobot. No policy-driven actuator access."""
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np
from arm_commissioning import stationary_status,coordinated_status

ROOT=Path('/home/vlad/Explorer')
class Demonstrations:
    """Original images, commands and human labels; learning is delegated to LeRobot."""
    def __init__(self, root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def episodes(self):
        return [json.loads(p.read_text()) for p in sorted(self.root.glob('*/episode.json'))]

    def eligible(self):
        return [e for e in self.episodes() if e.get('state')=='complete' and
                e.get('outcome')=='success' and e.get('label_source')=='operator' and
                e.get('source')=='operator_demonstration' and len(e.get('steps',[]))>=3]

    def status(self):
        episodes=self.episodes()
        skills={name:sum(e['name']==name for e in self.eligible()) for name in {e['name'] for e in episodes}}
        return dict(recordings=len(episodes),successful=len(self.eligible()),
                    skills=skills,
                    failed=sum(e.get('outcome')=='failure' for e in episodes),
                    episodes=[dict(id=e['id'],name=e['name'],outcome=e['outcome'],
                                   steps=len(e['steps']),state=e['state']) for e in episodes][-30:],
                    framework='LeRobot 0.6.1',policy_type='ACT',
                    training_required_successful_demonstrations=1,
                    joint_state_source='per_episode_provenance',automatic_motion_enabled=False)


class TeachingController:
    def __init__(self, root=ROOT):
        self.root=Path(root);self.store=Demonstrations(self.root/'data/demonstrations')
        self.lock=threading.Lock();self.active=None;self.error=None
        self.move=None;self.stop_revision=lambda:0
        self.measured_reference=None
        self.arm_reference=None
        self.on_complete=None
        # A restart never resumes recording or motion.
        for episode in self.store.episodes():
            if episode.get('state')=='recording':
                episode.update(state='interrupted',outcome='unknown')
                self.save(episode)

    def save(self,episode):
        folder=self.store.root/episode['id'];folder.mkdir(exist_ok=True)
        temporary=folder/'episode.tmp';temporary.write_text(json.dumps(episode));temporary.replace(folder/'episode.json')

    def pose(self,coordinated=False):
        if self.arm_reference is not None:return self.arm_reference()['servo_deg']
        state=json.loads((self.root/'data/status.json').read_text())
        (coordinated_status if coordinated else stationary_status)(state,time.time())
        if self.measured_reference is not None:
            return self.measured_reference()['servo_deg']
        arm=json.loads((self.root/'data/arm-state.json').read_text())
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if arm.get('boot_id')!=boot or arm.get('phase')!='command_elapsed_observation_required':
            raise ValueError('Исходное положение руки не подтверждено в текущем сеансе')
        fault=self.root/'data/arm-telemetry-fault.json'
        if fault.exists() and json.loads(fault.read_text()).get('at',0)>arm['at']:
            raise ValueError('После потери связи заново подготовьте руку')
        return arm['servo_deg']

    def observation(self):
        pose=self.pose()
        path=self.root/'data/frame-raw.jpg'
        if not 0<=time.time()-path.stat().st_mtime<2:raise ValueError('Нет свежего кадра камеры')
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Не удалось прочитать кадр')
        return pose,image

    def status(self):
        try:
            self.observation();blocked=None
        except (OSError,ValueError,KeyError) as exc:blocked=str(exc)
        return dict(self.store.status(),active=self.active,error=self.error,busy=self.lock.locked(),
                    blocked_by=blocked)

    def start(self,name,observing,workflow_id=''):
        if observing is not True:raise ValueError('Показ требует присутствия наблюдателя у робота')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        try:
            if self.active:raise ValueError('Запись уже идёт')
            pose,image=self.observation()
            episode=dict(id=uuid.uuid4().hex,created=time.time(),name=name[:80],state='recording',
                         outcome='unknown',label_source=None,source='operator_demonstration',
                         workflow_id=workflow_id,
                         start_deg=pose,steps=[],joint_positions_measured=self.arm_reference is None and self.measured_reference is not None,
                         joint_state_source='command_estimate' if self.arm_reference is not None else 'legacy')
            self.save(episode)
            cv2.imwrite(str(self.store.root/episode['id']/'start.jpg'),image)
            self.active=episode;self.error=None
            return self.status()
        finally:self.lock.release()

    def jog(self,joint,delta,observing,deadline=None,coordinated=False,speed='normal'):
        if observing is not True:raise ValueError('Подтвердите присутствие рядом с роботом')
        if self.active:return self.step(joint,delta,deadline)
        if type(joint) is not int or not 1<=joint<=6 or type(delta) is not int or delta not in (-10,-5,-3,-2,2,3,5,10):
            raise ValueError('Разрешён шаг одного сустава на 2, 3, 5 или 10°')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        try:
            if self.move is None:raise ValueError('Постоянный контроллер руки не готов')
            pose=self.pose(coordinated);goal=list(pose);goal[joint-1]+=delta
            return self.move(pose,goal,deadline,source='coordinated_operator' if coordinated else 'operator',speed=speed)
        finally:self.lock.release()

    def cartesian(self,model,axis,direction,observing,deadline=None):
        if observing is not True:raise ValueError('Подтвердите присутствие рядом с роботом')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        try:
            if self.active:raise ValueError('Для записи показа используйте шаги суставов; сначала завершите запись')
            if self.move is None:raise ValueError('Контроллер руки не готов')
            from cartesian_jog import propose
            revision=self.stop_revision()
            pose=self.pose();proposal=propose(model,pose,axis,direction)
            result=self.move(pose,proposal['goal_deg'],deadline=deadline,expected_stop_revision=revision)
            return dict(proposal,executed=True,command=result,attainment_verified=result.get('attained') is True)
        finally:self.lock.release()

    def teleop(self,model,xyz_delta,joint_delta,observing,deadline=None,precision=False,arm_mode='cartesian'):
        """Execute one bounded segment already integrated from proportional input."""
        if observing is not True:raise ValueError('Подтвердите присутствие рядом с роботом')
        if arm_mode not in ('cartesian','joint') or len(xyz_delta)!=3 or len(joint_delta)!=6:
            raise ValueError('Неверный вектор teleop')
        if not all(math.isfinite(float(v)) for v in xyz_delta) or any(type(v) is not int for v in joint_delta):
            raise ValueError('Неверный шаг teleop')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий сегмент руки ещё выполняется')
        try:
            if self.move is None:raise ValueError('Контроллер руки не готов')
            pose=self.pose(coordinated=True);goal=list(pose)
            xyz_error=None
            magnitude=math.sqrt(sum(float(value)**2 for value in xyz_delta))
            if arm_mode=='cartesian' and magnitude>0:
                from cartesian_jog import propose_delta
                try:goal=propose_delta(model,pose,[float(value) for value in xyz_delta],maximum_distance=.010)['goal_deg']
                except ValueError as exc:xyz_error=str(exc)
            indices=range(6) if arm_mode=='joint' else (5,)
            for index in indices:
                goal[index]=max(pose[index]-10,min(pose[index]+10,goal[index]+joint_delta[index]))
            if goal==pose:raise ValueError('Нет исполнимого движения руки')
            revision=self.stop_revision()
            result=self.move(pose,goal,deadline=deadline,expected_stop_revision=revision,
                             source='coordinated_operator',speed='precision' if precision else 'teleop')
            if xyz_error:result['cartesian_rejected']=xyz_error
            return result
        finally:self.lock.release()

    def step(self,joint,delta,deadline=None):
        if type(joint) is not int or not 1<=joint<=6 or type(delta) is not int or delta not in (-10,-5,-3,-2,2,3,5,10):
            raise ValueError('Разрешён один сустав и шаг 2, 3, 5 или 10°')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        try:
            if not self.active:raise ValueError('Сначала начните показ')
            pose,image=self.observation();goal=list(pose);goal[joint-1]+=delta
            index=len(self.active['steps']);folder=self.store.root/self.active['id']
            observed_at=time.time()
            before=f'{index:04d}-before.jpg';cv2.imwrite(str(folder/before),image)
            if self.move is None:raise ValueError('Постоянный контроллер руки не готов')
            result=self.move(pose,goal,deadline)
            (folder/f'{index:04d}-command.log').write_text(json.dumps(result))
            after_pose,after_image=self.observation()
            matched=np.allclose(after_pose,goal,atol=.5,rtol=0)
            if not matched:raise ValueError('Состояние руки не совпало с целью показа')
            after=f'{index:04d}-after.jpg';cv2.imwrite(str(folder/after),after_image)
            self.active['steps'].append(dict(at=time.time(),observation_at=observed_at,start_deg=pose,goal_deg=goal,
                before_image=before,after_image=after,measured=result.get('measured') is True,
                measured_after_deg=after_pose if result.get('measured') else None,
                q_estimated_deg=after_pose if not result.get('measured') else None,
                state_source=result.get('source'),action=dict(arm_command=goal,gripper_command=goal[5]),
                actuator_attainment_verified=result.get('attained') is True))
            self.save(self.active);return self.status()
        except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
            self.error=str(exc)
            if self.active:
                self.active.update(state='interrupted',outcome='unknown');self.save(self.active);self.active=None
            raise ValueError(self.error)
        finally:self.lock.release()

    def finish(self,outcome):
        if outcome not in ('success','failure','unknown'):raise ValueError('Неизвестный результат')
        if not self.lock.acquire(blocking=False):raise ValueError('Дождитесь окончания текущего короткого шага')
        try:
            if not self.active:raise ValueError('Нет текущего показа')
            self.active.update(state='complete',ended=time.time(),outcome=outcome,label_source='operator')
            self.save(self.active);completed=self.active;self.active=None
            if self.on_complete is not None:self.on_complete(completed,self.store.root/completed['id']/'episode.json')
            return self.status()
        finally:self.lock.release()
