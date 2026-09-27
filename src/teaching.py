"""Supervised demonstrations for LeRobot. No policy-driven actuator access."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np
from arm_commissioning import stationary_status

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
                    training_required_successful_demonstrations=10,
                    joint_state_source='commanded_not_measured',automatic_motion_enabled=False)


class TeachingController:
    def __init__(self, root=ROOT):
        self.root=Path(root);self.store=Demonstrations(self.root/'data/demonstrations')
        self.lock=threading.Lock();self.active=None;self.error=None
        self.move=None
        # A restart never resumes recording or motion.
        for episode in self.store.episodes():
            if episode.get('state')=='recording':
                episode.update(state='interrupted',outcome='unknown')
                self.save(episode)

    def save(self,episode):
        folder=self.store.root/episode['id'];folder.mkdir(exist_ok=True)
        temporary=folder/'episode.tmp';temporary.write_text(json.dumps(episode));temporary.replace(folder/'episode.json')

    def observation(self):
        state=json.loads((self.root/'data/status.json').read_text());stationary_status(state,time.time())
        arm=json.loads((self.root/'data/arm-state.json').read_text())
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if arm.get('boot_id')!=boot or arm.get('phase')!='command_elapsed_observation_required':
            raise ValueError('Исходное положение руки не подтверждено в текущем сеансе')
        fault=self.root/'data/arm-telemetry-fault.json'
        if fault.exists() and json.loads(fault.read_text()).get('at',0)>arm['at']:
            raise ValueError('После потери связи заново подготовьте руку')
        path=self.root/'data/frame-raw.jpg'
        if not 0<=time.time()-path.stat().st_mtime<2:raise ValueError('Нет свежего кадра камеры')
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Не удалось прочитать кадр')
        return arm['servo_deg'],image

    def status(self):
        try:
            self.observation();blocked=None
        except (OSError,ValueError,KeyError) as exc:blocked=str(exc)
        return dict(self.store.status(),active=self.active,error=self.error,busy=self.lock.locked(),
                    blocked_by=blocked)

    def start(self,name,observing):
        if observing is not True:raise ValueError('Показ требует присутствия наблюдателя у робота')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        try:
            if self.active:raise ValueError('Запись уже идёт')
            pose,image=self.observation()
            episode=dict(id=uuid.uuid4().hex,created=time.time(),name=name[:80],state='recording',
                         outcome='unknown',label_source=None,source='operator_demonstration',
                         start_deg=pose,steps=[],joint_positions_measured=False)
            self.save(episode)
            cv2.imwrite(str(self.store.root/episode['id']/'start.jpg'),image)
            self.active=episode;self.error=None
            return self.status()
        finally:self.lock.release()

    def jog(self,joint,delta,observing,deadline=None):
        if observing is not True:raise ValueError('Подтвердите присутствие рядом с роботом')
        if self.active:return self.step(joint,delta,deadline)
        if type(joint) is not int or not 1<=joint<=6 or type(delta) is not int or delta not in (-2,2):
            raise ValueError('Разрешён шаг одного сустава на 2°')
        if not self.lock.acquire(blocking=False):raise ValueError('Предыдущий шаг ещё выполняется')
        try:
            if self.move is None:raise ValueError('Постоянный контроллер руки не готов')
            pose,_=self.observation();goal=list(pose);goal[joint-1]+=delta
            return self.move(pose,goal,deadline)
        finally:self.lock.release()

    def step(self,joint,delta,deadline=None):
        if type(joint) is not int or not 1<=joint<=6 or type(delta) is not int or delta not in (-2,2):
            raise ValueError('Разрешён один сустав и шаг 2 градуса')
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
            if after_pose!=goal:raise ValueError('Состояние команды изменилось во время показа')
            after=f'{index:04d}-after.jpg';cv2.imwrite(str(folder/after),after_image)
            self.active['steps'].append(dict(at=time.time(),observation_at=observed_at,start_deg=pose,goal_deg=goal,
                before_image=before,after_image=after,measured=False,actuator_attainment_verified=False))
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
            self.save(self.active);self.active=None
            return self.status()
        finally:self.lock.release()
