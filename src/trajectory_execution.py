"""Observed MoveIt plans executed as finite, collision-rechecked servo steps.

No base motion, boot homing, learned success claim or client-supplied trajectory.
Held manual, finite observed and local mission execution share one executor.
"""
import json
import math
import threading
import time
import uuid
from pathlib import Path
import numpy as np
from lerobot_bridge import write_json
from arm_commissioning import stationary_status
from arm_commissioning import HARD_LIMITS


def motion_budget(root):
    """Use the arm's live motion policy, not the separate GPU-training budget."""
    status=json.loads((Path(root)/'data/status.json').read_text())
    stationary_status(status,time.time())


def command_steps(points,start):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=5 or not 2<=len(p)<=2000 or not np.isfinite(p).all():
        raise ValueError('Некорректная траектория')
    if not np.allclose(p[0],start[:5],atol=.1):raise ValueError('Начало пути не совпадает с положением руки')
    if np.any(p<0) or np.any(p>np.array([180,180,180,180,270])):
        raise ValueError('Траектория вне диапазона приводов')
    result=[];previous=list(start)
    for a,b in zip(p,p[1:]):
        # One-degree interpolation leaves room for integer quantization.
        count=max(1,math.ceil(float(np.max(np.abs(b-a)))))
        if count>300:raise ValueError('Слишком длинный сегмент')
        for i in range(1,count+1):
            goal=np.rint(a+(b-a)*i/count).astype(int).tolist()+[start[5]]
            if goal!=previous:
                if max(abs(x-y) for x,y in zip(goal,previous))>2:raise ValueError('Разрыв траектории')
                result.append(goal);previous=goal
                if len(result)>150:raise ValueError('Путь длиннее одного наблюдаемого сеанса')
    if not result:raise ValueError('Рука уже в заданной позе')
    return result

def gripper_steps(start,target):
    if type(target) is not int or not 30<=target<=170:
        raise ValueError('Проверенный диапазон захвата: 30–170°, сила не измеряется')
    result=[];pose=list(start)
    while pose[5]!=target:
        pose=list(pose);pose[5]+=max(-2,min(2,target-pose[5]));result.append(pose)
    return result


class TrajectoryExecution:
    def __init__(self,root,teaching,manual,planner):
        self.root=Path(root);self.teaching=teaching;self.manual=manual;self.planner=planner
        self.lock=threading.Lock();self.pending=None;self.state=dict(phase='idle')
        self.session=None;self.lease=0.;self.cancelled=threading.Event()
        self.execution_mode='operator_held';self.local_permit=None;self.deadline=0

    def status(self):
        return dict(self.state,busy=self.lock.locked(),requires_held_button=self.execution_mode=='operator_held',
                    physically_verified=False,base_motion=False)

    def plan(self,goal):
        if len(goal) not in (5,6) or any(type(v) is not int for v in goal):raise ValueError('Нужны пять или шесть целых углов')
        if any(not lo<=v<=hi for v,(lo,hi) in zip(goal,HARD_LIMITS)):raise ValueError('Угол вне диапазона')
        if not self.lock.acquire(blocking=False):raise ValueError('Уже идёт расчёт или выполнение пути')
        reserved=False
        try:
            if not self.teaching.lock.acquire(blocking=False):raise ValueError('Рука занята')
            reserved=True;self.pending=None
            if self.teaching.active:raise ValueError('Завершите запись показа')
            revision=self.manual.stop_revision
            start,_=self.teaching.observation()
            steps=[]
            if start[:5]!=goal[:5]:
                plan=self.planner().plan(start[:5],goal[:5],0.)
                if not plan.get('planned'):raise ValueError('MoveIt не нашёл путь')
                steps=command_steps(plan['servo_waypoints'],start)
            if len(goal)==6:steps+=gripper_steps(steps[-1] if steps else start,goal[5])
            if not steps:raise ValueError('Рука уже получила эту команду; достигнутая поза не измерена')
            a=start
            for b in steps:
                for shape in (0.,-.2,-.4,-.6,-.8):
                    if not self.manual.model().path(a[:5],b[:5],shape)['valid']:
                        raise ValueError('Округлённый путь пересекает робота или пол')
                a=b
            if revision!=self.manual.stop_revision:raise ValueError('Расчёт отменён кнопкой STOP')
            self.pending=dict(id=uuid.uuid4().hex,at=time.time(),start=start,goal=goal,
                              steps=steps,revision=revision)
            self.state=dict(phase='planned',plan_id=self.pending['id'],goal_deg=goal,
                            total_steps=len(steps),steps=0,executed=False,
                            estimated_duration_s=round(len(steps)*.4,1))
            return self.status()
        finally:
            if reserved:self.teaching.lock.release()
            self.lock.release()

    def permit(self):
        if self.cancelled.is_set():raise ValueError('Движение отменено')
        if self.execution_mode=='operator_held' and time.monotonic()>self.lease:raise ValueError('Кнопка отпущена или панель отключена')
        if self.execution_mode!='operator_held' and time.monotonic()>self.deadline:raise ValueError('Истёк срок конечного движения')
        if self.execution_mode=='local_mission':self.local_permit()
        if self.manual.stop_revision!=self.revision:raise ValueError('Нажат STOP')

    def heartbeat(self,session,held):
        if not self.lock.locked() or session!=self.session:raise ValueError('Сеанс не найден')
        if held:self.lease=time.monotonic()+.6
        else:self.stop()
        return self.status()

    def stop(self):
        self.cancelled.set();self.lease=0;self.manual.stop()
        return dict(stopping=True,accepted_servo_command_recallable=False)

    def start_local(self,plan_id,mission_permit):
        """Internal task API: a real local mission callback, never a browser lease."""
        if not callable(mission_permit):raise ValueError('Нужно разрешение локальной миссии')
        mission_permit()
        return self.start(plan_id,False,finite=True,mission_permit=mission_permit)

    def start(self,plan_id,observing,finite=False,mission_permit=None):
        if mission_permit is None and observing is not True:raise ValueError('Нужен наблюдатель у робота')
        if not self.lock.acquire(blocking=False):raise ValueError('Рука уже выполняет путь')
        reserved=False
        try:
            if not self.teaching.lock.acquire(blocking=False):raise ValueError('Рука занята')
            reserved=True
            p=self.pending
            if not p or p['id']!=plan_id or not 0<=time.time()-p['at']<60:
                raise ValueError('Рассчитайте свежий путь')
            if self.teaching.active:raise ValueError('Завершите показ')
            motion_budget(self.root)
            start,_=self.teaching.observation()
            if start!=p['start'] or p['revision']!=self.manual.stop_revision:
                raise ValueError('Положение или STOP изменились после расчёта')
            self.revision=p['revision'];self.session=uuid.uuid4().hex
            self.cancelled.clear();self.lease=time.monotonic()+.6;self.pending=None
            self.execution_mode='local_mission' if mission_permit else 'operator_finite' if finite else 'operator_held'
            self.local_permit=mission_permit
            self.deadline=time.monotonic()+min(180.,5.+len(p['steps'])*.7)
            self.state.update(phase='moving',session=self.session,steps=0,executed=False,execution_mode=self.execution_mode)
            threading.Thread(target=self.run,args=(p,),daemon=True).start()
            return self.status()
        except Exception:
            if reserved:self.teaching.lock.release()
            self.lock.release();raise

    def run(self,plan):
        records=[];start=plan['start']
        try:
            for goal in plan['steps']:
                self.permit();motion_budget(self.root)
                observed,_=self.teaching.observation()
                if observed!=start:raise ValueError('Положение руки изменилось')
                record=self.manual.move(start,goal,expected_stop_revision=self.revision,
                                        execution_permit=self.permit,source='local_mission' if self.execution_mode=='local_mission' else 'supervised_trajectory')
                records.append(record);start=goal
                self.state.update(steps=len(records),executed=True)
            self.state.update(phase='commanded',reason='Команды пути завершены; проверьте физическое положение')
        except (OSError,ValueError,KeyError) as exc:self.state.update(phase='stopped',reason=str(exc))
        finally:
            try:
                folder=self.root/'data/trajectory-runs';folder.mkdir(exist_ok=True)
                write_json(folder/(self.session+'.json'),dict(plan=plan,commands=records,result=self.state,
                                                            physical_attainment_verified=False))
            finally:self.teaching.lock.release();self.lock.release()
