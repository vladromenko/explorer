"""MoveIt paths executed with measured native feedback or explicit legacy steps.

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
    profile_path=Path(root)/'config/controller-profile.json'
    profile=json.loads(profile_path.read_text()) if profile_path.exists() else {}
    if profile.get('manual_reference_version')==1:
        from controller_arm_commissioning import require_arm_test_power
        require_arm_test_power(root)
    else:stationary_status(status,time.time())


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
        self.lock=threading.Lock();self.state_lock=threading.RLock();self.pending=None;self.state=dict(phase='idle',reached=False)
        self.last_result=None
        self.session=None;self.lease=0.;self.cancelled=threading.Event()
        self.execution_mode='operator_held';self.local_permit=None;self.deadline=0

    def status(self):
        with self.state_lock:
            busy=self.lock.locked()
            return dict(self.state,busy=busy,requires_held_button=self.execution_mode=='operator_held',
                        physically_verified=not busy and self.state.get('phase')=='reached' and self.state.get('reached') is True,
                        previous_result=dict(self.last_result) if self.last_result else None,base_motion=False)

    def plan(self,goal):
        if not isinstance(goal,list) or len(goal) not in (5,6) or any(type(v) not in (float,int) or not math.isfinite(v) for v in goal):raise ValueError('Нужны пять или шесть конечных углов')
        if getattr(self.manual,'native',False) is not True and any(type(v) is not int for v in goal):raise ValueError('Старый протокол требует целые углы')
        if any(not lo<=v<=hi for v,(lo,hi) in zip(goal,HARD_LIMITS)):raise ValueError('Угол вне диапазона')
        if not self.lock.acquire(blocking=False):raise ValueError('Уже идёт расчёт или выполнение пути')
        reserved=False
        try:
            with self.state_lock:self.state=dict(phase='planning',reached=False,executed=False)
            if not self.teaching.lock.acquire(blocking=False):raise ValueError('Рука занята')
            reserved=True;self.pending=None
            if self.teaching.active:raise ValueError('Завершите запись показа')
            revision=self.manual.stop_revision
            native=getattr(self.manual,'native',False) is True
            factory=getattr(self.manual,'factory_timed',False) is True
            start=self.teaching.pose() if native or factory else self.teaching.observation()[0]
            steps=[]
            plan=None
            if not np.allclose(start[:5],goal[:5],atol=.05):
                plan=self.planner().plan(start[:5],goal[:5],0.)
                if not plan.get('planned'):raise ValueError('MoveIt не нашёл путь')
                if not native and not factory:steps=command_steps(plan['servo_waypoints'],start)
            if native or factory:
                if len(goal)==5:goal=list(goal)+[start[5]]
                if plan is None and np.allclose(start,goal,atol=.05):raise ValueError('Рука уже в заданной позе')
                if revision!=self.manual.stop_revision:raise ValueError('Расчёт отменён кнопкой STOP')
                duration=plan['joint_trajectory']['times'][-1] if plan else 0.
                self.pending=dict(id=uuid.uuid4().hex,at=time.time(),start=start,goal=goal,
                    revision=revision,native=native,factory_timed=factory,steps=[],
                    joint_trajectory=plan['joint_trajectory'] if plan else None)
                self.state=dict(phase='planned',plan_id=self.pending['id'],goal_deg=goal,
                    reached=False,executed=False,estimated_duration_s=None,moveit_duration_s=duration,continuous_motion=True,
                    full_moveit_path_retained=plan is not None)
                return self.status()
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
                            total_steps=len(steps),steps=0,reached=False,executed=False,
                            estimated_duration_s=round(len(steps)*.4,1))
            return self.status()
        except Exception as exc:
            with self.state_lock:self.state.update(phase='failed',reached=False,reason=str(exc))
            raise
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
        with self.state_lock:
            planning=self.state.get("phase")=="planning"
            self.cancelled.set();self.lease=0
            self.state.update(phase='stopping' if self.lock.locked() else 'stopped',reached=False,reason='Движение отменено')
        if planning:
            planner=self.planner()
            if hasattr(planner,"cancel"):planner.cancel()
        self.manual.stop()
        return dict(stopping=True,cancel_requested=True,physical_stop_confirmed=False)

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
            with self.state_lock:self.state.update(reached=False,reason=None)
            if not self.teaching.lock.acquire(blocking=False):raise ValueError('Рука занята')
            reserved=True
            p=self.pending
            if not p or p['id']!=plan_id or not 0<=time.time()-p['at']<60:
                raise ValueError('Рассчитайте свежий путь')
            if self.teaching.active:raise ValueError('Завершите показ')
            motion_budget(self.root)
            start=self.teaching.pose() if p.get('native') or p.get('factory_timed') else self.teaching.observation()[0]
            same=np.allclose(start,p['start'],atol=.5) if p.get('native') else start==p['start']
            if not same or p['revision']!=self.manual.stop_revision:
                raise ValueError('Положение или STOP изменились после расчёта')
            self.revision=p['revision'];self.session=uuid.uuid4().hex
            self.cancelled.clear();self.lease=time.monotonic()+.6;self.pending=None
            self.execution_mode='local_mission' if mission_permit else 'operator_finite' if finite else 'operator_held'
            self.local_permit=mission_permit
            self.deadline=time.monotonic()+(600. if p.get('native') or p.get('factory_timed') else min(180.,5.+len(p['steps'])*.7))
            with self.state_lock:
                self.state.update(phase='moving',session=self.session,steps=0,reached=False,reason=None,
                                  executed=False,execution_mode=self.execution_mode)
            threading.Thread(target=self.run,args=(p,),daemon=True).start()
            return self.status()
        except Exception as exc:
            with self.state_lock:self.state.update(phase='failed',reached=False,reason=str(exc))
            if reserved:self.teaching.lock.release()
            self.lock.release();raise

    def run(self,plan):
        records=[];start=plan['start']
        try:
            if plan.get('factory_timed'):
                record=self.manual.execute_path(start,plan['goal'],plan['joint_trajectory'],self.permit,self.revision)
                records.append(record)
                with self.state_lock:
                    self.permit()
                    self.state.update(phase='command_completed',executed=True,command_completed=True,
                                      reached=False,measured=False,state_source=record['source'])
                return
            if plan.get('native'):
                self.permit();motion_budget(self.root)
                record=self.manual.move(start,plan['goal'],expected_stop_revision=self.revision,
                    execution_permit=self.permit,trajectory=plan['joint_trajectory'],
                    source='local_mission' if self.execution_mode=='local_mission' else 'supervised_trajectory')
                records.append(record)
                with self.state_lock:
                    self.permit()  # A late move result cannot overwrite STOP.
                    reached=record.get('attained') is True
                    completed=record.get('command_completed') is True
                    self.state.update(phase='reached' if reached else 'command_completed' if completed else 'unconfirmed',
                                      executed=True,reached=reached,command_completed=completed,
                                      measured=record.get('measured') is True,state_source=record.get('source'))
                return
            for goal in plan['steps']:
                self.permit();motion_budget(self.root)
                observed,_=self.teaching.observation()
                if observed!=start:raise ValueError('Положение руки изменилось')
                record=self.manual.move(start,goal,expected_stop_revision=self.revision,
                                        execution_permit=self.permit,source='local_mission' if self.execution_mode=='local_mission' else 'supervised_trajectory')
                records.append(record);start=goal
                self.state.update(steps=len(records),executed=True)
            with self.state_lock:
                self.permit()
                self.state.update(phase='commanded',reached=False,reason='Команды пути завершены; проверьте физическое положение')
        except Exception as exc:
            with self.state_lock:self.state.update(phase='stopped',reached=False,reason=str(exc))
        finally:
            try:
                folder=self.root/'data/trajectory-runs';folder.mkdir(exist_ok=True)
                with self.state_lock:
                    terminal=dict(self.state)
                    self.last_result=dict(terminal)
                write_json(folder/(self.session+'.json'),dict(plan=plan,commands=records,result=terminal,
                    physical_attainment_verified=terminal.get('phase')=='reached' and terminal.get('reached') is True))
            finally:self.teaching.lock.release();self.lock.release()
