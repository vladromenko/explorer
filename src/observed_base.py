"""Finite manual motions only. Never clears STOP or grants autonomous readiness."""
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

def arm_command_pending(root,status):
    """Core snapshots lag the arm executor by up to one persistence interval."""
    cached=status.get('arm_command_state',{})
    if cached.get('phase')!='command_in_progress':return False
    try:current=json.loads((Path(root)/'data/arm-state.json').read_text())
    except (OSError,ValueError):return True
    settled=(current.get('boot_id')==status.get('boot_id') and bool(status.get('boot_id'))
             and current.get('phase')=='command_elapsed_observation_required'
             and current.get('at',0)>=cached.get('at',float('inf'))
             and current.get('ends_monotonic',float('inf'))+.15<=time.monotonic())
    return not settled

class ObservedBase:
    def __init__(self,root,arm_busy=lambda:False):
        self.root=Path(root);self.lock=threading.Lock();self.state=dict(phase='idle')
        self.arm_busy=arm_busy
    def status(self):return dict(self.state,busy=self.lock.locked(),mode='observed_finite',unattended=False)
    def start(self,direction,duration,observing,compact):
        if direction not in ('forward','backward','left','right','ccw','cw'):raise ValueError('Неизвестное направление')
        if type(duration) not in (int,float) or not .1<=duration<=2:raise ValueError('Длительность: 0,1–2 секунды')
        if observing is not True or compact is not True:raise ValueError('Нужно наблюдение, свободный путь и сложенная рука')
        if self.arm_busy():raise ValueError('Дождитесь завершения конечного движения руки')
        s=json.loads((self.root/'data/status.json').read_text())
        if not 0<=time.time()-s.get('at',0)<1:raise ValueError('Нет свежего состояния')
        if s.get('stop_latched',True):raise ValueError('Сначала явно снимите STOP')
        if s.get('mode')!='MANUAL' or s.get('mission'):raise ValueError('Выберите ручной режим без задания')
        if arm_command_pending(self.root,s):raise ValueError('Дождитесь завершения движения руки')
        if not self.lock.acquire(False):raise ValueError('Предыдущее перемещение ещё выполняется')
        self.state=dict(phase='running',id=uuid.uuid4().hex,direction=direction,duration=duration,at=time.time())
        threading.Thread(target=self.run,args=(direction,duration),daemon=True).start()
        return self.status()
    def run(self,direction,duration):
        try:
            speed=.12 if direction in ('cw','ccw') else .04
            result=subprocess.run(['python3',str(self.root/'bin/commission-base.py'),direction,
                '--duration',str(duration),'--speed',str(speed),'--already-armed'],
                capture_output=True,text=True,timeout=20,cwd=self.root)
            if result.returncode:raise ValueError(result.stderr[-1200:] or result.stdout[-1200:])
            report=json.loads(result.stdout.strip().splitlines()[-1])
            stopped=all(abs(v)<.005 for v in report['final_velocity'].values())
            completed=report.get('execution_outcome',{}).get('completed') is True
            self.state.update(phase=('completed' if completed else 'interrupted') if stopped else 'unknown',result=report,at=time.time())
        except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
            self.state.update(phase='failed',error=str(exc),at=time.time())
        finally:self.lock.release()
