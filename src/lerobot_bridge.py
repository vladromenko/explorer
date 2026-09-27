"""Local LeRobot jobs. This module has no robot publishers or motor interfaces."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

ROOT=Path('/home/vlad/Explorer')

def write_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False));temp.replace(path)

def training_budget(root=ROOT):
    p=json.loads((root/'data/power.json').read_text())
    if not 0<=time.time()-p['at']<4:raise ValueError('Нет свежих данных питания')
    if p['state'] not in ('NORMAL','IDLE') or p['battery_voltage_v']<11.5:
        raise ValueError('Обучение отложено: зарядите выключенного робота; нужно не менее 11,5 В')
    s=json.loads((root/'data/status.json').read_text())
    if not 0<=time.time()-s['at']<2 or s.get('stop_latched') is not True:
        raise ValueError('Для обучения включите стоп шасси')
    arm=json.loads((root/'data/arm-state.json').read_text())
    if arm.get('phase')=='command_in_progress':raise ValueError('Рука ещё движется')
    return p

class LearningJobs:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.folder=self.root/'data/learning-jobs';self.folder.mkdir(exist_ok=True)
        self.lock=threading.Lock();self.process=None

    def status(self):
        records=[]
        for path in sorted(self.folder.glob('*/job.json')):
            record=json.loads(path.read_text())
            if record.get('state') in ('queued','exporting','training','validating') and time.time()-record['at']>15:
                active=subprocess.run(['systemctl','--user','is-active','explorer-train.service'],capture_output=True,text=True,timeout=2)
                if active.stdout.strip() not in ('active','activating','deactivating'):
                    record.update(state='interrupted',error='Служба завершилась или робот перезапущен; автоматического продолжения нет')
                    write_json(path,record)
            records.append(record)
        backend=self.root/'data/learning-backend.json'
        return dict(backend=json.loads(backend.read_text()) if backend.exists() else {'ready':False},
                    jobs=records[-12:],automatic_execution=False)

    def start(self,steps,task):
        if type(steps) is not int or steps not in (1000,5000,20000):raise ValueError('Неизвестная длительность обучения')
        from teaching import Demonstrations
        demos=[e for e in Demonstrations(self.root/'data/demonstrations').eligible() if e['name']==task]
        if len(demos)<10:raise ValueError('Сначала запишите минимум 10 успешных показов; лучше 30–50 в разных положениях')
        backend=self.status()['backend']
        if not backend.get('ready'):raise ValueError('Среда LeRobot ещё не прошла проверку')
        training_budget(self.root)
        if not self.lock.acquire(blocking=False):raise ValueError('Обучение уже выполняется')
        try:
            # One reserved systemd unit prevents overlapping GPU jobs after a web restart.
            active=subprocess.run(['systemctl','--user','is-active','explorer-train.service'],capture_output=True,text=True)
            if active.stdout.strip() in ('active','activating','deactivating'):raise ValueError('Обучение уже выполняется')
            ident=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            folder=self.folder/ident;folder.mkdir()
            record=dict(id=ident,at=time.time(),state='queued',steps=steps,
                        episodes=[d['id'] for d in demos],task=task,framework='lerobot',policy='act',
                        executed=False,automatic_execution=False)
            write_json(folder/'job.json',record)
            write_json(self.root/'data/learning-request.json',dict(job=ident))
            subprocess.run(['systemctl','--user','start','explorer-train.service'],check=True,timeout=8)
            return record
        finally:self.lock.release()

    def stop(self):
        subprocess.run(['systemctl','--user','stop','explorer-train.service'],check=True,timeout=10)
        return {'stopped':True,'robot_motion':False}

    def log(self,ident):
        if not ident or any(c not in '0123456789abcdef-' for c in ident):raise ValueError('Некорректная запись')
        path=self.folder/ident/'train.log'
        return path.read_text(errors='replace')[-12000:] if path.exists() else ''
