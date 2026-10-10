"""Local LeRobot jobs. This module has no robot publishers or motor interfaces."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import fcntl
from stored_records import records as stored_records

ROOT=Path('/home/vlad/Explorer')

def write_json(path, value):
    temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False));temp.replace(path)
    finally:
        if temp.exists():temp.unlink()


def save_job(path,record,terminal=False):
    """Cancellation survives a concurrent progress write from the learner."""
    with (path.parent.parent/"dispatch.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        current=json.loads(path.read_text()) if path.exists() else {}
        if current.get("cancel_requested"):
            if not terminal:raise InterruptedError("Обучение отменено оператором")
            record.update(state="cancelled",cancel_requested=True)
        write_json(path,record)

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
    for path in (root/"data/mobile-demonstrations").glob("*/episode.json"):
        record=json.loads(path.read_text())
        if record.get("state")=="recording":raise ValueError("Обучение отложено: идёт запись показа")
    manual_path=root/"data/manual-teleop.json"
    if manual_path.exists():
        manual=json.loads(manual_path.read_text())
        if time.time()-manual.get("at",0)<1 and any(abs(float(value))>.02 for value in
            (manual.get("normalized_actions") or {}).values()):
            raise ValueError("Обучение отложено: ручное управление имеет приоритет")
    return p

class LearningJobs:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.folder=self.root/'data/learning-jobs';self.folder.mkdir(exist_ok=True)
        self.lock=threading.Lock();self.process=None

    def status(self):
        records=[]
        rows,errors=stored_records(sorted(self.folder.glob("*/job.json")),("id","at","state"))
        for path,record in rows:
            records.append(record)
        backend=self.root/'data/learning-backend.json'
        return dict(backend=json.loads(backend.read_text()) if backend.exists() else {'ready':False},
                    record_errors=errors,jobs=records,automatic_execution=False,
                    pending=sum(record.get("state") in ("queued","deferred") for record in records))

    def start(self,steps,task):
        if type(steps) is not int or steps not in (1000,5000,20000):raise ValueError('Неизвестная длительность обучения')
        from teaching import Demonstrations
        demos=[e for e in Demonstrations(self.root/'data/demonstrations').eligible() if e['name']==task]
        if not demos:raise ValueError('Нужен хотя бы один успешный показ для ACT; самостоятельные попытки обучают scorer без ACT')
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

    def start_mobile(self,steps,task,skill_id=None):
        if type(steps) is not int or steps not in (1000,5000,20000):raise ValueError('Неизвестная длительность обучения')
        from mobile_demonstrations import episode_quality
        source=self.root/'data/mobile-demonstrations';episodes=[]
        rows,_=stored_records(sorted(source.glob("*/episode.json")),("id","state"))
        for path,episode in rows:
            quality=episode.get("quality") or episode_quality(path.parent)
            same_skill=episode.get("skill_id")==skill_id if skill_id else episode.get("name")==task
            if (same_skill and episode.get("state")=="complete" and episode.get("outcome")=="success" and
                    episode.get("label_source")=="operator" and quality.get("usable") is True):episodes.append(episode)
        if not episodes:raise ValueError('Нужен хотя бы один успешный полный показ для ACT; фиксированного минимума нет')
        inputs={e["id"]:{name:hashlib.sha256((source/e["id"]/name).read_bytes()).hexdigest()
            for name in ("episode.json","samples.jsonl")} for e in episodes}
        fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        rows,_=stored_records(self.folder.glob("*/job.json"),("id","state"))
        for path,existing in rows:
            if (existing.get("dataset_fingerprint")==fingerprint and existing.get("skill_id")==skill_id and
                    existing.get("state")!="superseded"):
                return existing
        if not self.lock.acquire(blocking=False):raise ValueError('Обучение уже выполняется')
        try:
            # Stable IDs prevent duplicate GPU jobs after process restarts. Data
            # remains on disk even while power or foreground work defers launch.
            ident=fingerprint[:32];folder=self.folder/ident;folder.mkdir(exist_ok=True)
            if (folder/"job.json").exists():return json.loads((folder/"job.json").read_text())
            record=dict(id=ident,at=time.time(),state='queued',steps=steps,episodes=[e['id'] for e in episodes],
                task=task,framework='lerobot',policy='act',dataset_kind='mobile_manipulation_9dof',executed=False,
                automatic_execution=False,skill_id=skill_id,episode_count=len(episodes),
                dataset_fingerprint=fingerprint,input_sha256=inputs,queue_version=1,created=time.time())
            write_json(folder/"job.json",record)
            return record
        finally:self.lock.release()

    def dispatch(self):
        """Admit one durable job; this never grants permission to move."""
        with (self.folder/"dispatch.lock").open("a") as guard:
            try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return {"state":"another_dispatcher"}
            try:
                active=subprocess.run(["systemctl","--user","is-active","explorer-train.service"],
                    capture_output=True,text=True,timeout=2)
            except (OSError,subprocess.SubprocessError) as exc:
                return {"state":"deferred","deferred_reason":str(exc)}
            if active.stdout.strip() in ("active","activating","deactivating"):
                return {"state":"busy"}
            rows,_=stored_records(self.folder.glob("*/job.json"),("id","at","state"))
            for path,record in rows:
                if record.get("state") in ("exporting","training","validating") and time.time()-record["at"]>15:
                    record.update(state="deferred",at=time.time(),deferred_reason="Служба прервана; продолжение из сохранённой точки")
                    write_json(path,record)
            pending=[(path,record) for path,record in rows if record.get("state") in ("queued","deferred")]
            pending.sort(key=lambda row:(row[1].get("episode_count",0),row[1].get("created",row[1]["at"])))
            newest={}
            for path,record in pending:
                key=record.get("skill_id") or record.get("task")
                newest[key]=(path,record)
            for path,record in pending:
                key=record.get("skill_id") or record.get("task")
                if newest[key][1]["id"]!=record["id"]:
                    record.update(state="superseded",superseded_by=newest[key][1]["id"])
                    write_json(path,record)
            waiting=sorted(newest.values(),key=lambda row:row[1].get("created",row[1]["at"]))
            if not waiting:return {"state":"idle"}
            path,record=waiting[0]
            try:
                if not self.status()["backend"].get("ready"):raise ValueError("Ожидается проверенная среда LeRobot")
                training_budget(self.root)
            except (OSError,ValueError,KeyError) as exc:
                record.update(state="deferred",deferred_reason=str(exc))
                write_json(path,record)
                return record
            write_json(self.root/"data/learning-request.json",{"job":record["id"]})
            record.update(state="queued",at=time.time(),deferred_reason=None)
            write_json(path,record)
            try:subprocess.run(["systemctl","--user","start","explorer-train.service"],check=True,timeout=8)
            except (OSError,subprocess.SubprocessError) as exc:
                record.update(state="deferred",deferred_reason="Служба обучения не запустилась: "+str(exc))
                write_json(path,record)
            return record

    def stop(self):
        with (self.folder/"dispatch.lock").open("a") as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            rows,_=stored_records(self.folder.glob("*/job.json"),("id","at","state"))
            for path,record in rows:
                if record.get("state") in ("queued","deferred","exporting","training","validating"):
                    record.update(state="cancelled",cancel_requested=True,at=time.time())
                    write_json(path,record)
        subprocess.run(['systemctl','--user','stop','explorer-train.service'],check=True,timeout=10)
        return {'stopped':True,'robot_motion':False}

    def log(self,ident):
        if not ident or any(c not in '0123456789abcdef-' for c in ident):raise ValueError('Некорректная запись')
        path=self.folder/ident/'train.log'
        return path.read_text(errors='replace')[-12000:] if path.exists() else ''
