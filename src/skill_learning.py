"""Persistent, operator-facing skill catalogue and automatic local training queue.

The coordinator never has actuator access. A completed demonstration is not a
successful one unless the operator said so, and a trained policy is only a
candidate until independent validation and a separate observed trial.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
import uuid

from mobile_demonstrations import episode_quality


class SkillLearning:
    def __init__(self, root, recorder, jobs, interval=5.0):
        self.root=Path(root)
        self.recorder=recorder
        self.jobs=jobs
        self.folder=self.root/"data/skill-learning"
        self.folder.mkdir(parents=True,exist_ok=True)
        self.database=self.folder/"skills.sqlite3"
        self.lock=threading.RLock()
        self.interval=interval
        self.closed=False
        self._schema()
        self._recover_links()
        if interval>0:
            threading.Thread(target=self._loop,daemon=True,name="explorer-skill-queue").start()

    def db(self):
        connection=sqlite3.connect(self.database,timeout=10)
        connection.row_factory=sqlite3.Row
        return connection

    def _schema(self):
        with self.db() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS skills(
                id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL,
                updated REAL NOT NULL, next_note TEXT NOT NULL DEFAULT 'Покажи пример');
              CREATE TABLE IF NOT EXISTS episodes(
                id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, outcome TEXT NOT NULL,
                state TEXT NOT NULL, usable INTEGER NOT NULL, quality TEXT NOT NULL,
                completed REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS training(
                fingerprint TEXT PRIMARY KEY, skill_id TEXT NOT NULL, job_id TEXT NOT NULL,
                created REAL NOT NULL);
            """)

    def _recover_links(self):
        for path in (self.root/"data/mobile-demonstrations").glob("*/episode.json"):
            try:
                record=json.loads(path.read_text())
                if record.get("skill_id") and record.get("state")!="recording":
                    self.ingest(record,path,schedule=False)
            except (OSError,ValueError,KeyError,sqlite3.Error,subprocess.SubprocessError):
                pass

    def create(self,name):
        title=str(name).strip()
        if not 3<=len(title)<=80:raise ValueError("Назовите навык в 3–80 символов")
        with self.lock,self.db() as db:
            existing=db.execute("SELECT id FROM skills WHERE name=? ORDER BY created LIMIT 1",(title,)).fetchone()
            if existing:return self.get(existing["id"])
            identifier=uuid.uuid4().hex
            now=time.time()
            db.execute("INSERT INTO skills(id,name,created,updated) VALUES(?,?,?,?)",
                       (identifier,title,now,now))
        return self.get(identifier)

    def rename(self,identifier,name):
        title=str(name).strip()
        if not 3<=len(title)<=80:raise ValueError("Назовите навык в 3–80 символов")
        with self.lock,self.db() as db:
            if not db.execute("SELECT 1 FROM skills WHERE id=?",(identifier,)).fetchone():
                raise ValueError("Навык не найден")
            db.execute("UPDATE skills SET name=?,updated=? WHERE id=?",(title,time.time(),identifier))
        return self.get(identifier)

    def get(self,identifier):
        if not isinstance(identifier,str) or len(identifier)!=32 or any(ch not in "0123456789abcdef" for ch in identifier):
            raise ValueError("Некорректный идентификатор навыка")
        with self.db() as db:
            row=db.execute("SELECT * FROM skills WHERE id=?",(identifier,)).fetchone()
            if row is None:raise ValueError("Навык не найден")
            episodes=[dict(item) for item in db.execute(
                "SELECT * FROM episodes WHERE skill_id=? ORDER BY completed DESC",(identifier,)).fetchall()]
            jobs=[dict(item) for item in db.execute(
                "SELECT * FROM training WHERE skill_id=? ORDER BY created DESC",(identifier,)).fetchall()]
        data=dict(row)
        data["episodes"]=[dict(item,quality=json.loads(item["quality"])) for item in episodes]
        data["successful_usable"]=sum(item["outcome"]=="success" and item["usable"] for item in episodes)
        data["failed"]=sum(item["outcome"]=="failure" for item in episodes)
        data["training_jobs"]=jobs
        latest=self.jobs.status()["jobs"]
        if jobs:
            data["latest_job"]=next((item for item in latest if item["id"]==jobs[0]["job_id"]),None)
        else:data["latest_job"]=None
        if data["latest_job"]:
            state=data["latest_job"]["state"]
            if state in ("queued","exporting","training","validating"):
                data["next_action"]="Данные обрабатываются на Explorer"
            elif state=="validated_offline":
                data["next_action"]="Модель готова к отдельной наблюдаемой проверке"
            elif state=="trained_unvalidated":
                data["next_action"]="Покажи ещё пример для независимой проверки"
            elif state in ("failed","interrupted"):
                data["next_action"]="Обучение остановилось: открой диагностику"
            else:data["next_action"]="Покажи ещё пример"
        elif episodes and episodes[0]["outcome"]=="success" and not episodes[0]["usable"]:
            data["next_action"]="Нужен новый показ: "+json.loads(episodes[0]["quality"]).get("reason","")
        elif data["successful_usable"]:
            data["next_action"]=data["next_note"]
        else:data["next_action"]="Покажи пример"
        return data

    def list(self):
        with self.db() as db:
            ids=[row["id"] for row in db.execute("SELECT id FROM skills ORDER BY updated DESC")]
        return [self.get(identifier) for identifier in ids]

    def ingest(self,record,path,schedule=True):
        identifier=record.get("skill_id")
        if not identifier:return None
        quality=record.get("quality") or episode_quality(Path(path).parent)
        with self.lock,self.db() as db:
            if not db.execute("SELECT 1 FROM skills WHERE id=?",(identifier,)).fetchone():
                raise ValueError("Запись ссылается на неизвестный навык")
            inserted=db.execute("INSERT INTO episodes(id,skill_id,outcome,state,usable,quality,completed) "
                       "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                       "outcome=excluded.outcome,state=excluded.state,usable=excluded.usable,"
                       "quality=excluded.quality,completed=excluded.completed",
                       (record["id"],identifier,record.get("outcome","unknown"),record.get("state","interrupted"),
                        int(quality.get("usable") is True),json.dumps(quality,ensure_ascii=False),
                        float(record.get("ended") or time.time())))
            if inserted.rowcount:
                db.execute("UPDATE skills SET updated=? WHERE id=?",(time.time(),identifier))
        if schedule:self.tick()
        return self.get(identifier)

    def tick(self):
        with self.lock:
            self._recover_links()
            for skill in self.list():
                eligible=[item for item in skill["episodes"] if item["outcome"]=="success" and
                          item["state"]=="complete" and item["usable"]]
                if not eligible:pass
                else:
                    ids=sorted(item["id"] for item in eligible)
                    fingerprint=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    if not any(item["fingerprint"]==fingerprint for item in skill["training_jobs"]):
                        active=any(item.get("state") in ("queued","exporting","training","validating")
                                   for item in self.jobs.status()["jobs"])
                        if not active:
                            try:
                                job=self.jobs.start_mobile(1000,skill["name"],skill_id=skill["id"])
                            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
                                with self.db() as db:
                                    db.execute("UPDATE skills SET next_note=? WHERE id=?",(str(exc),skill["id"]))
                            else:
                                with self.db() as db:
                                    db.execute("INSERT OR IGNORE INTO training VALUES(?,?,?,?)",
                                               (fingerprint,skill["id"],job["id"],time.time()))

    def _loop(self):
        while not self.closed:
            try:self.tick()
            except (OSError,ValueError,KeyError,sqlite3.Error,subprocess.SubprocessError):pass
            time.sleep(self.interval)
