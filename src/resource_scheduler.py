"""Persistent scheduling metadata for heavy, cancellable Explorer jobs."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import uuid


KINDS={'train_act':dict(memory_mb=3200,gpu=True,unit='explorer-train.service'),
       'train_scorer':dict(memory_mb=800,gpu=False,unit='explorer-learning.service'),
       'train_intervention_rl':dict(memory_mb=1800,gpu=True,unit='explorer-train.service'),
       'analyze_episodes':dict(memory_mb=700,gpu=False,unit='explorer-learning.service'),
       'inspect_scene':dict(memory_mb=2600,gpu=True,unit='explorer-llm.service')}


class ResourceScheduler:
    def __init__(self, root):
        self.root=Path(root);(self.root/'data').mkdir(parents=True,exist_ok=True);self.path=self.root/'data/resource-jobs.sqlite3'
        with self.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,request_id TEXT UNIQUE,kind TEXT,
              priority INTEGER,state TEXT,created REAL,updated REAL,spec TEXT,result TEXT,unit TEXT)''')
            db.execute("UPDATE jobs SET state='interrupted',updated=?,result=? WHERE state IN ('starting','running')",
                       (time.time(),json.dumps({'reason':'scheduler_restart'})))

    def db(self):
        db=sqlite3.connect(self.path,timeout=3);db.row_factory=sqlite3.Row;return db

    def submit(self,request_id,kind,spec,priority=50):
        if kind not in KINDS:raise ValueError('Unknown resource job kind')
        if not isinstance(priority,int) or not 0<=priority<=100:raise ValueError('Invalid priority')
        encoded=json.dumps(spec,ensure_ascii=False,allow_nan=False,sort_keys=True)
        with self.db() as db:
            old=db.execute('SELECT * FROM jobs WHERE request_id=?',(request_id,)).fetchone()
            if old:
                if old['spec']!=encoded or old['kind']!=kind:raise ValueError('request_id conflict')
                return dict(old)
            identifier=uuid.uuid4().hex;now=time.time();unit=KINDS[kind]['unit']
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)',(identifier,request_id,kind,priority,'queued',now,now,encoded,'{}',unit))
        return self.get(identifier)

    def cancel(self,identifier):
        job=self.get(identifier)
        if job['state'] in ('starting','running'):
            subprocess.run(['systemctl','--user','stop',job['unit']],capture_output=True,timeout=10,check=False)
        with self.db() as db:db.execute("UPDATE jobs SET state='cancelled',updated=?,result=? WHERE id=?",(time.time(),json.dumps({'reason':'operator_cancelled'}),identifier))
        return self.get(identifier)

    def get(self,identifier):
        with self.db() as db:row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
        if not row:raise ValueError('Resource job not found')
        value=dict(row);value['spec']=json.loads(value['spec']);value['result']=json.loads(value['result']);return value

    def status(self):
        with self.db() as db:rows=[dict(row) for row in db.execute('SELECT * FROM jobs ORDER BY priority DESC,created')]
        memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2}
        disk=shutil.disk_usage(self.root)
        return dict(jobs=rows,ram_available_mb=memory.get('MemAvailable',0)//1024,disk_free_gb=round(disk.free/1024**3,2),
                    concurrent_heavy_limit=1,control_services_preempt_heavy_jobs=True)

    def next(self):
        with self.db() as db:
            active=db.execute("SELECT count(*) FROM jobs WHERE state IN ('starting','running')").fetchone()[0]
            row=None if active else db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY priority DESC,created LIMIT 1").fetchone()
        if not row:return None
        requirement=KINDS[row['kind']];status=self.status()
        if status['ram_available_mb']<requirement['memory_mb'] or status['disk_free_gb']<2:return None
        return self.get(row['id'])

    def mark(self,identifier,state,result=None):
        if state not in ('starting','running','completed','failed','interrupted','cancelled'):raise ValueError('Invalid resource job state')
        with self.db() as db:db.execute('UPDATE jobs SET state=?,updated=?,result=? WHERE id=?',(state,time.time(),json.dumps(result or {}),identifier))
        return self.get(identifier)
