"""Persistent high-level autonomy and bounded experiment series.

The supervisor owns goals and plans, never a motor publisher.  Its gateway is
the already existing delivery/mission executor.  Browser state is not used.
"""
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from autonomy_contracts import GoalSpec, HelpRequest, PolicyVersion, Truth, new_id, record
from episode_store import EpisodeStore
from learning_stack import CandidateScorer, PolicyRegistry, backend_status
from skill_planner import SkillRegistry, TaskPlanner, recovery_plan


TERMINAL=("succeeded","failed","cancelled","blocked","budget_exhausted","interrupted")


class AutonomySupervisor:
    def __init__(self, root, world_provider, readiness_provider, action_gateway, reset_gateway=None):
        self.root=Path(root);self.path=self.root/'data/autonomy.sqlite3';self.lock=threading.RLock()
        self.world_provider=world_provider;self.readiness_provider=readiness_provider
        self.action_gateway=action_gateway;self.reset_gateway=reset_gateway
        self.registry=SkillRegistry(root);self.planner=TaskPlanner(self.registry);self.episodes=EpisodeStore(root)
        self.scorer=CandidateScorer(root);self.policies=PolicyRegistry(root);self.stop_event=threading.Event()
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, request_id TEXT UNIQUE, kind TEXT, state TEXT,
              created REAL, updated REAL, spec TEXT, snapshot TEXT, plan TEXT, result TEXT, current_episode TEXT,
              cancel_requested INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, job_id TEXT, at REAL, phase TEXT, content TEXT);
            CREATE TABLE IF NOT EXISTS help(id TEXT PRIMARY KEY, job_id TEXT, state TEXT, request TEXT, answer TEXT);
            CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, job_id TEXT, episode_id TEXT, state TEXT,
              outcome TEXT, evidence TEXT, started REAL, ended REAL);
            ''')
            rows=db.execute("SELECT id FROM jobs WHERE state IN ('queued','observing','planning','validating','executing','verifying','recording','resetting')").fetchall()
            for row in rows:
                db.execute("UPDATE jobs SET state='interrupted',updated=?,result=? WHERE id=?",
                           (time.time(),json.dumps({"recovered":True,"reason":"Process restart: explicit operator resumption required; no motion replay"}),row[0]))
        self.episodes.recover();self.thread=threading.Thread(target=self.loop,daemon=True,name='autonomy-supervisor');self.thread.start()

    def db(self):
        db=sqlite3.connect(self.path,timeout=3);db.row_factory=sqlite3.Row;return db

    def submit(self, request_id, spec):
        if not isinstance(request_id,str) or not 16<=len(request_id)<=100:raise ValueError('Stable request_id required')
        spec=dict(spec)
        goal=GoalSpec(spec.get('goal_id') or request_id,str(spec.get('goal','')).strip(),
                      dict(spec.get('target_predicates') or {'placed':True}),
                      object_query=str(spec.get('object_query','')),destination=dict(spec.get('destination') or {}),
                      constraints=dict(spec.get('constraints') or {}),created_at=float(spec.get('created_at',0))).validate()
        spec['goal_spec']=record(goal);spec['goal_id']=goal.id
        encoded=json.dumps(spec,ensure_ascii=False,allow_nan=False,sort_keys=True)
        if len(encoded)>20000:raise ValueError('Autonomy job is too large')
        with self.db() as db:
            old=db.execute('SELECT * FROM jobs WHERE request_id=?',(request_id,)).fetchone()
            if old:
                if old['spec']!=encoded:raise ValueError('request_id already belongs to another job')
                return self.row(old)
            identifier=new_id();now=time.time()
            db.execute('INSERT INTO jobs(id,request_id,kind,state,created,updated,spec,snapshot,plan,result,current_episode) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                       (identifier,request_id,spec.get('kind','goal'),'queued',now,now,encoded,'{}','{}','{}',None))
            self.event(db,identifier,'queued',{'spec':spec})
        return self.get(identifier)

    def cancel(self, identifier=None, reason='Manual takeover or STOP'):
        with self.db() as db:
            query="SELECT id FROM jobs WHERE state NOT IN ('succeeded','failed','cancelled','budget_exhausted')"
            rows=db.execute(query).fetchall() if identifier is None else db.execute(query+' AND id=?',(identifier,)).fetchall()
            for row in rows:
                db.execute("UPDATE jobs SET cancel_requested=1,state='cancelled',updated=?,result=? WHERE id=?",
                           (time.time(),json.dumps({'reason':reason}),row['id']))
                db.execute("UPDATE help SET state='cancelled',answer=? WHERE job_id=? AND state='open'",
                           (json.dumps({'reason':reason}),row['id']))
                self.event(db,row['id'],'cancelled',{'reason':reason})
        return {'cancelled':[row['id'] for row in rows]}

    def resume(self, identifier):
        with self.db() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row:raise ValueError('Job not found')
            if row["state"] not in ("blocked","failed","cancelled","interrupted"):raise ValueError("Only stopped jobs can be resumed")
            db.execute("UPDATE jobs SET state='queued',cancel_requested=0,updated=?,snapshot='{}',plan='{}',result='{}' WHERE id=?",
                       (time.time(),identifier));self.event(db,identifier,'resumed',{})
        return self.get(identifier)

    def answer(self, help_id, answer):
        with self.db() as db:
            row=db.execute("SELECT * FROM help WHERE id=? AND state='open'",(help_id,)).fetchone()
            if not row:raise ValueError('Open help request not found')
            request=json.loads(row['request']);db.execute("UPDATE help SET state='answered',answer=? WHERE id=?",(json.dumps(answer),help_id))
            db.execute("UPDATE jobs SET state='queued',updated=? WHERE id=?",(time.time(),row['job_id']))
            self.event(db,row['job_id'],'help_answered',{'help_id':help_id,'answer':answer,'scope':request.get('blocks_skill')})
        return self.get(row['job_id'])

    def get(self, identifier):
        with self.db() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if not row:raise ValueError('Job not found')
            result=self.row(row);result['events']=[dict(item,content=json.loads(item['content'])) for item in db.execute('SELECT * FROM events WHERE job_id=? ORDER BY id',(identifier,))]
            result['help']=[dict(item,request=json.loads(item['request']),answer=json.loads(item['answer']) if item['answer'] else None) for item in db.execute('SELECT * FROM help WHERE job_id=? ORDER BY rowid',(identifier,))]
            return result

    def status(self):
        with self.db() as db:
            rows=[self.row(row) for row in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 30')]
            helps=[dict(row,request=json.loads(row['request'])) for row in db.execute("SELECT * FROM help WHERE state='open' ORDER BY rowid")]
        readiness=self.readiness_provider()
        return dict(at=time.time(),jobs=rows,active=[row for row in rows if row['state'] not in TERMINAL],help=helps,
                    skills=self.registry.catalog(),learning=backend_status(self.root),storage=self.episodes.usage(),
                    hardware_readiness=readiness.get('hardware',{}),experiment_permissions=readiness.get('permissions',{}),
                    skill_quality=dict(accepted_policies=len([row for row in self.policies.all() if row['state']=='accepted']),
                                       scorer_version=self.scorer.version),
                    browser_required=False,motor_owner='existing delivery/mission executors')

    @staticmethod
    def row(row):
        result=dict(row)
        for key in ('spec','snapshot','plan','result'):result[key]=json.loads(result[key] or '{}')
        result['cancel_requested']=bool(result['cancel_requested'])
        return result

    @staticmethod
    def event(db,job_id,phase,content):
        db.execute('INSERT INTO events(job_id,at,phase,content) VALUES(?,?,?,?)',(job_id,time.time(),phase,json.dumps(content,ensure_ascii=False,allow_nan=False)))

    def set_phase(self, identifier, phase, **values):
        fields=['state=?','updated=?'];args=[phase,time.time()]
        for key,value in values.items():fields.append(key+'=?');args.append(json.dumps(value,ensure_ascii=False,allow_nan=False) if key in ('snapshot','plan','result') else value)
        args.append(identifier)
        with self.db() as db:
            updated=db.execute("UPDATE jobs SET "+",".join(fields)+" WHERE id=? AND cancel_requested=0",args)
            if updated.rowcount:self.event(db,identifier,phase,values)

    def help_request(self, job, question, kind, choices, skill):
        item=HelpRequest(new_id(),job['id'],question,kind,tuple(choices),skill,expires_at=time.time()+86400).validate()
        with self.db() as db:
            db.execute('INSERT INTO help VALUES(?,?,?,?,?)',
                       (item.id,item.job_id,'open',json.dumps(record(item),ensure_ascii=False),''))
            db.execute("UPDATE jobs SET state='blocked',updated=?,result=? WHERE id=?",(time.time(),json.dumps({'reason':question,'help_id':item.id}),job['id']))
            self.event(db,job['id'],'help_requested',record(item))
        return item

    def loop(self):
        while not self.stop_event.wait(.2):
            row=None
            try:
                with self.db() as db:row=db.execute("SELECT * FROM jobs WHERE state='queued' AND cancel_requested=0 ORDER BY created LIMIT 1").fetchone()
                if row:self.process(self.row(row))
            except Exception as exc:
                if row:self.set_phase(row['id'],'failed',result={'reason':str(exc),'kind':'infrastructure_error'})

    def process(self, job):
        spec=job['spec'];attempts=int(spec.get('attempts',1));time_budget=float(spec.get('time_budget_s',900))
        if not 1<=attempts<=500 or not 30<=time_budget<=86400:raise ValueError('Invalid autonomy budget')
        started=time.monotonic();counts={'proposed':0,'rejected':0,'started':0,'completed':0,'success':0,'failure':0,
                                       'unknown':0,'interventions':0,'resets':0,'reset_failures':0,'infrastructure_errors':0}
        for attempt in range(1,attempts+1):
            if self.cancelled(job['id']):return
            if time.monotonic()-started>time_budget:
                self.set_phase(job['id'],'budget_exhausted',result={'counts':counts,'reason':'time_budget'});return
            self.set_phase(job['id'],'observing',result={'counts':counts,'attempt':attempt})
            snapshot=self.world_provider();scene=str(snapshot.get('scene_version') or snapshot.get('map_epoch') or 'unknown')
            predicates={key:(value if value in ('true','false','unknown') else 'true' if value is True else 'false' if value is False else 'unknown')
                        for key,value in snapshot.get('predicates',{}).items()}
            self.set_phase(job['id'],'planning',snapshot=snapshot)
            target=spec.get('target_predicates',{'placed':True});permissions=tuple(spec.get('permissions',()))
            plan=self.planner.plan(predicates,target,scene,spec.get('allowed_skills'),permissions);counts['proposed']+=len(plan['steps'])
            if plan.get('blocked'):
                self.set_phase(job['id'],'blocked',plan=plan,result={'counts':counts,'reason':plan['reason']});return
            self.set_phase(job['id'],'validating',plan=plan)
            readiness=self.readiness_provider();missing=self.validate_plan(plan,readiness,spec)
            if missing:
                counts['rejected']+=1
                actionable=[item for item in missing if item.startswith('укажите') or item.startswith('enable permission')]
                if actionable:
                    self.help_request(job,'Нужно одно действие: '+actionable[0],'readiness',actionable,'validate_plan')
                else:
                    self.set_phase(job['id'],'blocked',plan=plan,result={'counts':counts,
                        'reason':'Physical dependency is not accepted','blocked_by':missing})
                return
            current=self.world_provider();current_scene=str(current.get('scene_version') or current.get('map_epoch') or 'unknown')
            if current_scene!=scene:
                self.set_phase(job['id'],'queued',result={'counts':counts,'reason':'Scene changed before execution; replanning'});return
            episode=self.episodes.begin(job['id'],spec.get('goal',''),scene,spec.get('policy_version','geometry-baseline'),spec.get('reward_version','outcome_v1'),{'attempt':attempt})
            self.set_phase(job['id'],'executing',current_episode=episode['id']);counts['started']+=1
            outcome=self.action_gateway(spec,plan,episode['id'])
            if outcome.get('state') not in ('success','failure','unknown','cancelled','infrastructure_error'):
                outcome={'state':'unknown','reason':'Gateway returned no independently verified outcome','evidence':outcome}
            self.set_phase(job['id'],'verifying',result={'counts':counts,'latest':outcome})
            self.episodes.finish(episode['id'],outcome['state'],outcome);counts['completed']+=1
            count_key="infrastructure_errors" if outcome["state"]=="infrastructure_error" else outcome["state"]
            counts[count_key if count_key in counts else "unknown"]+=1
            self.record_scorer_sample(episode['id'],outcome)
            with self.db() as db:
                db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)',(new_id(),job['id'],episode['id'],'complete',outcome['state'],json.dumps(outcome,ensure_ascii=False),episode['started_at'],time.time()))
            if outcome["state"]=="cancelled":
                self.cancel(job["id"],reason=outcome.get("reason") or "Executor cancelled")
            if self.cancelled(job["id"]):return
            if attempt<attempts:
                self.set_phase(job['id'],'resetting',result={'counts':counts})
                reset=self.reset_gateway(spec,episode['id']) if self.reset_gateway else {'state':'unknown','reason':'No physical reset gateway'}
                if reset.get('state')=='success':counts['resets']+=1
                else:
                    counts['reset_failures']+=1
                    self.set_phase(job['id'],'blocked',result={'counts':counts,'reason':'Reset did not establish the next observable start state','reset':reset});return
        completed=counts["success"]==attempts
        self.set_phase(job["id"],"succeeded" if completed else "failed",
                       result={"counts":counts,"goal_completed":completed})

    def record_scorer_sample(self,episode_id,outcome):
        events=outcome.get('events',[]) if isinstance(outcome,dict) else []
        approaches=[row.get('result',{}).get('grasp_selection') for row in events if row.get('stage','').startswith('approach')]
        holds=[row.get('result',{}) for row in events if row.get('stage') in ('verify_hold','verify_regrasp')]
        selection=next((row for row in reversed(approaches) if isinstance(row,dict) and row.get('rows')),None)
        hold=holds[-1] if holds else None
        if selection and hold and hold.get('outcome') in ('success','failure'):
            selected=selection['rows'][selection['selected']]['candidate']
            sample=dict(episode_id=episode_id,context=selection.get('context',{}),action=selected,
                        outcome=hold['outcome'],policy_version=selection.get('policy_version'),
                        verifier=hold.get('evidence',{}).get('verifier'),recorded_at=time.time())
            path=self.root/'data/models/grasp-scorer/samples.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as stream:stream.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows)>=20 and len(rows)%10==0:
                try:
                    candidate=self.scorer.train(rows)
                    version=PolicyVersion(candidate['id'],'contextual_logistic_scorer',candidate['feature_version'],
                        'candidate',str(self.scorer.root/(candidate['id']+'.json')),
                        tuple(candidate['train_episodes']),tuple(candidate['validation_episodes']),
                        dict(candidate['metrics'],skill='grasp'))
                    try:self.policies.add(version)
                    except ValueError:pass
                except (OSError,ValueError,KeyError,TypeError):pass

    def cancelled(self, identifier):
        with self.db() as db:row=db.execute('SELECT cancel_requested,state FROM jobs WHERE id=?',(identifier,)).fetchone()
        return not row or bool(row['cancel_requested']) or row['state']=='cancelled'

    @staticmethod
    def validate_plan(plan, readiness, spec):
        missing=[];hardware=readiness.get('hardware',readiness);permission=readiness.get('permissions',{})
        if spec.get('task_type')=='delivery' and not spec.get('object_query'):
            missing.append('укажите, какой предмет искать')
        if spec.get('task_type')=='delivery' and not (spec.get('destination') or {}).get('name'):
            missing.append('укажите целевое место')
        for step in plan['steps']:
            if step['executor'] in ('delivery_robot','missions') and hardware.get('physical_execution_ready') is not True:
                missing.extend(hardware.get('blocked_by') or ['physical execution readiness is not confirmed'])
        for required in spec.get('permissions',()):
            if permission.get(required) is not True:missing.append('enable permission '+required)
        return list(dict.fromkeys(missing))

    def shutdown(self):self.stop_event.set()
