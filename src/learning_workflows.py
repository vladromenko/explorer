"""One persistent learning workflow for demonstrations, autonomy and interventions."""
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from episode_store import EpisodeStore


MODES=('HUMAN_DEMONSTRATION','AUTONOMOUS','HUMAN_INTERVENTION','BOOTSTRAP_THEN_AUTONOMOUS')
TERMINAL=('complete','cancelled','failed')


class LearningWorkflows:
    def __init__(self,root,autonomy_submit=None,train_submit=None):
        self.root=Path(root);self.folder=self.root/'data/learning-workflows';self.folder.mkdir(parents=True,exist_ok=True)
        self.database=self.folder/'workflows.sqlite3';self.episodes=EpisodeStore(self.root)
        self.autonomy_submit=autonomy_submit;self.train_submit=train_submit;self.lock=threading.RLock()
        self._schema();self._interrupt_active()

    def db(self):
        connection=sqlite3.connect(self.database,timeout=10);connection.row_factory=sqlite3.Row
        return connection

    def _schema(self):
        with self.db() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS workflows(
                id TEXT PRIMARY KEY, mode TEXT, skill TEXT, target TEXT, state TEXT,
                human_target INTEGER, autonomous_target INTEGER, human_done INTEGER DEFAULT 0,
                autonomous_done INTEGER DEFAULT 0, interventions INTEGER DEFAULT 0,
                success INTEGER DEFAULT 0, failure INTEGER DEFAULT 0, unknown_count INTEGER DEFAULT 0,
                current_policy TEXT, training_state TEXT, active_episode TEXT,
                created REAL, updated REAL, detail TEXT);
              CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,workflow_id TEXT,at REAL,kind TEXT,payload TEXT);
            ''')

    def _interrupt_active(self):
        with self.db() as db:
            rows=db.execute("SELECT id FROM workflows WHERE state NOT IN ('complete','cancelled','failed','waiting_demo','needs_reset')").fetchall()
            for row in rows:
                db.execute("UPDATE workflows SET state='interrupted',updated=?,detail=? WHERE id=?",
                           (time.time(),json.dumps({'reason':'Process restarted; no action replayed'}),row['id']))

    def start(self,mode,skill,target,human_demonstrations=0,autonomous_trials=0):
        if mode not in MODES:raise ValueError('Unknown learning mode')
        if skill not in ('grasp','place','push','mobile_pick_place'):raise ValueError('Unsupported learning skill')
        if not str(target).strip():raise ValueError('Target is required')
        human=int(human_demonstrations);autonomous=int(autonomous_trials)
        if not 0<=human<=500 or not 0<=autonomous<=500 or human+autonomous<1:
            raise ValueError('At least one bounded demonstration or autonomous trial is required')
        if mode=='HUMAN_DEMONSTRATION':autonomous=0
        if mode=='AUTONOMOUS':human=0
        identifier=uuid.uuid4().hex;now=time.time()
        state='waiting_demo' if human else 'queued_autonomous'
        with self.db() as db:
            db.execute('INSERT INTO workflows VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (identifier,mode,skill,str(target).strip()[:80],state,human,autonomous,0,0,0,0,0,0,
                 'geometry-baseline','idle','',now,now,'{}'))
            self.event(db,identifier,'created',{'mode':mode,'human_target':human,'autonomous_target':autonomous})
        return self.get(identifier)

    def attach_demonstration(self,identifier,legacy_manifest):
        with self.lock:
            workflow=self.get(identifier)
            if workflow['state'] not in ('waiting_demo','interrupted'):
                raise ValueError('Workflow is not waiting for a demonstration')
            imported=self.episodes.import_legacy(legacy_manifest,'HUMAN_DEMONSTRATION',
                task=workflow['skill']+' '+workflow['target'],skill=workflow['skill'])
            outcome=imported['outcome'];human=workflow['human_done']+1
            next_state='waiting_demo'
            if human>=workflow['human_target']:
                next_state='training' if workflow['autonomous_target'] else 'complete'
            with self.db() as db:
                db.execute('UPDATE workflows SET human_done=?,state=?,updated=? WHERE id=?',
                           (human,next_state,time.time(),identifier))
                self._increment(db,identifier,outcome)
                self.event(db,identifier,'demonstration_ingested',{'episode_id':imported['id'],'outcome':outcome})
            if next_state=='training':self._train_then_autonomy(identifier)
            return self.get(identifier)

    def begin_autonomous_episode(self,identifier,scene_version,policy_version='geometry-baseline',reward_version='grasp_outcome_v1'):
        with self.lock:
            workflow=self.get(identifier)
            if workflow['state'] not in ('queued_autonomous','autonomous','interrupted'):
                raise ValueError('Workflow is not ready for an autonomous trial')
            episode=self.episodes.begin(identifier,workflow['skill']+' '+workflow['target'],scene_version,
                policy_version,reward_version,source='AUTONOMOUS',task=workflow['skill']+' '+workflow['target'],
                skill=workflow['skill'],target=workflow['target'])
            with self.db() as db:
                db.execute("UPDATE workflows SET state='autonomous',active_episode=?,updated=? WHERE id=?",
                           (episode['id'],time.time(),identifier))
                self.event(db,identifier,'autonomous_episode_started',{'episode_id':episode['id']})
            return episode

    def finish_autonomous_episode(self,identifier,outcome,evidence,failure_reason='',reward=None):
        with self.lock:
            workflow=self.get(identifier);episode=workflow['active_episode']
            if not episode:raise ValueError('No active autonomous episode')
            if workflow['state']=='reobserve_and_replan':
                raise ValueError('Fresh observation and replanning are required after intervention')
            self.episodes.finish(episode,outcome,evidence,failure_reason,reward)
            done=workflow['autonomous_done']+1
            state='complete' if done>=workflow['autonomous_target'] else 'queued_autonomous'
            with self.db() as db:
                db.execute("UPDATE workflows SET autonomous_done=?,state=?,active_episode='',updated=? WHERE id=?",
                           (done,state,time.time(),identifier));self._increment(db,identifier,outcome)
                self.event(db,identifier,'autonomous_episode_finished',{'episode_id':episode,'outcome':outcome,
                    'failure_reason':failure_reason,'reward':reward})
            return self.get(identifier)

    def intervention(self,identifier,proposed_action,executed_action,started_at,ended_at,metadata=None):
        with self.lock:
            workflow=self.get(identifier);episode=workflow['active_episode']
            if not episode:raise ValueError('No active autonomous episode')
            item=self.episodes.intervention(episode,started_at,ended_at,proposed_action,executed_action,metadata)
            with self.db() as db:
                db.execute('UPDATE workflows SET interventions=interventions+1,state=?,updated=? WHERE id=?',
                           ('reobserve_and_replan',time.time(),identifier))
                self.event(db,identifier,'human_intervention',{'episode_id':episode,
                    'old_action_invalidated':True,'fresh_observation_required':True})
            return dict(workflow=self.get(identifier),intervention=item)

    def mark_replanned(self,identifier,observation):
        workflow=self.get(identifier)
        if workflow['state']!='reobserve_and_replan':raise ValueError('Workflow does not require replanning')
        self.episodes.append(workflow['active_episode'],'observation',observation)
        with self.db() as db:
            db.execute("UPDATE workflows SET state='autonomous',updated=? WHERE id=?",(time.time(),identifier))
            self.event(db,identifier,'fresh_observation_and_replan',{'scene_version':observation.get('scene_version')})
        return self.get(identifier)

    def needs_reset(self,identifier,instruction='Положите предмет обратно в видимую тренировочную область'):
        with self.db() as db:
            db.execute('UPDATE workflows SET state=?,updated=?,detail=? WHERE id=?',
                       ('needs_reset',time.time(),json.dumps({'instruction':instruction},ensure_ascii=False),identifier))
            self.event(db,identifier,'needs_reset',{'instruction':instruction})
        return self.get(identifier)

    def reset_observed(self,identifier,evidence):
        workflow=self.get(identifier)
        if workflow['state']!='needs_reset':raise ValueError('Workflow is not waiting for reset')
        if evidence.get('target_visible') is not True or evidence.get('fresh') is not True:
            raise ValueError('Reset is not visually established')
        with self.db() as db:
            db.execute("UPDATE workflows SET state='queued_autonomous',updated=?,detail='{}' WHERE id=?",
                       (time.time(),identifier));self.event(db,identifier,'reset_observed',evidence)
        return self.get(identifier)

    def cancel(self,identifier):
        with self.db() as db:
            db.execute("UPDATE workflows SET state='cancelled',updated=?,active_episode='' WHERE id=?",
                       (time.time(),identifier));self.event(db,identifier,'cancelled',{})
        return self.get(identifier)

    def _train_then_autonomy(self,identifier):
        workflow=self.get(identifier)
        result={'queued':False,'reason':'training backend callback is not bound'}
        if self.train_submit is not None:result=self.train_submit(workflow)
        state='queued_autonomous' if result.get('queued') is True else 'training_blocked'
        with self.db() as db:
            db.execute('UPDATE workflows SET state=?,training_state=?,updated=?,detail=? WHERE id=?',
                       (state,'queued' if result.get('queued') else 'blocked',time.time(),json.dumps(result),identifier))
            self.event(db,identifier,'bootstrap_training',result)

    @staticmethod
    def _increment(db,identifier,outcome):
        column={'success':'success','failure':'failure'}.get(outcome,'unknown_count')
        db.execute('UPDATE workflows SET '+column+'='+column+'+1 WHERE id=?',(identifier,))

    @staticmethod
    def event(db,identifier,kind,payload):
        db.execute('INSERT INTO events(workflow_id,at,kind,payload) VALUES(?,?,?,?)',
                   (identifier,time.time(),kind,json.dumps(payload,ensure_ascii=False,allow_nan=False)))

    def get(self,identifier):
        with self.db() as db:
            row=db.execute('SELECT * FROM workflows WHERE id=?',(identifier,)).fetchone()
            if not row:raise ValueError('Learning workflow not found')
            result=dict(row);result['detail']=json.loads(result['detail'] or '{}')
            result['events']=[dict(item,payload=json.loads(item['payload'])) for item in
                db.execute('SELECT * FROM events WHERE workflow_id=? ORDER BY id',(identifier,))]
        return result

    def status(self):
        with self.db() as db:rows=[dict(row) for row in db.execute('SELECT * FROM workflows ORDER BY created DESC LIMIT 30')]
        for row in rows:row['detail']=json.loads(row['detail'] or '{}')
        return dict(at=time.time(),modes=list(MODES),workflows=rows,
            active=[row for row in rows if row['state'] not in TERMINAL],episode_storage=self.episodes.usage(),
            common_episode_format='explorer_typed_episode_v1',motor_path='existing manual/autonomy executors',
            replay_after_restart=False)
