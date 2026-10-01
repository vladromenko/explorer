"""Small trainable baselines, versioning and regression gates for Explorer.

These models are deliberately transparent and local.  They provide a working
learning cycle when larger ACT/VLA backends are unavailable; they never bypass
the existing executor, collision checks or outcome verifier.
"""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid
import numpy as np
from autonomy_contracts import PolicyVersion, record


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2))
    temporary.replace(path)


class CandidateScorer:
    """Online logistic contextual-bandit scorer with a fixed geometry baseline."""
    FEATURE_VERSION='context_action_v1'
    def __init__(self, root, dimensions=18):
        self.root=Path(root)/'data/models/grasp-scorer';self.root.mkdir(parents=True,exist_ok=True)
        self.dimensions=dimensions;self.path=self.root/'accepted.json'
        self.weights=np.zeros(dimensions);self.bias=0.;self.version='geometry-baseline'
        self.load()

    def load(self):
        try:
            item=json.loads(self.path.read_text())
            if item['feature_version']==self.FEATURE_VERSION and len(item['weights'])==self.dimensions:
                self.weights=np.asarray(item['weights'],dtype=float);self.bias=float(item['bias']);self.version=item['id']
        except (OSError,ValueError,KeyError,TypeError):pass

    def vector(self, context, action):
        raw=[context.get('object_width_m'),context.get('object_height_m'),context.get('distance_m'),
             context.get('visibility'),context.get('depth_uncertainty_m'),context.get('target_distance_m'),
             context.get('softness'),context.get('scene_clutter'),context.get('previous_failures'),
             action.get('approach_x'),action.get('approach_y'),action.get('approach_z'),
             action.get('aperture_m'),action.get('roll_rad'),action.get('base_shift_m'),
             action.get('clearance_m'),action.get('trajectory_cost'),1.]
        x=np.asarray(raw,dtype=float)
        if x.shape!=(self.dimensions,) or not np.isfinite(x).all():raise ValueError('Incomplete finite scorer features')
        return x

    @staticmethod
    def geometry_baseline(context, action):
        clearance=float(action.get('clearance_m',0));uncertainty=float(context.get('depth_uncertainty_m',1))
        reach=max(0.,1.-float(action.get('trajectory_cost',1))/10)
        return max(0.,min(1.,.45*reach+.35*min(1.,clearance/.03)+.2*max(0.,1.-uncertainty/.03)))

    def score(self, context, action):
        x=self.vector(context,action);logit=float(x@self.weights+self.bias)
        learned=1/(1+math.exp(-max(-30,min(30,logit))))
        return dict(score=learned if self.version!='geometry-baseline' else self.geometry_baseline(context,action),
                    learned_score=learned,baseline_score=self.geometry_baseline(context,action),
                    policy_version=self.version,feature_version=self.FEATURE_VERSION)

    def choose(self, context, candidates, exploration=0., seed=None):
        if not candidates:raise ValueError('At least one candidate is required')
        rows=[dict(candidate=item,prediction=self.score(context,item)) for item in candidates]
        rng=np.random.default_rng(seed)
        exploratory=exploration>0 and float(rng.random())<exploration
        selected=int(rng.integers(len(rows))) if exploratory else int(np.argmax([row['prediction']['score'] for row in rows]))
        return dict(selected=selected,rows=rows,method='epsilon_contextual_bandit' if exploratory else 'ranked_contextual_scorer',
                    exploration=float(exploration),policy_version=self.version)

    def train(self, samples, epochs=300, learning_rate=.04):
        usable=[row for row in samples if row.get('outcome') in ('success','failure')]
        scenes=sorted({str(row['episode_id']) for row in usable})
        if len(usable)<20 or len(scenes)<4:raise ValueError('Need 20 verified attempts from at least four episodes')
        validation={scene for scene in scenes if int(hashlib.sha256(scene.encode()).hexdigest(),16)%5==0}
        if not validation:validation={scenes[-1]}
        train=[row for row in usable if str(row['episode_id']) not in validation]
        valid=[row for row in usable if str(row['episode_id']) in validation]
        if not train or not valid:raise ValueError('Episode-level split could not be formed')
        x=np.asarray([self.vector(row['context'],row['action']) for row in train]);y=np.asarray([row['outcome']=='success' for row in train],float)
        vx=np.asarray([self.vector(row['context'],row['action']) for row in valid]);vy=np.asarray([row['outcome']=='success' for row in valid],float)
        weights=np.zeros(self.dimensions);bias=0.
        for _ in range(epochs):
            logits=np.clip(x@weights+bias,-30,30);probability=1/(1+np.exp(-logits));error=probability-y
            weights-=learning_rate*(x.T@error/len(x)+.002*weights);bias-=learning_rate*float(np.mean(error))
        probability=1/(1+np.exp(-np.clip(vx@weights+bias,-30,30)))
        predictions=probability>=.5;accuracy=float(np.mean(predictions==vy))
        baseline=np.asarray([self.geometry_baseline(row['context'],row['action']) for row in valid])>=.5
        baseline_accuracy=float(np.mean(baseline==vy))
        identifier=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        checkpoint=dict(id=identifier,backend='contextual_logistic_scorer',feature_version=self.FEATURE_VERSION,
                        weights=weights.tolist(),bias=bias,metrics={'validation_accuracy':accuracy,
                        'geometry_baseline_accuracy':baseline_accuracy,'validation_samples':len(valid)},
                        train_episodes=[scene for scene in scenes if scene not in validation],validation_episodes=sorted(validation))
        atomic_json(self.root/(identifier+'.json'),checkpoint)
        return checkpoint

    def promote(self, checkpoint, minimum_margin=0.):
        metrics=checkpoint['metrics']
        evaluation=self.evaluation_status(checkpoint['id'])
        accepted=(metrics['validation_accuracy']>=metrics['geometry_baseline_accuracy']+minimum_margin and
                  evaluation['ready'] and evaluation['candidate_success_rate']>=evaluation['baseline_success_rate'])
        if accepted:
            previous=json.loads(self.path.read_text()) if self.path.exists() else None
            if previous:atomic_json(self.root/'rollback.json',previous)
            atomic_json(self.path,checkpoint);self.load()
        return dict(accepted=accepted,policy_version=checkpoint['id'],metrics=metrics,
                    physical_evaluation=evaluation,
                    reason='offline_and_physical_gates_passed' if accepted else 'offline_or_physical_gate_rejected')

    def record_evaluation(self,checkpoint_id,arm,outcome,episode_id):
        if arm not in ('baseline','candidate') or outcome not in ('success','failure','unknown'):
            raise ValueError('Invalid physical evaluation result')
        path=self.root/'physical-evaluation.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(dict(at=time.time(),checkpoint_id=checkpoint_id,arm=arm,
                outcome=outcome,episode_id=episode_id),allow_nan=False)+'\n')
        return self.evaluation_status(checkpoint_id)

    def evaluation_status(self,checkpoint_id,minimum_per_arm=6):
        path=self.root/'physical-evaluation.jsonl';rows=[]
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    item=json.loads(line)
                    if item.get('checkpoint_id')==checkpoint_id and item.get('outcome') in ('success','failure'):
                        rows.append(item)
        baseline=[row for row in rows if row['arm']=='baseline'];candidate=[row for row in rows if row['arm']=='candidate']
        rate=lambda values:sum(row['outcome']=='success' for row in values)/len(values) if values else None
        return dict(ready=len(baseline)>=minimum_per_arm and len(candidate)>=minimum_per_arm,
            minimum_per_arm=minimum_per_arm,baseline_trials=len(baseline),candidate_trials=len(candidate),
            baseline_success_rate=rate(baseline),candidate_success_rate=rate(candidate),
            assignment='deterministic_alternation',unknown_excluded=True)

    def rollback(self):
        path=self.root/'rollback.json'
        if not path.exists():raise ValueError('No accepted rollback checkpoint')
        current=json.loads(self.path.read_text()) if self.path.exists() else None
        previous=json.loads(path.read_text());atomic_json(self.path,previous)
        if current:atomic_json(self.root/'rolled-back.json',current)
        self.load();return dict(active=self.version,rolled_back=True)


class OutcomeModel:
    """Ridge model of observed physical deltas; command-to-command pairs are rejected."""
    VERSION='observed_delta_ridge_v1'
    def __init__(self, root, state_dimensions=6, action_dimensions=6):
        self.root=Path(root)/'data/models/outcome';self.root.mkdir(parents=True,exist_ok=True)
        self.sd=state_dimensions;self.ad=action_dimensions;self.weights=None

    def train(self, transitions):
        valid=[row for row in transitions if row.get('observation_source') in ('rgbd','lidar_odom','verified_hold') and
               row.get('confidence',0)>=.8]
        if len(valid)<12:raise ValueError('Need 12 independently observed transitions')
        x=np.asarray([list(row['state_before'])+list(row['action'])+[1.] for row in valid],float)
        y=np.asarray([row['observed_delta'] for row in valid],float)
        if x.shape[1]!=self.sd+self.ad+1 or y.ndim!=2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid transition geometry')
        weights=np.linalg.solve(x.T@x+.01*np.eye(x.shape[1]),x.T@y);prediction=x@weights
        error=float(np.sqrt(np.mean((prediction-y)**2)))
        identifier=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        model=dict(id=identifier,version=self.VERSION,weights=weights.tolist(),rmse=error,samples=len(valid))
        atomic_json(self.root/(identifier+'.json'),model);atomic_json(self.root/'candidate.json',model)
        self.weights=weights;return model

    def predict(self, state, action):
        if self.weights is None:
            try:self.weights=np.asarray(json.loads((self.root/'candidate.json').read_text())['weights'])
            except (OSError,ValueError,KeyError):raise ValueError('Outcome model has no trained checkpoint')
        x=np.asarray([*state,*action,1.],float)
        if x.shape!=(self.sd+self.ad+1,) or not np.isfinite(x).all():raise ValueError('Invalid outcome-model input')
        return (x@self.weights).tolist()


class InterventionActorLearner:
    """Bounded linear actor/critic learner for short correction skills.

    The environment records proposed and actually executed actions separately.
    It is an actor/learner backend, not a claim of full HIL-SERL reproduction.
    """
    def __init__(self, root, observations=12, actions=3):
        self.root=Path(root)/'data/models/intervention-rl';self.root.mkdir(parents=True,exist_ok=True)
        self.od=observations;self.ad=actions;self.actor=np.zeros((observations,actions));self.critic=np.zeros(observations+actions+1)

    def update(self, transitions, epochs=80, rate=.01):
        rows=[row for row in transitions if row.get('reward') is not None and row.get('terminated') in (True,False) and row.get('truncated') in (True,False)]
        if len(rows)<32:raise ValueError('Need 32 labelled short-skill transitions')
        rng=np.random.default_rng(2026)
        for _ in range(epochs):
            row=rows[int(rng.integers(len(rows)))];obs=np.asarray(row['observation'],float);executed=np.asarray(row['executed_action'],float)
            proposed=np.asarray(row['proposed_action'],float);next_obs=np.asarray(row['next_observation'],float)
            if obs.shape!=(self.od,) or next_obs.shape!=(self.od,) or executed.shape!=(self.ad,) or proposed.shape!=(self.ad,):
                raise ValueError('Invalid actor/learner transition')
            qx=np.r_[obs,executed,1.];q=float(qx@self.critic)
            next_action=np.tanh(next_obs@self.actor);next_q=float(np.r_[next_obs,next_action,1.]@self.critic)
            target=float(row['reward'])+(0. if row['terminated'] else .97*next_q);self.critic+=rate*(target-q)*qx
            desired=executed if row.get('intervention') else proposed
            self.actor+=rate*np.outer(obs,(desired-np.tanh(obs@self.actor))*(1-np.tanh(obs@self.actor)**2))
        identifier=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        checkpoint=dict(id=identifier,backend='bounded_intervention_actor_critic',actor=self.actor.tolist(),
                        critic=self.critic.tolist(),samples=len(rows),observation_dimensions=self.od,action_dimensions=self.ad)
        atomic_json(self.root/(identifier+'.json'),checkpoint)
        return checkpoint


class PolicyRegistry:
    def __init__(self, root):
        self.root=Path(root)/'data/policies';self.root.mkdir(parents=True,exist_ok=True)
        self.index=self.root/'registry.json'

    def all(self):
        try:return json.loads(self.index.read_text()).get('policies',[])
        except (OSError,ValueError,TypeError):return []

    def add(self, version):
        version.validate();rows=self.all()
        if any(row['id']==version.id for row in rows):raise ValueError('Policy version already exists')
        rows.append(record(version));atomic_json(self.index,{'policies':rows})
        return record(version)

    def transition(self, identifier, state, metrics=None):
        rows=self.all();found=False
        for row in rows:
            if row['id']==identifier:
                found=True;row['state']=state
                if metrics is not None:row['metrics']=metrics
                PolicyVersion(**row).validate()
        if not found:raise ValueError('Policy version not found')
        atomic_json(self.index,{'policies':rows});return [row for row in rows if row['id']==identifier][0]

    def accepted(self, skill):
        return [row for row in self.all() if row['state']=='accepted' and row['metrics'].get('skill')==skill]


def backend_status(root):
    root=Path(root);learning=root/'.venv-learning'
    def available(module):
        path=learning/'lib'
        return learning.exists() and any((folder/'site-packages'/module).exists() for folder in path.glob('python*'))
    return dict(
      scorer=dict(implementation=True,weights=(root/'data/models/grasp-scorer/accepted.json').exists()),
      act=dict(implementation=(root/'bin/policy-worker.py').exists(),environment=learning.exists(),weights=any((root/'data/learning-jobs').glob('*/model/checkpoints/last/pretrained_model'))),
      smolvla=dict(implementation='adapter_pending_compatible_weights',environment=learning.exists(),weights=False,
                   blocked_by=['No compatible accepted Explorer SmolVLA checkpoint','8 GB resource validation not completed']),
      intervention_rl=dict(implementation=True,weights=any((root/'data/models/intervention-rl').glob('*.json')),
                           claim='bounded actor/critic baseline; not full HIL-SERL'))
