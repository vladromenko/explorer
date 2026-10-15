import json
from pathlib import Path

import numpy as np
import pytest

from simulator.skill_parameters import PARAMETERS,parameters
from simulator.skill_training import (SkillTraining,outcome_score,propose,update_search)


@pytest.fixture(autouse=True)
def deterministic_resource_probe(monkeypatch):
    monkeypatch.setattr('simulator.resource_budget._mac_available',lambda total:None)


def test_learnable_parameters_cannot_change_evidence_physics_or_joint_limits():
    values=parameters()
    assert values['lift_clearance_m']>=.12
    for bad in ({'friction':0.},{'hold_tolerance':1.},{'pick_fraction':np.nan},{'view_baseline_m':.001}):
        with pytest.raises(ValueError):parameters(bad)
    assert all(group in ('pick','view','place') for _,_,_,group in PARAMETERS.values())


def test_command_acks_and_physics_only_placement_are_not_success_rewards():
    report=dict(success=False,execution=dict(held_verified=False,placed_verified=False,
        events=[dict(stage=s,result={'outcome':'success'}) for s in ('find','approach','reobserve','lift','carry','release')]),
        independent_physics_outcome=dict(lifted=True,success=True))
    assert outcome_score(report)==3.
    report['execution']['held_verified']=True
    assert outcome_score(report)==20.
    report['success']=True
    assert outcome_score(report)==20.
    report['execution']['placed_verified']=True
    assert outcome_score(report)==100.


def test_proposals_change_only_empirically_blocked_skill_and_are_reproducible(tmp_path):
    trainer=SkillTraining(tmp_path,'a'*12);state=trainer.create(attempts=8)
    population=propose(state)
    assert population==propose(state) and population[0]==parameters()
    assert len({json.dumps(p,sort_keys=True) for p in population})==4
    for p in population:
        assert parameters(p)==p
        assert all(p[k]==state['incumbent'][k] for k,spec in PARAMETERS.items() if spec[3]!='place')


def paired(state,scores):
    state['batch']={'population':propose(state)}
    return [dict(candidate=i,score=score,success=score==100.,failure_group='place')
            for i,row in enumerate(scores) for score in row]


def test_search_cannot_improve_average_by_regressing_worst_paired_scene(tmp_path):
    state=SkillTraining(tmp_path,'b'*12).create(attempts=8)
    reports=paired(state,[[10,10],[3,100],[11,11],[10,12]])
    assert not update_search(state,reports)
    assert state['last_selection']['candidate']==2
    assert state['incumbent']==state['batch']['population'][2]


def test_budget_is_finite_and_saves_policy_without_claiming_readiness(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'c'*12);trainer.create(attempts=8)
    def batch(state,specs,*args):
        rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],
            status='complete',split='train',score=10.+(s['candidate']==1),success=False,failure_group='place') for s in specs]
        state['episodes']+=rows;state['completed_attempts']+=len(rows);return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    result=trainer.run()
    assert result['state']=='budget_paused' and result['completed_attempts']==8
    assert not result['simulation_ready'] and not result['physical_ready']
    policy=json.loads((trainer.folder/'policy.json').read_text())
    assert policy['parameters']==result['incumbent'] and not policy['simulation_ready']


def test_operator_pause_is_not_resumed_without_explicit_request(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'d'*12);trainer.create(attempts=8)
    assert trainer.run(cancelled=lambda:True)['state']=='operator_paused'
    monkeypatch.setattr(trainer,'_batch',lambda *args:pytest.fail('operator pause was ignored'))
    assert trainer.run()['state']=='operator_paused'


def test_qualification_uses_two_disjoint_ten_scene_groups_never_for_search(tmp_path,monkeypatch):
    from simulator.domain_randomization import sample_sock_scene
    trainer=SkillTraining(tmp_path,'e'*12);state=trainer.create(attempts=8)
    (trainer.folder/'policy.json').write_text(json.dumps(dict(parameters=state['incumbent'],simulation_ready=False)))
    state['episodes']=[dict(episode_id='training-fixture',seed=906160004,status='complete',split='train')]
    seen=[]
    def batch(state,specs,*args):
        rows=[]
        for s in specs:
            assert s['split']=='validation' and s['scene_seed']!=906160004
            seen.append(s['scene_seed'])
            rows.append(dict(episode_id=s['episode_id'],seed=s['scene_seed'],status='complete',split='validation',
                success=True,domain_variation=sample_sock_scene(s['scene_seed'])))
        state['episodes']+=rows;return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    trainer._qualify(state,lambda:False,lambda value:None)
    assert len(seen)==len(set(seen))==20 and len(state['validation'])==2
    assert state['simulation_ready'] and not state['physical_ready']
    assert not state['automatic_hardware_execution'] and state['policy_sha256']
    assert state['archive']['verified'] and Path(state['archive']['path']).is_file()
    assert not state['validation_pending']


def test_validation_resume_reuses_completed_groups_and_keeps_split(tmp_path,monkeypatch):
    from simulator.domain_randomization import sample_sock_scene
    trainer=SkillTraining(tmp_path,'f'*12);state=trainer.create(attempts=8)
    (trainer.folder/'policy.json').write_text(json.dumps(dict(parameters=state['incumbent'],simulation_ready=False)))
    calls=[]
    def batch(state,specs,*args):
        calls.append([s['scene_seed'] for s in specs])
        if len(calls)==2:raise InterruptedError('pause')
        rows=[dict(episode_id=s['episode_id'],seed=s['scene_seed'],status='complete',split='validation',
            success=True,domain_variation=sample_sock_scene(s['scene_seed'])) for s in specs]
        state['episodes']+=rows;return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    with pytest.raises(InterruptedError):trainer._qualify(state,lambda:False,lambda state:None)
    assert state['validation_pending'] and len(state['validation'])==1
    trainer._qualify(state,lambda:False,lambda state:None)
    assert calls[1]==calls[2] and set(calls[0]).isdisjoint(calls[2])
    assert len(state['episodes'])==20 and all(r['split']=='validation' for r in state['episodes'])


def test_parent_interrupt_sets_cooperative_flag_before_waiting_for_pool(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import simulator.skill_training as training
    trainer=SkillTraining(tmp_path,'1'*12);state=trainer.create(attempts=8)
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):assert trainer.pause.exists()
        def submit(self,*args):return object()
    monkeypatch.setattr(training,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(total=24*2**30,available=12*2**30))
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr(training,'wait',interrupted)
    spec=trainer._spec(state,state['incumbent'],0,906160004)
    with pytest.raises(KeyboardInterrupt):trainer._batch(state,[spec],lambda:False,lambda value:None)
    assert trainer.pause.exists() and spec['owner_pid']>0


def test_low_memory_waits_without_spawning_and_remains_cancellable(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import simulator.skill_training as training
    trainer=SkillTraining(tmp_path,'2'*12);state=trainer.create(attempts=8)
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(total=24*2**30,available=3*2**30))
    monkeypatch.setattr(training,'ProcessPoolExecutor',lambda **kwargs:pytest.fail('spawned without capacity'))
    seen=[]
    with pytest.raises(InterruptedError):trainer._batch(state,[],lambda:True,lambda row:seen.append(row['state']))
    assert seen==['waiting_for_memory'] and state['resource_capacity']==0
    assert not state['active_workers']


def test_service_restart_preserves_results_and_exposes_explicit_resume(tmp_path):
    from simulator.skill_training import recover_trainings
    trainer=SkillTraining(tmp_path,'3'*12);state=trainer.create(attempts=8)
    state.update(state='waiting_for_memory',episodes=[{'episode_id':'saved'}]);trainer.save(state)
    recover_trainings(tmp_path)
    recovered=json.loads(trainer.path.read_text())
    assert recovered['state']=='interrupted' and recovered['episodes']==state['episodes']
    assert trainer.pause.exists()
    assert trainer.run()['state']=='interrupted'
