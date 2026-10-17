import json
from pathlib import Path

import numpy as np
import pytest

from simulator.skill_parameters import PARAMETERS,parameters
from simulator.skill_training import (SkillTraining,outcome_score,phase_credit,propose,update_search)


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
        assert all(p[k]==state['incumbent'][k] for k,spec in PARAMETERS.items() if spec[3]!=state['phase_group'])


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


def test_stagnation_is_local_to_the_stage_instead_of_stopping_a_new_skill(tmp_path):
    state=SkillTraining(tmp_path,'a1'*6).create(attempts=100)
    for _ in range(7):
        state['phase_group']='pick'
        rows=paired(state,[[3,3]]*4)
        for row in rows:row['failure_group']='pick'
        update_search(state,rows)
    assert state['stagnation']==7
    rows=paired(state,[[11,11]]*4)
    update_search(state,rows)
    assert state['phase_group']=='place' and state['stagnation']==0
    assert state['phase_stagnation']['pick']==8


def test_curriculum_solves_three_training_pairs_before_independent_qualification(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'b1'*6);trainer.create(attempts=24)
    seeds=[];qualified=[]
    def batch(state,specs,*args):
        seeds.append({s['scene_seed'] for s in specs})
        rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],
            status='complete',split='train',score=100.,success=True,failure_group='place') for s in specs]
        state['episodes']+=rows;state['completed_attempts']+=len(rows);return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    monkeypatch.setattr(trainer,'_qualify',lambda state,*args:qualified.append(state['curriculum_pairs_passed']))
    result=trainer.run()
    assert qualified==[3] and result['completed_attempts']==24
    assert len(set.union(*seeds))==6
    assert not set.union(*seeds).intersection(seed for group in result['validation_seeds'] for seed in group)


def test_unsolved_curriculum_reuses_same_pair_for_comparable_policy_updates(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'c1'*6);trainer.create(attempts=16)
    seen=[]
    def batch(state,specs,*args):
        seen.append({s['scene_seed'] for s in specs})
        rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],status='complete',
            split='train',score=3.,success=False,failure_group='pick') for s in specs]
        state['episodes']+=rows;state['completed_attempts']+=len(rows);return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    trainer.run()
    assert len(seen)==2 and seen[0]==seen[1]


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


def test_program_update_drains_comparison_before_releasing_owner(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'d2'*6);trainer.create(attempts=16)
    seen=[]
    def batch(state,specs,*args):
        seen.append(len(specs));(trainer.folder/'update.request').write_text('{}')
        rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],status='complete',
            split='train',score=3.,success=False,failure_group='pick') for s in specs]
        state['episodes']+=rows;state['completed_attempts']+=len(rows);return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    result=trainer.run()
    assert seen==[8] and result['completed_attempts']==8
    assert result['state']=='maintenance_paused' and not trainer.pause.exists()
    assert len(result['episodes'])==8 and result['batch']
    assert trainer.run()['state']=='maintenance_paused'


def test_user_pause_takes_precedence_over_program_update(tmp_path):
    trainer=SkillTraining(tmp_path,'d3'*6);state=trainer.create(attempts=8)
    (trainer.folder/'update.request').write_text('{}');trainer.pause.touch()
    assert not trainer._pause_for_update(state)
    assert state['state']=='queued'


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
        def __init__(self,**kwargs):assert kwargs['max_workers']>=1
        def __enter__(self):return self
        def __exit__(self,*args):assert trainer.pause.exists()
        def submit(self,*args):
            class Future:
                def done(self):return interrupted()
            return Future()
    monkeypatch.setattr(training,'EpisodePool',Pool)
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(total=24*2**30,available=12*2**30))
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    spec=trainer._spec(state,state['incumbent'],0,906160004)
    with pytest.raises(KeyboardInterrupt):trainer._batch(state,[spec],lambda:False,lambda value:None)
    assert trainer.pause.exists() and spec['owner_pid']>0


def test_low_memory_waits_without_spawning_and_remains_cancellable(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import simulator.skill_training as training
    trainer=SkillTraining(tmp_path,'2'*12);state=trainer.create(attempts=8)
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(total=24*2**30,available=3*2**30))
    monkeypatch.setattr(training,'EpisodePool',lambda **kwargs:pytest.fail('spawned without capacity'))
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


def test_resume_finishes_existing_paired_batch_near_budget_without_adding_attempts(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'4'*12);state=trainer.create(attempts=100)
    specs=[trainer._spec(state,state['incumbent'],i,906160004+j) for i in range(4) for j in range(2)]
    state.update(state='operator_paused',completed_attempts=93,generation=11,
        batch=dict(population=propose(state),specs=specs,results=[]))
    prior=[dict(episode_id='prior'+str(i),seed=i,status='complete',split='train') for i in range(88)]
    current=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],status='complete',split='train',score=3.,success=False,failure_group='pick') for s in specs[:5]]
    state['episodes']=prior+current;trainer.save(state);sizes=[]
    def batch(saved,pending,*args):
        sizes.append(len(pending));rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],status='complete',split='train',score=3.,success=False,failure_group='pick') for s in pending]
        saved['episodes']+=rows;saved['completed_attempts']+=len(rows);return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    result=trainer.run(resume=True)
    assert sizes==[3] and result['completed_attempts']==96 and result['options']['attempts']==100
    assert result['state']=='budget_paused' and not result.get('batch')


def test_phase_objective_improves_hard_pick_without_being_blocked_by_easy_placement(tmp_path):
    state=SkillTraining(tmp_path,'5'*12).create(attempts=8)
    rows=paired(state,[[3,20],[3,11],[3,3],[3,3]])
    for i,row in enumerate(rows):
        row['seed']=i%2
        row['phase_credit']={'pick':9. if row['seed'] else 4.,'place':0.,'view':1.}
        row['failure_group']='pick' if row['seed']==0 else 'place'
    rows[2]['phase_credit']['pick']=6.
    update_search(state,rows)
    assert state['last_selection']['candidate']==1
    assert state['last_selection']['objective']=='lexicographic_worst_phase_first'
    assert state['skill_archive']['pick'][0]['parameters']==state['incumbent']


def test_phase_objective_never_sacrifices_a_verified_pick_for_command_credit(tmp_path):
    state=SkillTraining(tmp_path,'6'*12).create(attempts=8)
    rows=paired(state,[[3,20],[20,3],[3,3],[3,3]])
    for i,row in enumerate(rows):
        row['seed']=i%2;row['phase_credit']={'pick':4. if i%2==0 else 9.}
    rows[2]['phase_credit']['pick']=9.;rows[3]['phase_credit']['pick']=8.
    update_search(state,rows)
    assert state['last_selection']['candidate']==0


def test_program_changes_search_strategy_and_continues_past_eight_failed_rounds(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'7'*12);trainer.create(attempts=96)
    def batch(state,specs,*args):
        rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],status='complete',
            split='train',score=3.,phase_credit={'pick':6.},success=False,failure_group='pick',
            reason='unconfirmed_hold') for s in specs]
        state['episodes']+=rows;state['completed_attempts']+=len(rows);return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    result=trainer.run()
    assert result['completed_attempts']==96 and result['generation']==12
    assert result['state']=='budget_paused' and not result['simulation_ready']
    assert {r['next_mode'] for r in result['recovery_history']}=={'geometry','contact','hold_view'}
    assert all(r['evidence_thresholds_changed'] is False for r in result['recovery_history'])
    assert len(list((trainer.folder/'checkpoints').glob('*.json')))==12


def test_diagnostic_phase_credit_is_not_a_qualification_or_phantom_hold():
    report=dict(success=False,execution=dict(held_verified=False,placed_verified=False,
        events=[dict(stage=s,result={'outcome':'success'}) for s in ('find','approach','reobserve','pregrasp','grasp','lift')]),
        independent_physics_outcome=dict(lifted=False,success=False))
    assert phase_credit(report)['pick']==6.
    report['execution']['held_verified']=True
    assert phase_credit(report)['pick']==6.
    report['independent_physics_outcome']['lifted']=True
    assert phase_credit(report)['pick']==9.
    assert outcome_score(report)<100.


def test_failed_holdout_is_quarantined_and_program_continues_on_new_training_scenes(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'8'*12);trainer.create(attempts=48)
    training_seeds=[];holdouts=[]
    def batch(state,specs,*args):
        training_seeds.append({s['scene_seed'] for s in specs})
        rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],status='complete',
            split='train',score=100.,phase_credit={'pick':9.,'place':5.},success=True,failure_group='place') for s in specs]
        state['episodes']+=rows;state['completed_attempts']+=len(rows);return rows
    def qualify(state,*args):
        holdouts.append({s for group in state['validation_seeds'] for s in group})
        if len(holdouts)==1:
            state.update(state='validation_failed',validation=[dict(results=[dict(seed=s,success=False,failure_group='place')
                for s in sorted(holdouts[0])])])
        else:state.update(state='simulation_qualified',simulation_ready=True)
    monkeypatch.setattr(trainer,'_batch',batch);monkeypatch.setattr(trainer,'_qualify',qualify)
    result=trainer.run()
    assert result['state']=='simulation_qualified' and result['completed_attempts']==48
    assert len(holdouts)==2 and holdouts[0].isdisjoint(holdouts[1])
    assert training_seeds[0].isdisjoint(training_seeds[3])
    assert set.union(*training_seeds).isdisjoint(set.union(*holdouts))
    assert result['validation_history'][0]['training_use_forbidden'] is True


def test_previously_examined_holdout_cannot_be_reused_to_claim_readiness(tmp_path):
    trainer=SkillTraining(tmp_path,'9'*12);state=trainer.create(attempts=8)
    state['validation_history']=[dict(seeds=state['validation_seeds'])]
    with pytest.raises(RuntimeError,match='Previously examined'):trainer._qualify(state,lambda:False,lambda s:None)


def test_automatic_quality_curriculum_requires_sixteen_different_training_pairs(tmp_path,monkeypatch):
    from simulator.scene_recipes import compile_recipes
    recipe=compile_recipes(dict(fully_supported=True,task_ids=['sock_basket']),[])['recipes'][0]
    trainer=SkillTraining(tmp_path,'de'*6);state=trainer.create(attempts=128,scene_recipe=recipe)
    pairs=[];qualified=[]
    def batch(saved,specs,*args):
        pairs.append({s['scene_seed'] for s in specs})
        rows=[dict(episode_id=s['episode_id'],candidate=s['candidate'],seed=s['scene_seed'],status='complete',
            split='train',score=100.,success=True,failure_group='place') for s in specs]
        saved['episodes']+=rows;saved['completed_attempts']+=len(rows);return rows
    monkeypatch.setattr(trainer,'_batch',batch)
    monkeypatch.setattr(trainer,'_qualify',lambda saved,*args:qualified.append(saved['curriculum_pairs_passed']))
    trainer.run()
    assert qualified==[16] and len(pairs)==16 and len(set.union(*pairs))==32
