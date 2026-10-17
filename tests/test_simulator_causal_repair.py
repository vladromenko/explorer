import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simulator.failure_repair import (program,diagnose_report,diagnosis,repair_proposals,record_trial)
from simulator.repair_acceptance import confirms_nonregression
from simulator.skill_training import SkillTraining,update_search


@pytest.fixture(autouse=True)
def deterministic_resources(monkeypatch):
    monkeypatch.setattr('simulator.resource_budget._mac_available',lambda total:None)


@pytest.mark.parametrize('bad',[{'schema':1,'operators':['change_friction']},
    {'schema':1,'operators':[],'code':'import os'}, {'schema':True,'operators':[]},
    {'schema':1,'operators':['extended_hold_probe','extended_hold_probe']},
    {'schema':1,'operators':['reduce_acceptance_threshold']}])
def test_program_cannot_change_physics_acceptance_or_execute_model_generated_code(bad):
    with pytest.raises(ValueError):program(bad)


def report(stage,reason,lifted=False,contact=False,success=False):
    return dict(success=False,phase=stage,reason=reason,execution=dict(stage=stage),
        independent_physics_outcome=dict(lifted=lifted,grasp_contact=contact,success=success))


def test_diagnosis_distinguishes_unobservable_hold_from_physical_miss_and_uncertainty():
    assert diagnose_report(report('verify_hold','Камера не подтверждает',True,True))==('held_but_unseen','identified')
    assert diagnose_report(report('verify_hold','Камера не подтверждает',False,False))==('missed_material','identified')
    assert diagnose_report(report('carry','Камера не подтверждает',True,False))==('hold_lost_or_occluded','hypothesis')
    assert diagnose_report(report('verify_place','нет двух кадров',True,False,True))==('placement_unseen','identified')
    assert diagnose_report(report('support','Проверочное движение недостижимо'))==('probe_infeasible','identified')
    assert diagnose_report(report('pregrasp','Нет точного захвата'))==('ik_infeasible','identified')
    assert diagnose_report(report('find','unrecognised error'))==('unclassified','unknown')


def test_diagnosis_records_exact_post_stop_evidence_and_rejects_outside_storage(tmp_path):
    p=tmp_path/'report.json';p.write_text(json.dumps(report('verify_hold','Камера не подтверждает',False,False)))
    rows=[dict(status='complete',success=False,report=str(p),episode_id='one'),
          dict(status='running',success=False,report=str(p),episode_id='active'),
          dict(status='complete',success=False,report='/etc/passwd',episode_id='outside')]
    result=diagnosis(rows,tmp_path)
    assert len(result['cases'])==1 and result['cases'][0]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert result['causes'][0]['code']=='missed_material'
    assert not result['acceptance_changed']


@pytest.mark.parametrize('reason,code',[
    ('support: Нет пути изменения обзора: Обзорная поза недостижима: ошибка наведения 2.1 см','view_infeasible'),
    ('ValueError: Нет достижимой позы размещения с проверенным зазором корпуса руки над бортиком','placement_infeasible'),
])
def test_explicit_path_failure_is_not_misdiagnosed_as_lost_visual_hold(reason,code):
    # Real failure strings from the background run: earlier physical holding
    # must not turn a later kinematic failure into a perception diagnosis.
    for lifted,contact in ((True,True),(True,False),(False,False)):
        assert diagnose_report(report('support',reason,lifted,contact))==(code,'identified')
    state=dict(controller_program=program(),experience={'ik_seeds':[[0]*5]})
    from simulator.failure_repair import CAUSES
    choices=repair_proposals(state,{'causes':[{'operators':list(CAUSES[code][1])}]})
    assert choices[1]['operators']==['history_ik_seeds']


def test_cause_selects_a_structural_repair_and_does_not_repeat_an_exhausted_test():
    state={'generation':0,'controller_program':program()}
    analysis={'causes':[{'operators':['extended_hold_probe','probe_feasible_view']}]}
    choices=repair_proposals(state,analysis)
    assert choices[0]==program() and choices[1]['operators']==['extended_hold_probe']
    # Coupled repairs remain available even when neither single change helped.
    assert choices[3]['operators']==['extended_hold_probe','probe_feasible_view']
    record_trial(state,choices,0,False)
    assert repair_proposals(state,analysis) is None
    state['curriculum_pairs_passed']=1
    assert repair_proposals(state,analysis) is not None


def confirmation():
    return [dict(candidate=c,seed=s,status='complete',success=False,score=11.,phase_credit={'pick':9.,'place':0.})
            for c in (0,1) for s in (101,102)]


def test_new_repair_is_explored_before_repeating_old_ideas_without_acceptance_credit():
    state=dict(controller_program=program(),historical_repair_trials=[
        dict(program=program({'schema':1,'operators':['extended_hold_probe']}))])
    analysis={'causes':[{'operators':['extended_hold_probe','kinematic_hold_probe']}]}
    proposals=repair_proposals(state,analysis)
    assert proposals[1]['operators']==['kinematic_hold_probe']
    assert any(p['operators']==['extended_hold_probe'] for p in proposals)
    assert not state.get('repair_trials')


def test_version_update_keeps_old_best_program_only_as_unverified_hint(tmp_path):
    from simulator.skill_training import SkillTraining
    from simulator.scene_recipes import compile_recipes
    recipe=compile_recipes(dict(fully_supported=True,task_ids=['sock_basket']))['recipes'][0]
    old=SkillTraining(tmp_path,'ac'*6);state=old.create(attempts=8,scene_recipe=recipe)
    state['generation']=3;state['incumbent']['pick_fraction']=.25
    state['controller_program']=program({'schema':1,'operators':['extended_hold_probe']})
    state['repair_trials']=[dict(program=state['controller_program'])];old.save(state)
    new=SkillTraining(tmp_path,'ad'*6).create(attempts=8,scene_recipe=recipe)
    assert new['incumbent']['pick_fraction']==.25 and new['controller_program']==state['controller_program']
    assert not new['episodes'] and not new['repair_trials'] and not new['simulation_ready']
    assert new['initialization']['old_episodes_counted_for_qualification'] is False
    assert len(new['historical_repair_trials'])==1


def test_confirmation_rejects_pick_regression_malformed_evidence_and_unpaired_scenes():
    rows=confirmation();assert confirms_nonregression(rows,'pick')[0]
    rows[2]['phase_credit']['pick']=8.;assert confirms_nonregression(rows,'pick')==(False,'regressed_verified_pick')
    rows=confirmation();rows[2]['score']=float('nan');assert not confirms_nonregression(rows,'pick')[0]
    rows=confirmation();rows[2]['seed']=999;assert not confirms_nonregression(rows,'pick')[0]
    rows=confirmation();rows[0]['success']=True;assert confirms_nonregression(rows,'pick')==(False,'regressed_full_task')


def setup_causal_batch(trainer):
    state=trainer.create(attempts=16)
    choices=[program(),program({'schema':1,'operators':['extended_hold_probe']}),program(),program()]
    population=[copy.deepcopy(state['incumbent']) for _ in range(4)]
    state['batch']=dict(population=population,repair_programs=choices)
    rows=[dict(candidate=c,seed=s,score=3. if c!=1 else 11.,success=False,
        phase_credit={'pick':6. if c!=1 else 9.},failure_group='pick',episode_id=f'{c}-{s}')
        for c in range(4) for s in (1,2)]
    return state,rows


def test_causal_attribution_requires_same_numeric_settings_and_scene_pairs(tmp_path):
    trainer=SkillTraining(tmp_path,'a'*12);state,rows=setup_causal_batch(trainer)
    state['batch']['population'][1]['pick_fraction']=.3
    with pytest.raises(RuntimeError,match='identical numeric'):update_search(state,rows)
    state,rows=setup_causal_batch(trainer);rows[2]['seed']=20
    with pytest.raises(RuntimeError,match='paired scene'):update_search(state,rows)


def test_causal_promotion_cannot_bypass_fresh_scene_confirmation(tmp_path):
    trainer=SkillTraining(tmp_path,'ea'*6);state,rows=setup_causal_batch(trainer)
    with pytest.raises(RuntimeError,match='new-scene confirmation'):update_search(state,rows)
    assert state['controller_program']==program()


def test_initial_attempt_limit_allocates_untried_repairs_but_never_clears_operator_pause(tmp_path):
    trainer=SkillTraining(tmp_path,'eb'*6);state=trainer.create(attempts=8)
    p=tmp_path/'report.json';p.write_text(json.dumps(report('support','Проверочное движение недостижимо')))
    state.update(state='running',completed_attempts=8,episodes=[dict(status='complete',success=False,report=str(p))])
    assert trainer._extend_for_repair(state)
    assert state['options']['attempts']==16 and state['force_repair'] is True
    assert state['recovery_history'][-1]['wall_time_extended'] is False
    state['pause_kind']='operator'
    assert not trainer._extend_for_repair(state) and state['options']['attempts']==16


def test_new_scene_confirmation_can_reject_a_lucky_repair_without_rewriting_results(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'b'*12);state,rows=setup_causal_batch(trainer)
    def batch(state,specs,*args):
        output=[]
        for spec in specs:
            output.append(dict(episode_id=spec['episode_id'],seed=spec['scene_seed'],candidate=spec['candidate'],
                status='complete',split='train',success=False,score=11. if spec['candidate']==0 else 3.,
                phase_credit={'pick':9. if spec['candidate']==0 else 6.}))
        state['episodes']+=output;state['completed_attempts']+=4;return output
    monkeypatch.setattr(trainer,'_batch',batch)
    assert not trainer._confirm_program(state,1,lambda:False,lambda s:None)
    assert state['controller_program']==program()
    state['batch']['rejected_candidates']=[1]
    update_search(state,rows)
    assert state['last_selection']['candidate']==0 and state['controller_program']==program()
    assert rows[2]['score']==11. and state['repair_trials'][0]['promoted'] is False


def test_successful_causal_change_is_saved_in_policy_and_replicated_on_disjoint_training_scenes(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'c'*12);state,rows=setup_causal_batch(trainer)
    def batch(state,specs,*args):
        output=[dict(episode_id=s['episode_id'],seed=s['scene_seed'],candidate=s['candidate'],
            status='complete',split='train',success=False,score=11.,phase_credit={'pick':9.}) for s in specs]
        state['episodes']+=output;state['completed_attempts']+=4;return output
    monkeypatch.setattr(trainer,'_batch',batch)
    assert trainer._confirm_program(state,1,lambda:False,lambda s:None)
    update_search(state,rows)
    assert state['controller_program']['operators']==['extended_hold_probe']
    assert len({r['seed'] for r in state['episodes']})==2
    assert {r['seed'] for r in state['episodes']}.isdisjoint({1,2})
    assert not state['simulation_ready']


def test_extended_probe_adds_a_real_fallback_without_relaxing_rotation_or_floor_gates(monkeypatch):
    from types import SimpleNamespace
    from simulator.manipulation_planning import plan_hold_probe
    env=SimpleNamespace(jaw_center=lambda:np.array([.18,0.,.12]),base_pose=lambda:[0.,0.,0.],
        data=SimpleNamespace(body=lambda name:SimpleNamespace(xmat=np.eye(3).ravel())))
    seen=[]
    def solve(env,target,rotation,**kwargs):
        seen.append((target.copy(),kwargs['check_path']))
        return np.r_[np.full(5,90.),170.] if target[1]>.01 else None
    monkeypatch.setattr('simulator.manipulation_planning._plan_oriented_translation',solve)
    with pytest.raises(ValueError):plan_hold_probe(env,lowest_material_height_m=.02)
    result=plan_hold_probe(env,lowest_material_height_m=.02,extended=True)
    assert result.direction=='side_left'
    assert all(path and target[2]>=.12 for target,path in seen)


def test_visible_anchor_prediction_rejects_behind_camera_and_outside_frame_without_scene_truth():
    from types import SimpleNamespace
    from simulator.repair_motion import predicted_anchor_visible
    jaw=np.array([0.,0.,.1])
    env=SimpleNamespace(jaw_center=lambda data:jaw.copy(),model=SimpleNamespace(camera=lambda name:SimpleNamespace(id=0)))
    data=SimpleNamespace(body=lambda name:SimpleNamespace(xmat=np.eye(3).ravel()),
        cam_xpos=np.array([[0.,0.,.3]]),cam_xmat=np.array([np.eye(3).ravel()]))
    assert predicted_anchor_visible(env,data,[0.,0.,0.],raycast=False)
    assert not predicted_anchor_visible(env,data,[1.,0.,0.],raycast=False)
    jaw[2]=.4
    assert not predicted_anchor_visible(env,data,[0.,0.,0.],raycast=False)


def test_new_repairs_cannot_change_a_running_old_training_or_claim_its_qualification(tmp_path):
    trainer=SkillTraining(tmp_path,'d'*12);saved=trainer.create(attempts=8)
    saved.update(state='running',controller={'sha256':'old-frozen-version','files':{}});trainer.save(saved)
    before=trainer.path.read_bytes()
    choices=repair_proposals(saved,{'causes':[{'operators':['extended_hold_probe']}]})
    assert choices and trainer.path.read_bytes()==before
    assert saved['simulation_ready'] is False


def test_visible_anchor_robot_only_ray_uses_real_api_without_changing_physics():
    from types import SimpleNamespace
    from simulator.environment import ExplorerEnv
    from simulator.repair_motion import predicted_anchor_visible
    env=ExplorerEnv(render=False)
    try:
        before=env.data.qpos.copy();camera=env.model.camera('wrist_depth').id
        anchor=env.data.cam_xpos[camera]-.10*env.data.cam_xmat[camera].reshape(3,3)[:,2]
        robot=SimpleNamespace(model=env.model,jaw_center=lambda data:anchor)
        assert type(predicted_anchor_visible(robot,env.data,[0.,0.,0.])) is bool
        np.testing.assert_array_equal(env.data.qpos,before)
    finally:env.close()


@pytest.mark.parametrize('tracking_bias',[-.005,.005])
def test_cartesian_feedback_corrects_measured_sag_but_never_pushes_below_floor_target(tmp_path,monkeypatch,tracking_bias):
    from types import SimpleNamespace
    from simulator.skill_port import SimulatorSkillPort
    jaw=np.array([.21,0.,.02])
    env=SimpleNamespace(command=np.r_[np.full(6,90.),0.,0.,0.],pending_command=np.zeros(9),
        jaw_center=lambda:jaw.copy(),data=SimpleNamespace(ctrl=np.zeros(4)),wheel_ids=[0,1,2,3],last_error=None)
    port=SimulatorSkillPort(env,tmp_path/'port',repair_program={'schema':1,'operators':['cartesian_feedback']})
    aims=[]
    def solve(env,point,gripper,**kwargs):aims.append(np.asarray(point).copy());return np.r_[point,90.,90.,gripper]
    def move(command,ctx):jaw[:]=command[:3]+[0.,0.,tracking_bias]
    port.move_angles=move
    monkeypatch.setattr('simulator.grasp_planning.solve_pick_ik',solve)
    target=np.array([.21,0.,.017])
    try:
        if tracking_bias<0:
            port.move_pick_feedback(target,30.,None)
            assert np.linalg.norm(jaw-target)<=.003 and len(aims)==2
        else:
            with pytest.raises(ValueError,match='3 мм'):port.move_pick_feedback(target,30.,None)
        assert all(a[2]>=target[2] for a in aims)
    finally:port.close()
