"""Fault recovery checks; these do not qualify the learned policy."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from simulator.executor_recovery import comparison,propose
from simulator.learning import write_json
from simulator.room_learning import RoomLearning
from simulator.room_policy import mission_limits
from simulator.scan_navigation import ScanMap
from src.room_cleanup import RoomCleanup,ViewUnavailable
from test_room_cleanup import Port,run


def test_partly_completed_blocked_route_reacquires_before_another_approach():
    p=Port();original=p.navigate
    def navigate(*a,**k):
        result=original(*a,**k)
        if len(p.moves)==1:raise ViewUnavailable('Fresh scan no longer admits that route')
        return result
    p.navigate=navigate;result=run(p)
    assert result['state']=='succeeded' and len(p.moves)==3
    assert p.moves[0][0]!=p.moves[1][0]
    assert min(p.observations[2:])>=2  # failed route also invalidated pre-drive evidence
    assert any(e['stage']=='approach_route_unavailable' for e in result['events'])


def test_loss_of_target_after_arrival_uses_another_view_not_old_coordinates():
    p=Port();observe=p.observe
    def current():
        value=observe()
        if len(p.moves)==1:value['objects']=[o for o in value['objects'] if o['kind']!='sock']
        return value
    p.observe=current;result=run(p)
    assert result['state']=='succeeded' and len(p.moves)==3
    assert p.phases==['pick','place']
    assert any(e['stage']=='approach_target_missing' for e in result['events'])


def test_failed_camera_survey_still_invalidates_frames_after_base_reposition():
    p=Port();location=[0.,0.,0.];p.pose=lambda:location
    def moving(*a,**k):
        location[0]=.04;raise ViewUnavailable('Opening not visible after camera baseline')
    p.look_at=moving;core=RoomCleanup(p,lambda _:None,clock=p.clock)
    core.seen={'sock':{'frame_id':'old'}}
    with pytest.raises(ViewUnavailable):core.view('look_at',{})
    assert core.base_revision==1 and not core.seen and p.rev==1


def test_real_motion_guard_is_never_retried_as_a_missing_route():
    p=Port();original=p.navigate
    def blocked(*a,**k):original(*a,**k);raise ValueError('Lidar stopped room motion: unaccounted near obstacle')
    p.navigate=blocked;result=run(p)
    assert result['state']=='failed' and len(p.moves)==1 and not p.phases and p.stop_count==1


def test_search_moves_to_unseen_floor_instead_of_zero_gain_rotation_loop():
    p=Port();p.survey_candidates=lambda:[dict(x=0.,y=0.,yaw=1.57,information_gain=0),
        dict(x=.45,y=0.,yaw=0.,information_gain=40)]
    p.plan=lambda goal,**k:dict(reachable=True,cost=0. if goal['x']==0 else .45)
    original=p.observe
    def observe():
        v=original()
        if not p.moves:v['objects']=[]
        return v
    p.observe=observe;result=run(p)
    assert result['state']=='succeeded' and p.moves[0][0]['x']==.45


def test_current_dual_scan_evidence_clears_ghost_but_preserves_concurrent_hits_and_unknown():
    grid=ScanMap();origin=np.array([0.,0.]);forward=np.array([[1.,0.]])
    grid.add(origin,forward,[.4]);ghost=grid.cell([.4,0.]);assert grid.occupied[ghost]
    grid.add_current_scans([(origin,forward,[1.]),(np.array([1.,0.]),-forward,[.6])])
    assert grid.occupied[ghost]  # second lidar still sees that obstacle
    grid.add_current_scans([(origin,forward,[1.])]);assert not grid.occupied[ghost]
    assert not grid.free[grid.cell([.6,.6])]  # a new empty map was not assumed


def rows(level,success=False):
    stages=['approach_sock','policy_pick','verify_hold','approach_basket','policy_place','verify_place']
    events=([dict(stage='search',result={'found':['sock','basket']})]+[dict(stage=s) for s in stages])[:level]
    return [dict(seed=i,success=success,collision_free=True,events=events,reason='Search budget exhausted; sock and basket not both observed') for i in range(10)]


def test_paired_executor_improvement_is_diagnostic_and_does_not_relax_or_qualify():
    control={'results':rows(0)};candidate={'results':rows(1)}
    result=comparison(control,candidate)
    assert result['accepted'] and len(result['improved_scene_ids'])==10
    assert result['simulation_qualified'] is False and result['neural_training'] is False
    candidate['results'][2]['collision_free']=False
    assert not comparison(control,candidate)['accepted']
    candidate={'results':rows(1)};control['results'][3]=rows(7,success=True)[3]
    assert not comparison(control,candidate)['accepted']
    with pytest.raises(ValueError):comparison({'results':rows(0)+rows(0)[:1]},candidate)


def test_only_identified_exhausted_proposals_allow_extra_coverage_or_approach():
    def diagnosis(cause,reason):return {'findings':[dict(cause=cause,reason=reason)]}
    result=propose([],diagnosis('search_or_perception','Search budget exhausted; sock not observed'))
    assert result==['expanded_coverage']
    assert propose(result,diagnosis('search_or_perception','Search budget exhausted')) is None
    assert propose([],diagnosis('search_or_perception','Vision backend unavailable')) is None
    assert propose([],diagnosis('alignment_or_reacquisition','Lidar stopped room motion')) is None
    before=mission_limits({});after=mission_limits({'executor_recovery':['expanded_coverage','expanded_approaches']})
    assert after.search_stops>before.search_stops and after.realignments>before.realignments
    assert after.target_max_age_s==before.target_max_age_s==2. and after.task_timeout_s==before.task_timeout_s


def test_program_compares_executor_repair_before_promoting_and_keeps_same_scene_ids(tmp_path,monkeypatch):
    folder=tmp_path/'room-learning'/'aaaaaaaaaaaa'
    folder.mkdir(parents=True);profile=folder/'handoff.json';write_json(profile,{'executor_recovery':[]})
    runner=RoomLearning(tmp_path,folder,{},lambda **v:None,set())
    baseline=dict(results=rows(0),successes=0);seeds=list(range(10))
    def evaluate(model,path,actual_seeds,stage):
        assert seeds==actual_seeds
        assert json.loads(path.read_text())['executor_recovery']==['expanded_coverage']
        return dict(results=rows(1),successes=0)
    monkeypatch.setattr(runner,'evaluate_room',evaluate)
    result=runner.repair_executor(Path('/unused-model'),profile,seeds,baseline,folder/'round-00')
    assert result is not None and runner.executor_program==['expanded_coverage']
    receipt=json.loads((folder/'executor-program.json').read_text())
    assert receipt['simulation_qualified'] is False


def test_waiting_source_upgrade_cannot_interrupt_a_live_stage_or_clear_operator_pause(tmp_path):
    path=Path(__file__).resolve().parents[1]/'bin/queue-room-cleanup.py'
    spec=importlib.util.spec_from_file_location('room_queue_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value=dict(state='waiting_for_local_learning',phase='waiting',active_pids=[])
    module.validate_waiting_upgrade(tmp_path,value)
    for state in ('running','operator_paused','failed'):
        with pytest.raises(ValueError):module.validate_waiting_upgrade(tmp_path,dict(value,state=state))
    (tmp_path/'pause.request').touch()
    with pytest.raises(ValueError):module.validate_waiting_upgrade(tmp_path,value)


def test_search_repair_can_expose_approach_cause_without_retraining_weights(tmp_path,monkeypatch):
    folder=tmp_path/'room-learning'/'aaaaaaaaaaaa';folder.mkdir(parents=True)
    profile=folder/'handoff.json';write_json(profile,{'executor_recovery':[]})
    runner=RoomLearning(tmp_path,folder,{},lambda **v:None,set());calls=[]
    def evaluate(model,path,seeds,stage):
        selected=json.loads(path.read_text())['executor_recovery'];calls.append(selected)
        if len(selected)==1:
            result=rows(1)
            for row in result:
                row['events'].append(dict(stage='approach_sock_moving'))
                row['reason']='Fresh geometric alignment did not converge: sock'
        else:result=rows(7,success=True)
        return dict(results=result,successes=sum(r['success'] for r in result))
    monkeypatch.setattr(runner,'evaluate_room',evaluate)
    repaired,path=runner.repair_executor(Path('/same-weights'),profile,list(range(10)),
        dict(results=rows(0),successes=0),folder/'round-00')
    assert repaired['successes']==10 and len(calls)==2
    assert calls==[['expanded_coverage'],['expanded_approaches','expanded_coverage']]
    assert json.loads(path.read_text())['executor_recovery']==runner.executor_program


def test_accepted_executor_repair_still_requires_twenty_fresh_complete_missions(tmp_path,monkeypatch):
    import simulator.room_learning as module
    folder=tmp_path/'room-learning'/'aaaaaaaaaaaa';folder.mkdir(parents=True)
    runner=RoomLearning(tmp_path,folder,{},lambda **v:None,set());calls=[]
    monkeypatch.setattr(runner,'check',lambda:None)
    monkeypatch.setattr(module,'training_seeds',lambda _:set())
    def profile(model,path):
        write_json(path,dict(executor_recovery=runner.executor_program));return path
    def evaluate(model,path,seeds,stage):
        calls.append((list(seeds),json.loads(path.read_text())['executor_recovery']))
        result=rows(0) if len(calls)==1 else rows(7,success=True)
        for row,seed in zip(result,seeds):row['seed']=seed
        return dict(results=result,successes=sum(r['success'] for r in result))
    monkeypatch.setattr(runner,'profile',profile);monkeypatch.setattr(runner,'evaluate_room',evaluate)
    monkeypatch.setattr(runner,'jobs',lambda *a,**k:pytest.fail('No gradient job fixes navigation'))
    result=runner.run_learning(tmp_path/'same-weights')
    assert len(calls)==4 and calls[0][0]==calls[1][0]
    assert all(c[1]==['expanded_coverage'] for c in calls[1:])
    assert len(set(calls[0][0]+calls[2][0]+calls[3][0]))==30
    assert len(result['groups'])==2
