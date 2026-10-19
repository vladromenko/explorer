"""Correction/data control tests, not neural or physical qualification."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.learning import inspect_episode,expert_segments,write_json
from simulator.room_learning import diagnose,correction_plan,PhaseRecorder,RoomLearning


def failed(seed,stage,**extra):
    return dict(seed=seed,success=False,reason='unconfirmed',events=[{'stage':stage}],**extra)


def test_neural_updates_do_not_pretend_to_fix_search_navigation_or_collisions():
    for row in (failed(1,'search'),failed(2,'approach_sock_moving'),
                failed(3,'search',collision_free=False)):
        diagnosis=diagnose([row])
        assert not diagnosis['neural_correction_supported']
        with pytest.raises(ValueError):correction_plan(diagnosis,set(),100)
    diagnosis=diagnose([failed(4,'policy_pick_running'),failed(5,'policy_place_running')])
    assert diagnosis['neural_correction_supported']
    assert diagnosis['causes']==['pickup_or_transport','placement']
    assert all(not r['root_cause_proven'] for r in diagnosis['findings'])
    after_pick=failed(6,'policy_pick_running');after_pick['events'].append({'stage':'approach_basket_moving'})
    assert not diagnose([after_pick])['neural_correction_supported']


def test_corrective_scene_selection_reuses_failure_without_touching_holdouts():
    diagnosis=diagnose([failed(4,'policy_pick_running'),failed(5,'policy_place_running')])
    plan=correction_plan(diagnosis,{5,100,101},100)
    assert len(plan)==24 and len({r['seed'] for r in plan})==24
    assert plan[0]==dict(seed=4,from_failure=True)
    assert not {5,100,101}&{r['seed'] for r in plan}
    assert set(r['seed']%5 for r in plan)==set(range(5))


def test_expert_recording_uses_preaction_pixels_and_actual_bounded_commands(tmp_path):
    command=np.array([90.,70.,80.,50.,90.,30.,0.,0.,0.]);env=SimpleNamespace(command=command,
        data=SimpleNamespace(time=0.),domain_variation={'parameters':{'bin':{}}})
    def step(action,applied=False):
        assert applied is True
        env.command=np.asarray(action).copy();env.data.time+=.05
    env.step=step;env.render_rgb=lambda _:np.full((240,320,3),round(env.command[0]),np.uint8)
    recorder=PhaseRecorder(env,tmp_path/'correction',dict(pick_elapsed_s=6.),400)
    recorder.begin('pick',{})
    for _ in range(4):env.step(env.command+np.r_[10.,np.zeros(8)])
    paths=recorder.finish(True,{'success':True})
    assert env.step is step and len(paths)==1
    episode,rows,fps,_=inspect_episode(paths[0]);segments=expert_segments(rows,fps)
    assert episode['generator']=='room_observed_geometric_teacher' and episode['corrective_demonstration']
    assert rows[0]['observation_state'][0]==90 and rows[0]['applied_action'][0]==92
    assert rows[0]['actor_elapsed_s']==6. and segments[0][-1][0]==pytest.approx(6.15)
    assert all(row['applied_action'][6:]==[0.,0.,0.] for row in rows)
    assert json.loads((Path(paths[0])/'episode.json').read_text())['full_mission_success'] is True


def test_failed_teacher_evidence_is_preserved_but_never_enters_success_dataset(tmp_path):
    env=SimpleNamespace(command=np.array([90.]*5+[30.,0.,0.,0.]),data=SimpleNamespace(time=0.),
        domain_variation={},step=lambda *a,**k:None,render_rgb=lambda _:np.zeros((240,320,3),np.uint8))
    recorder=PhaseRecorder(env,tmp_path/'correction',dict(pick_elapsed_s=6.),1)
    recorder.begin('pick',{});env.step(env.command)
    assert recorder.finish(False,{})==[]
    evidence=json.loads(next(tmp_path.glob('correction/*/episode.json')).read_text())
    assert evidence['success'] is False and evidence['physics_verified_success'] is False


def test_room_loop_diagnoses_executor_without_starting_a_useless_gradient_job(tmp_path,monkeypatch):
    import simulator.room_learning as module
    folder=tmp_path/'room-learning'/'aaaaaaaaaaaa'
    folder.mkdir(parents=True)
    state={};save=lambda **v:state.update(v)
    runner=RoomLearning(tmp_path,folder,state,save,set())
    monkeypatch.setattr(runner,'check',lambda:None)
    monkeypatch.setattr(module,'training_seeds',lambda _:set())
    monkeypatch.setattr(runner,'profile',lambda model,path:path)
    monkeypatch.setattr(runner,'evaluate_room',lambda *a:dict(successes=0,results=[failed(10,'search')]))
    monkeypatch.setattr(runner,'jobs',lambda *a,**k:pytest.fail('No blind retraining on search failure'))
    assert runner.run_learning(tmp_path/'model') is None
    assert state['state']=='needs_room_algorithm_repair'
    assert (folder/'round-00/diagnosis.json').exists()


def test_room_loop_requires_both_fresh_groups_before_export(tmp_path,monkeypatch):
    import simulator.room_learning as module
    folder=tmp_path/'room-learning'/'aaaaaaaaaaaa';folder.mkdir(parents=True)
    state={};runner=RoomLearning(tmp_path,folder,state,lambda **v:state.update(v),{1601008001})
    monkeypatch.setattr(runner,'check',lambda:None)
    monkeypatch.setattr(module,'training_seeds',lambda _:{1,2})
    monkeypatch.setattr(runner,'profile',lambda model,path:path)
    evaluated=[]
    def evaluate(model,profile,seeds,path):
        evaluated.append(seeds)
        return dict(successes=len(seeds),episodes=len(seeds),results=[dict(seed=s,success=True) for s in seeds])
    monkeypatch.setattr(runner,'evaluate_room',evaluate)
    result=runner.run_learning(tmp_path/'model')
    assert len(evaluated)==3 and len(result['groups'])==2
    assert [len(s) for s in evaluated]==[10,50,50]
    assert len(set().union(*map(set,evaluated)))==110
    protocol=json.loads((folder/'round-00/acceptance100-protocol.json').read_text())
    assert protocol['required_successes']==100 and protocol['allowed_collisions']==0
    assert not {1,2,1601008001}&set().union(*map(set,evaluated))


def test_failed_room_manipulation_drives_data_collection_real_train_request_and_new_tests(tmp_path,monkeypatch):
    import simulator.room_learning as module
    import simulator.checkpoint_experiment as checkpoints
    folder=tmp_path/'room-learning'/'aaaaaaaaaaaa';folder.mkdir(parents=True)
    state={};runner=RoomLearning(tmp_path,folder,state,lambda **v:state.update(v),set())
    monkeypatch.setattr(runner,'check',lambda:None)
    monkeypatch.setattr(module,'training_seeds',lambda _:set())
    monkeypatch.setattr(module,'historical_episodes',lambda *a:[tmp_path/'old'])
    monkeypatch.setattr(runner,'profile',lambda model,path:path)
    monkeypatch.setattr(checkpoints,'shortlist',lambda _ : [])
    requests=[];evaluations=[]
    def jobs(specs,parallel=None):
        requests.extend(specs)
        if specs[0]['kind']=='room_correct':
            return [dict(success=True,episodes=[str(tmp_path/f'expert-{i}')],
                student_interventions=[dict(applied_steps=spec['student_steps'])] if spec['student_steps'] else [])
                for i,spec in enumerate(specs)]
        assert len(specs)==1 and parallel==1 and specs[0]['kind']=='train'
        write_json(Path(specs[0]['output'])/'checkpoint-selection.json',{})
        return [dict(state='trained_simulation_unvalidated')]
    def evaluate(model,profile,seeds,path):
        evaluations.append(list(seeds))
        if len(evaluations)==1:return dict(episodes=10,successes=0,results=[failed(s,'policy_pick_running') for s in seeds])
        return dict(episodes=len(seeds),successes=len(seeds),results=[dict(seed=s,success=True) for s in seeds])
    monkeypatch.setattr(runner,'jobs',jobs);monkeypatch.setattr(runner,'evaluate_room',evaluate)
    result=runner.run_learning(tmp_path/'model')
    assert result is not None and len(requests)==25
    assert [r['student_steps'] for r in requests[:-1]][:4]==[0,50,0,100]
    training=requests[-1];assert str(tmp_path/'old') in training['episodes'] and len(training['episodes'])==25
    assert len(training['corrective_training_seeds'])==24
    assert not set(evaluations[0])&set(evaluations[1])
    assert not set(training['corrective_training_seeds'])&set(evaluations[-1]+evaluations[-2])


def test_room_loop_does_not_train_when_only_unperturbed_teacher_examples_succeed(tmp_path,monkeypatch):
    import simulator.room_learning as module
    folder=tmp_path/'room-learning'/'aaaaaaaaaaaa';folder.mkdir(parents=True)
    state={};runner=RoomLearning(tmp_path,folder,state,lambda **v:state.update(v),set())
    monkeypatch.setattr(runner,'check',lambda:None)
    monkeypatch.setattr(module,'training_seeds',lambda _:set())
    monkeypatch.setattr(runner,'profile',lambda model,path:path)
    monkeypatch.setattr(runner,'evaluate_room',lambda *a:dict(successes=0,results=[failed(10,'policy_pick_running')]))
    def only_controls(specs,parallel=None):
        assert all(s['kind']=='room_correct' for s in specs)
        return [dict(seed=s['seed'],success=not s['student_steps'],events=[],reason='correction failed',
            student_interventions=[],episodes=[]) for s in specs]
    monkeypatch.setattr(runner,'jobs',only_controls)
    assert runner.run_learning(tmp_path/'model') is None
    assert state['state']=='needs_room_algorithm_repair'
    coverage=json.loads((folder/'round-00/correction-coverage.json').read_text())
    assert coverage['full_successes']==12 and coverage['successful_student_state_corrections']==0


def test_train_worker_really_requests_eight_passes_and_preserves_warm_start(tmp_path,monkeypatch):
    import simulator.learning as learning
    from simulator.neural_recovery import worker
    captured={}
    def train(folder,episodes,**kwargs):
        captured.update(kwargs);return dict(state='trained_simulation_unvalidated')
    monkeypatch.setattr(learning,'train_job',train)
    request=tmp_path/'request.json'
    write_json(request,dict(kind='train',output=str(tmp_path/'training'),stop=str(tmp_path/'pause'),
        model=str(tmp_path/'prior'),episodes=[str(tmp_path/'episode')],seed=1,task_id='sock_basket',title='test'))
    worker(request)
    assert captured['minimum_epochs']==8 and captured['correction_replays']==4
    assert captured['warm_start']==str(tmp_path/'prior')


def test_observed_basket_wall_base_cannot_become_a_separate_sock_crop():
    from simulator.semantic_vision import floor_material_candidates
    wall=dict(bounds_min=[-.78,-.80,.005],bounds_max=[-.55,-.68,.073])
    basket=dict(bounds_min=[-.80,-.94,.08],bounds_max=[-.55,-.68,.30])
    sock=dict(bounds_min=[-1.2,1.1,.005],bounds_max=[-1.,1.2,.02])
    assert floor_material_candidates({'low_objects':[wall,sock]},[basket])==[sock]
    shallow_wall=dict(wall,bounds_max=[-.55,-.68,.03])
    assert floor_material_candidates({'low_objects':[shallow_wall]},[basket])==[]


def test_room_status_pause_and_download_refer_to_full_room_owner(tmp_path,monkeypatch):
    from test_simulator_server import client,auth
    import simulator.room_control as control
    import simulator.neural_recovery as local
    from simulator.learning import sha256
    archive=tmp_path/'exports/room.tar.gz';archive.parent.mkdir();archive.write_bytes(b'checked fixture')
    state=dict(id='aaaaaaaaaaaa',pid=1,alive=True,state='running',phase='room_validation',
        model='/models/current',message='independent rooms',completed=3,total=10,details={},views=[],
        archive={'path':str(archive),'sha256':sha256(archive)})
    monkeypatch.setattr(local,'current',lambda _:None)
    monkeypatch.setattr(control,'current',lambda _:state)
    paused=[];monkeypatch.setattr(control,'control',lambda root,action:paused.append(action) or {})
    with client(tmp_path) as connection:
        snapshot=connection.get('/api/state').json()
        assert snapshot['learning_status']['external_room_id']==state['id']
        assert snapshot['learning_status']['simulation_ready'] is False
        assert snapshot['jobs'][-1]['external_room_learning']
        assert connection.post('/api/jobs/train',json={},headers=auth(connection)).status_code==409
        assert connection.post('/api/cancel',json={},headers=auth(connection)).status_code==200
        assert paused==['pause']
        assert connection.get('/api/room-learning/package').status_code==404
        state.update(state='ready_for_observed_transfer',alive=False)
        assert connection.get('/api/room-learning/package').status_code==200
        archive.write_bytes(b'tampered');assert connection.get('/api/room-learning/package').status_code==404


def test_actor_opening_is_admitted_only_above_a_confirmed_observed_opening():
    from simulator.room_policy import admit_release
    target=dict(bounds_min=[.2,-.125,0.],bounds_max=[.45,.125,.3],uncertainty_m=.012,
        evidence={'shape':'raised_walls_with_visible_opening'})
    admit_release(target,[.325,0.,.35])
    for jaw in ([.325,0.,.31],[.21,0.,.35],[.325,.2,.35]):
        with pytest.raises(ValueError):admit_release(target,jaw)


def test_expanded_room_dataset_does_not_silently_reduce_eight_full_passes():
    from simulator.learning import training_budget,training_config
    budget=training_budget(1000,1100000,32,8,maximum_steps=1000000)
    assert budget['effective_steps']==275000 and budget['expected_dataset_passes']==8
    assert not budget['step_cap_reached']
    training_config(Path('/unused'),275000,'cpu',32,1,maximum_steps=1000000)


def test_final_truth_rejects_transient_ever_success_after_object_has_left():
    from types import SimpleNamespace
    from simulator.room_validation import independent_outcome
    env=SimpleNamespace(state=lambda:dict(task=dict(success=True,placed=True)),placement_hold_s=0.,
        placement_diagnostics=dict(inside_fraction=.5,within_height=True,sock_speed_max_m_s=.01),contacts=[False,False])
    result=independent_outcome(env)
    assert result['ever_success'] and not result['success'] and not result['placed']
    env.placement_hold_s=1.1;env.placement_diagnostics['inside_fraction']=1.
    assert independent_outcome(env)['success']
    env.contacts=[True,False]
    assert not independent_outcome(env)['success']
