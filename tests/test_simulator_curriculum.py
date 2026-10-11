import copy
import json
from pathlib import Path
from unittest.mock import Mock,patch
import pytest
from simulator.curriculum import NightQueue,task_spec
from simulator.server import Runtime,JobRequest
from src.simulation_task_contract import simulation_qualification
from simulator.transfer import export_curriculum,import_curriculum,export_package
import test_simulator_learning as fixtures


def test_queue_resumes_without_repeating_finished_stages_and_keeps_failures(tmp_path):
    queue=NightQueue(tmp_path)
    plan=queue.create(['toy_box','sock_basket'],2,1,16,8,100,rounds=1)
    calls=[]
    def execute(stage,task,options,cancelled):
        calls.append((task['task_id'],stage))
        if stage=='train' and task['task_id']=='toy_box' and len(calls)==2:
            raise InterruptedError('pause')
        if task['task_id']=='sock_basket':
            raise ValueError('no successful demonstrations')
        return dict(model_id='model',episodes=10,successes=0,heldout_independent=True)
    paused=queue.run(plan['id'],execute,lambda:False,lambda text:None)
    assert paused['state']=='paused'
    resumed=queue.run(plan['id'],execute,lambda:False,lambda text:None)
    assert resumed['state']=='complete' and resumed['failed_tasks']==1
    assert calls.count(('toy_box','trials'))==1
    assert resumed['tasks'][0]['simulation_qualified'] is False
    assert resumed['tasks'][1]['error']=='no successful demonstrations'
    assert resumed['automatic_hardware_execution'] is False


def test_unsuccessful_model_gets_more_data_and_actual_training_then_selects_best(tmp_path):
    queue=NightQueue(tmp_path);plan=queue.create(['toy_box'],2,10,16,8,100,rounds=3)
    seen=[]
    def execute(stage,task,options,cancelled):
        seen.append((stage,task['round'],copy.deepcopy(options)))
        return dict(model_id=str(task['round']),episodes=10,successes=3 if task['round']==1 else 1,heldout_independent=True)
    result=queue.run(plan['id'],execute,lambda:False,lambda text:None)
    task=result['tasks'][0]
    assert len(task['history'])==3
    assert task['stages']['train']['model_id']=='1'
    trials=[options for stage,_,options in seen if stage=='trials']
    assert [options['count'] for options in trials]==[2,4,6]
    assert len({options['seed'] for options in trials})==3
    assert [options['steps'] for stage,_,options in seen if stage=='train']==[10,20,30]


def test_restart_requires_explicit_resume_and_cancel_starts_no_work(tmp_path):
    q=NightQueue(tmp_path);plan=q.create(['sock_basket'],2,1,16,8,100)
    path=q.folder/(plan['id']+'.json');plan['state']='running';path.write_text(json.dumps(plan))
    fresh=NightQueue(tmp_path);assert fresh.get(plan['id'])['state']=='interrupted'
    execute=Mock();fresh.run(plan['id'],execute,lambda:True,lambda text:None)
    execute.assert_not_called()
    with pytest.raises(ValueError):fresh.get('../escape')


def test_model_requires_real_per_episode_results_and_exact_execution_settings(tmp_path):
    bundle={'skill_id':'toy_box','checkpoint_sha256':{'model.safetensors':'weights','config.json':'config'}}
    report=dict(validation_kind='closed_loop_policy',task_id='toy_box',heldout_independent=True,
                episodes=10,successes=10,physics_validated=False,real_hardware_accepted=False,model_weights_sha256='weights',model_checkpoint_sha256=bundle['checkpoint_sha256'],results=[dict(success=True)]*10)
    path=tmp_path/'physics-validation.json';path.write_text(json.dumps(report))
    assert simulation_qualification(tmp_path,bundle)['simulation_qualified']
    for changed in (dict(results=[]),dict(task_id='sock_basket'),dict(successes=9),dict(model_checkpoint_sha256={'model.safetensors':'weights'}),dict(heldout_independent=False)):
        path.write_text(json.dumps({**report,**changed}))
        assert not simulation_qualification(tmp_path,bundle)['simulation_qualified']
    assert simulation_qualification(tmp_path,bundle)['physical_accepted'] is False


def test_wrong_task_model_never_selected_for_cookie(tmp_path):
    runtime=Runtime.__new__(Runtime);runtime.data_root=tmp_path
    runtime.learning=Mock();runtime.learning.read_bundle.return_value=({'skill_id':'sock_basket'},None)
    with pytest.raises(ValueError,match='другой задаче'):
        runtime.model_for_request(JobRequest(task_id='cookie_delivery',model_id='other'))
    with pytest.raises(ValueError):
        runtime.model_for_request(JobRequest(model_id='../escape'))


def test_curriculum_transfer_preserves_task_binding_and_inactive_import(tmp_path):
    fixture=fixtures.SimulatorLearningTests();fixture.setUp()
    try:
        model=fixture.model();path=model/'bundle.json';bundle=json.loads(path.read_text());bundle['skill_id']='toy_box';path.write_text(json.dumps(bundle))
        package=export_package(model,tmp_path/'toy.tar.gz')
        plan=dict(id='a'*12,tasks=[dict(task_id='toy_box',state='complete',stages={'export':package})])
        result=export_curriculum(plan,tmp_path/'night.tar.gz')
        imported=import_curriculum(result['path'],tmp_path/'robot',register=True)
        assert len(imported['tasks'])==1
        assert import_curriculum(result['path'],tmp_path/'robot',register=True)['tasks'][0]['job']['id']==imported['tasks'][0]['job']['id']
        record=imported['tasks'][0]['job']
        assert record['state']=='imported_simulation'
        assert record['skill_id']=='toy_box'
        assert record['physical_success_verified'] is False
        assert record['simulation_qualification']['simulation_qualified'] is False
        assert imported['automatic_execution_allowed'] is False
        assert len(list((tmp_path/'robot/data/simulation-curricula').glob('*.json')))==1
    finally:fixture.tearDown()


def test_changed_report_cannot_unlock_imported_policy(tmp_path):
    import hashlib
    bundle={'skill_id':'toy_box','checkpoint_sha256':{'model.safetensors':'weights','config.json':'config'}}
    report=dict(validation_kind='closed_loop_policy',task_id='toy_box',heldout_independent=True,
        episodes=10,successes=10,physics_validated=False,real_hardware_accepted=False,
        model_weights_sha256='weights',model_checkpoint_sha256=bundle['checkpoint_sha256'],results=[dict(success=True)]*10)
    path=tmp_path/'physics-validation.json';path.write_text(json.dumps(report))
    (tmp_path/'source-transfer.json').write_text(json.dumps(dict(files={'physics-validation.json':hashlib.sha256(path.read_bytes()).hexdigest()})))
    assert simulation_qualification(tmp_path,bundle)['simulation_qualified']
    report['untrusted_extra']='edited';path.write_text(json.dumps(report))
    assert not simulation_qualification(tmp_path,bundle)['simulation_qualified']


def test_simulation_candidate_joins_only_observed_selector_after_closed_loop_proof(tmp_path):
    from mobile_policy_execution import MobilePolicyExecution
    bundle={'skill_id':'toy_box','checkpoint_sha256':{'model.safetensors':'weights','config.json':'config'}}
    folder=tmp_path/'data/learning-jobs'/('a'*32);folder.mkdir(parents=True)
    (folder/'job.json').write_text(json.dumps(dict(id=folder.name,state='imported_simulation',skill_id='toy_box')))
    report=dict(validation_kind='closed_loop_policy',task_id='toy_box',heldout_independent=True,
        episodes=10,successes=10,physics_validated=False,real_hardware_accepted=False,
        model_weights_sha256='weights',model_checkpoint_sha256=bundle['checkpoint_sha256'],results=[dict(success=True)]*10)
    runner=MobilePolicyExecution.__new__(MobilePolicyExecution);runner.root=tmp_path;runner.jobs=Mock();runner.jobs.status.return_value={'jobs':[]}
    with patch('mobile_policy_execution.read_bundle',return_value=(bundle,folder)):
        with pytest.raises(ValueError):runner._candidate('toy_box')
        (folder/'physics-validation.json').write_text(json.dumps(report))
        record,_,_=runner._candidate('toy_box')
        assert record['state']=='imported_simulation'
        with pytest.raises(ValueError):runner.start('toy_box',False)
        report['successes']=9;(folder/'physics-validation.json').write_text(json.dumps(report))
        with pytest.raises(ValueError):runner._candidate('toy_box')


def test_new_physical_scenes_have_free_objects_and_actual_room_geometry():
    import numpy as np
    from simulator.environment import ExplorerEnv
    env=ExplorerEnv(render=False)
    try:
        for task in ('cookie_delivery','toy_box','cloth_basket'):
            obs,info=env.reset(seed=101,options={'task_id':task})
            assert set(obs)=={'state','wrist'}
            assert info['task']['id']==task and not info['task']['success']
            assert info['provenance']['real_hardware_accepted'] is False
            if task=='cookie_delivery':
                assert env.model.geom('room_wall_-1').size[1]>.5
                assert env.sock_position[0]>1.
                assert env.model.joint('task_object_free').type==0
                assert len(env._sock_points())==8
                assert np.all(env.bin_half_size<.08)
    finally:env.close()


def test_selected_model_has_final_unused_scenes_and_resumable_final_export(tmp_path):
    queue=NightQueue(tmp_path);plan=queue.create(['toy_box'],2,10,16,8,100,rounds=2)
    seeds=[];interrupted=[False]
    def execute(stage,task,options,cancelled):
        if stage=='validate':seeds.append(options['validation_seed'])
        if stage=='export' and 'final_validate' in task['stages'] and not interrupted[0]:
            interrupted[0]=True;raise InterruptedError('pause final export')
        return dict(model_id=str(task['round']),episodes=10,successes=0,heldout_independent=True)
    first=queue.run(plan['id'],execute,lambda:False,lambda text:None)
    assert first['state']=='paused'
    assert first['tasks'][0]['training_complete']
    assert seeds==[20010000,20020000,500000000]
    second=queue.run(plan['id'],execute,lambda:False,lambda text:None)
    assert second['state']=='complete'
    assert seeds==[20010000,20020000,500000000]


def test_corrupt_old_queue_does_not_break_application(tmp_path):
    folder=tmp_path/'night-batches';folder.mkdir();(folder/('a'*12+'.json')).write_text('{broken')
    queue=NightQueue(tmp_path)
    assert queue.list()[0]['state']=='corrupt'
    assert (folder/('a'*12+'.json')).read_text()=='{broken'
