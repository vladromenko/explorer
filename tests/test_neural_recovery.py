import json
import os
from pathlib import Path

import pytest

from simulator.learning import expert_segments,export_dataset,write_json
from simulator.neural_recovery import curriculum,diagnose,fresh_seeds,Recovery
from src.mobile_policy_contract import ACTOR_CLOCK,CONTRACT_SHA256,read_bundle


def test_student_actions_are_excluded_and_future_chunks_cannot_cross_handoff():
    rows=[dict(actor_elapsed_s=i*.05,training_target_valid=i not in (4,5)) for i in range(10)]
    segments=expert_segments(rows,20)
    assert [len(s) for s in segments]==[4,4]
    assert segments[1][0][0]==pytest.approx(.30)
    assert rows[4]['training_target_valid'] is False
    rows[7]['actor_elapsed_s']=0
    with pytest.raises(ValueError,match='discontinuous'):expert_segments(rows,20)
    with pytest.raises(ValueError,match='No contiguous'):expert_segments([dict(training_target_valid=False)]*4,20)


def test_corrective_export_keeps_clock_and_replay_weight_without_counting_repeats_as_new_scenes(tmp_path):
    from PIL import Image
    from lerobot.datasets import LeRobotDataset
    episode=tmp_path/'original';episode.mkdir()
    Image.new('RGB',(320,240)).save(episode/'frame.jpg')
    write_json(episode/'episode.json',dict(source='simulation',seed=123,state='complete',fps=20,
        success=True,learner_intervention={'frames':2},sample_contract_sha256=CONTRACT_SHA256))
    rows=[dict(at=100+i*.05,observation=[90.]*6+[0.]*3,action=[91.]*6+[0.]*3,
        wrist_rgb='frame.jpg',actor_elapsed_s=i*.05,training_target_valid=i not in (4,5)) for i in range(10)]
    (episode/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    destination=tmp_path/'dataset'
    provenance=export_dataset([episode],destination,correction_replays=4)
    data=LeRobotDataset('explorer/simulator',root=destination,video_backend='pyav')
    assert data.num_episodes==8 and len(data)==32
    assert float(data[4][ACTOR_CLOCK['feature']][0])==pytest.approx(.30/80)
    assert len(provenance['episodes'])==1 and provenance['episodes'][0]['samples']==32
    assert provenance['episodes'][0]['replay_weight']==4
    assert (episode/'samples.jsonl').read_text().count('false')==2


def test_curriculum_reuses_failures_never_recorded_validation_and_includes_user_basket():
    evidence=dict(failures={'approach':2,'transport':1,'place':0,'release':0},results=[
        dict(seed=11,phase='approach'),dict(seed=12,phase='transport'),dict(seed=11,phase='approach')])
    plan=curriculum(evidence,{12},5000)
    assert len(plan)==24 and len({r['seed'] for r in plan})==24
    assert plan[0]==dict(seed=11,phase='approach') and all(r['seed']!=12 for r in plan)
    assert sum(r['seed']%10==4 for r in plan)>=6
    assert set(r['seed']%5 for r in plan)==set(range(5))
    assert not set(fresh_seeds(set(range(100,110)),100,20))&set(range(100,110))


def test_post_rollout_diagnosis_distinguishes_failure_location_and_execution_errors(tmp_path):
    expected=[('approach',False,0.,[0,0,0]),('transport',True,0.,[1,0,0]),
              ('release',True,.7,[0,0,.1]),('place',True,0.,[0,0,.1])]
    for index,(_,lifted,fraction,cloth) in enumerate(expected):
        path=tmp_path/str(index)
        write_json(path/'episode.json',dict(generator='policy',attempt_completed=True,seed=index,success=False,outcome={'lifted':lifted}))
        (path/'samples.jsonl').write_text(json.dumps({'simulator_diagnostics':dict(
            sock_center_m=cloth,jaw_center_m=[0,0,.1],placement={'inside_fraction':fraction})}))
    result=diagnose(tmp_path)
    assert [r['phase'] for r in result['results']]==[r[0] for r in expected]
    assert result['privileged_diagnostics_only'] is True
    saved=json.loads((tmp_path/'0/episode.json').read_text());saved['error']='renderer failed'
    write_json(tmp_path/'0/episode.json',saved)
    with pytest.raises(ValueError,match='Execution error'):diagnose(tmp_path)


def test_completed_stage_rejects_changed_model_or_scene_on_resume(tmp_path):
    identifier='a'*12;folder=tmp_path/'neural-recovery'/identifier
    write_json(folder/'state.json',dict(id=identifier,state='running'))
    runner=Recovery(tmp_path,identifier);output=folder/'evaluation'
    spec=dict(kind='evaluate',model='first',output=str(output),seeds=[1])
    write_json(output/'request.json',dict(spec,stop=str(runner.stop)))
    write_json(output/'result.json',dict(successes=1))
    assert runner.jobs([spec])==[dict(successes=1)]
    with pytest.raises(ValueError,match='contract changed'):runner.jobs([dict(spec,seeds=[2])])


def test_pause_and_status_use_external_owner_and_block_duplicate_jobs(tmp_path):
    from test_simulator_server import client,auth
    identifier='a'*12;folder=tmp_path/'neural-recovery'/identifier
    write_json(tmp_path/'neural-recovery/current.json',{'id':identifier})
    write_json(folder/'state.json',dict(id=identifier,pid=os.getpid(),state='running',phase='trials',cycle_id='b'*12,
        model='/models/current',message='corrective shows',details={},completed=3,total=24,views=[]))
    with client(tmp_path) as connection:
        snapshot=connection.get('/api/state').json()
        assert snapshot['learning_status']['state']=='running'
        assert snapshot['learning_status']['external_recovery_id']==identifier
        assert snapshot['jobs'][-1]['external_recovery']
        assert connection.post('/api/jobs/train',json={},headers=auth(connection)).status_code==409
        assert connection.post('/api/cancel',json={},headers=auth(connection)).status_code==200
        assert (folder/'pause.request').exists()


def test_two_exact_independent_groups_survive_inactive_export_and_import(tmp_path):
    from test_simulator_learning import SimulatorLearningTests
    from simulator.transfer import export_package,import_package
    from src.simulation_task_contract import simulation_qualification
    helper=SimulatorLearningTests();helper.root=tmp_path
    model=helper.model();bundle,_=read_bundle(model)
    for part,seed in [('train',1),('validation',2)]:
        write_json(model/part/'explorer-provenance.json',dict(dataset_fps=20,episodes=[dict(episode={'seed':seed})]))
    reports=[]
    for start in (100,200):
        reports.append(dict(validation_kind='closed_loop_policy',task_id=None,successes=10,episodes=10,
            seeds=list(range(start,start+10)),results=[dict(success=True) for _ in range(10)],heldout_independent=True,
            model_weights_sha256=bundle['checkpoint_sha256']['model.safetensors'],model_checkpoint_sha256=bundle['checkpoint_sha256'],
            physics_validated=False,real_hardware_accepted=False))
    write_json(model/'physics-validation.json',reports[-1])
    write_json(model/'qualification-evidence.json',dict(groups=reports))
    assert simulation_qualification(model,bundle)['simulation_qualified']
    package=export_package(model,tmp_path/'package.tar.gz')
    imported=import_package(package['path'],tmp_path/'inactive')
    assert imported['job']['simulation_qualification']['simulation_qualified']
    assert imported['automatic_execution_allowed'] is False
    write_json(model/'qualification-evidence.json',dict(groups=[reports[-1]]*2))
    assert not simulation_qualification(model,bundle)['simulation_qualified']


@pytest.mark.parametrize('passes',[True,False])
def test_recovery_orchestration_reaches_only_verified_inactive_package(tmp_path,monkeypatch,passes):
    """Exercise stages, data retention and real archive verification without GPU training."""
    import shutil
    from test_simulator_learning import SimulatorLearningTests
    from simulator.learning import create_bundle
    from simulator import neural_recovery as module
    root=tmp_path;identifier='a'*12;cycle='b'*12
    helper=SimulatorLearningTests();helper.root=tmp_path
    source=helper.model()
    state=json.loads((source/'job.json').read_text());state.update(skill_id='toy_box',scene_profile='baseline')
    write_json(source/'job.json',state)
    for part,seed in [('train',1),('validation',2)]:
        write_json(source/part/'explorer-provenance.json',dict(dataset_fps=20,episodes=[dict(episode={'seed':seed})]))
    create_bundle(source,state)
    write_json(root/'learning-cycles'/(cycle+'.json'),dict(state='needs_diagnosis',tasks=[dict(task_id='toy_box')]))
    folder=root/'neural-recovery'/identifier
    write_json(folder/'state.json',dict(id=identifier,cycle_id=cycle,state='queued',task_id='toy_box',scene_profile='baseline',
        model=str(source),evidence='initial',seed_base=1400000000,code_revision='test'))
    runner=Recovery(root,identifier);calls=[]
    monkeypatch.setattr(runner,'check',lambda:None)
    monkeypatch.setattr(module,'diagnose',lambda _:dict(results=[dict(seed=11,phase='approach')],failures={p:int(p=='approach') for p in module.PHASES}))
    def jobs(specs,parallel=2):
        output=[]
        for spec in specs:
            calls.append(spec['kind']);path=Path(spec['output'])
            if spec['kind']=='correct':
                episode=path/('recovery-'+path.name)
                value=dict(success=True,task_id='toy_box',scene_profile='baseline',episode_folder=str(episode),learner_intervention={'phase':spec['phase']})
                write_json(episode/'episode.json',value);output.append(value)
            elif spec['kind']=='train':
                shutil.copytree(source,path)
                checkpoint=path/'model/checkpoints/001000/pretrained_model'
                shutil.copytree(path/'model/checkpoints/last/pretrained_model',checkpoint)
                write_json(path/'checkpoint-selection.json',{'candidates':[dict(checkpoint_relative='model/checkpoints/001000/pretrained_model',normalized_mae=.01)]})
                output.append({'model_id':path.name})
            else:
                bundle,_=read_bundle(spec['model']);successes=0 if Path(spec['model'])==source or not passes else 10
                output.append(dict(task_id='toy_box',scene_profile='baseline',successes=successes,episodes=10,seeds=spec['seeds'],
                    results=[dict(success=i<successes) for i in range(10)],heldout_independent=True,validation_kind='closed_loop_policy',
                    model_weights_sha256=bundle['checkpoint_sha256']['model.safetensors'],model_checkpoint_sha256=bundle['checkpoint_sha256'],
                    physics_validated=False,real_hardware_accepted=False))
        return output
    monkeypatch.setattr(runner,'jobs',jobs)
    runner._run()
    if passes:
        assert runner.state['state']=='ready_for_transfer' and runner.state['physical_ready'] is False
        assert calls.count('train')==1 and calls.count('correct')==24 and calls.count('evaluate')==4
        assert (folder/'archive-verification.json').is_file()
        assert json.loads((root/'learning-cycles'/(cycle+'.json')).read_text())['state']=='ready_for_transfer'
    else:
        assert runner.state['state']=='needs_algorithm_repair'
        assert calls.count('train')==3 and calls.count('evaluate')==6
        assert all(h['selected']==str(source) for h in runner.state['history'])
        assert not (root/'exports').exists()
        assert len(list((root/'runs').glob('*/checkpoint-selection.json')))==3
    assert json.loads((source/'job.json').read_text())==state
