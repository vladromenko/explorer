import json
from pathlib import Path
from unittest.mock import Mock
import fcntl

import numpy as np
import pytest
from fastapi.testclient import TestClient

from simulator.automatic_learning import AutomaticLearning
from simulator.automatic_server import create_app
from simulator.language_learning import Interpretation,grounded_plan
from simulator.scene_recipes import SceneRequest,compile_recipes,reset_options,validate_recipe
from simulator.learning import write_json,sha256
from simulator.scan_navigation import ScanMap
from simulator.semantic_vision import associate


@pytest.fixture(autouse=True)
def deterministic_memory(monkeypatch):
    probe=lambda:dict(total_bytes=24*2**30,available_bytes=12*2**30)
    monkeypatch.setattr('simulator.resource_budget.memory_budget',probe)
    monkeypatch.setattr('simulator.skill_training.memory_budget',probe)
    monkeypatch.setattr('simulator.automatic_learning.keep_awake',lambda:None)


def interpretation(tasks=('sock_basket',),**changes):
    return Interpretation(**(dict(title='Убрать вещи',task_ids=list(tasks),supported=True,requested_handover='none',
        missing=[],clarification='',explanation='Найти и разместить.')|changes))


def recipe(task='sock_basket',**dimensions):
    parsed=interpretation((task,),scene_requests=[dict(task_id=task,**dimensions)])
    plan=grounded_plan('Убрать вещи',parsed,{'model':'test'})
    return compile_recipes(plan,plan['scene_requests'])['recipes'][0]


def test_scene_compiler_uses_exact_dimensions_and_preserves_broad_blind_validation():
    r=recipe(basket_height_cm=30,basket_diameter_cm=25,object_length_cm=16)
    scene=reset_options(r,101)['generated_scene']
    assert scene['parameters']['bin']==dict(shape='round',height_m=.30,inner_half_size_m=[.125,.125])
    assert scene['parameters']['sock']['length_m']==.16
    # An independent general group is not reduced to the requested easy scene.
    assert reset_options(r,101,validation=True)['generated_scene']['parameters']['bin']['height_m']==.12
    bad=dict(r,query='anything')
    with pytest.raises(ValueError):validate_recipe(bad)
    with pytest.raises(ValueError):SceneRequest(task_id='cookie_delivery',basket_height_cm=30)
    with pytest.raises(ValueError):SceneRequest(task_id='sock_basket',basket_height_cm=29)


def test_unknown_or_partial_task_never_becomes_an_unrelated_training_scene():
    plan=grounded_plan('Налей воду',interpretation((),supported=False,missing=['Нет навыка работы с водой']),{'model':'test'})
    result=compile_recipes(plan)
    assert result['state']=='needs_clarification' and not result['recipes']
    assert all(recipe(task)['task_id']==task for task in ('sock_basket','cloth_basket','toy_box','cookie_delivery'))


def test_scan_route_avoids_obstacles_and_never_crosses_unknown_cells():
    grid=ScanMap(extent=1.,resolution=.05,radius=.10)
    grid.free[:]=True
    grid.block_box([-.05,-1.],[.05,.1])
    route=grid.route([-.5,0.],[.5,0.])
    assert route[:,1].max()>.2
    grid.free[:]=False
    with pytest.raises(ValueError):grid.route([-.5,0.],[.5,0.])


def test_semantic_association_rejects_duplicate_geometry_and_keeps_hypothesis_scope():
    frame=dict(frame_id='rgbd1',optical_to_world=np.eye(4).tolist(),intrinsics=dict(fx=100,fy=100,cx=100,cy=100))
    item=dict(bounds_min=[-.1,-.1,1.],bounds_max=[.1,.1,1.1],center=[0,0,1.05])
    detection=dict(bbox=[88,88,112,112],confidence=.6,label='sock')
    result=associate([detection],[item],frame)
    assert result['semantic_evidence']['identity_proven'] is False
    with pytest.raises(ValueError):associate([detection],[item,item],frame)
    with pytest.raises(ValueError):associate([], [item],frame)


class Trainer:
    calls=[]
    def __init__(self,root,identifier):self.root=Path(root);self.id=identifier;self.path=self.root/'skill-training'/identifier/'state.json'
    def create(self,**options):
        if self.path.exists():return json.loads(self.path.read_text())
        state=dict(id=self.id,state='queued',completed_attempts=0,generation=0,phase_group='pick',
            options=dict(attempts=options['attempts']),simulation_ready=False,scene_recipe=options['scene_recipe'])
        self.save(state);return state
    def save(self,state):write_json(self.path,state)
    def run(self,**kwargs):
        state=json.loads(self.path.read_text());self.calls.append(state['scene_recipe']['task_id'])
        target=self.root/'exports'/('observed-skill-'+self.id+'.tar.gz');target.parent.mkdir(exist_ok=True);target.write_bytes(b'test-only-qualified-child')
        state.update(state='simulation_qualified',simulation_ready=True,archive=dict(path=str(target),sha256=sha256(target),verified=True));self.save(state)
        return state


def engine(tmp_path,tasks=('sock_basket',)):
    planner=Mock();planner.plan.return_value=(interpretation(tasks),{'model':'test'})
    book=AutomaticLearning(tmp_path,planner=planner,trainer_factory=Trainer,scene_builder=lambda value,task:dict(preview='test',manifest=dict(recipe_sha256=task['recipe']['sha256'])))
    book.model_status=Mock(return_value=dict(ready=True));return book


def test_text_to_multiple_scenes_learning_and_verified_result_is_one_pipeline(tmp_path):
    Trainer.calls=[];book=engine(tmp_path,('sock_basket','cloth_basket','toy_box','cookie_delivery'))
    row=book.create('Убери вещи и привези печенье в упаковке на поднос',attempts=8)
    result=book.advance(row['id'])
    assert result['state']=='simulation_qualified' and result['simulation_ready']
    assert Trainer.calls==['sock_basket','cloth_basket','toy_box','cookie_delivery']
    assert result['package']['verified'] and Path(result['package']['path']).is_file()
    assert not result['physical_ready'] and not result['automatic_hardware_execution']
    assert len({t['training_id'] for t in result['tasks']})==4


def test_active_physics_owner_is_not_stopped_or_duplicated(tmp_path):
    book=engine(tmp_path);row=book.create('Убрать носок',attempts=8)
    folder=tmp_path/'skill-training';folder.mkdir()
    with (folder/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=book.advance(row['id'])
        assert result['state']=='waiting_for_current' and result['plan']['fully_supported']
        assert not any((folder/t['training_id']).exists() for t in result['tasks'])
    result=book.advance(row['id']);assert result['state']=='simulation_qualified'


def test_operator_pause_is_not_automatically_removed(tmp_path):
    book=engine(tmp_path);row=book.create('Убрать носок',attempts=8)
    book.pause(row['id']);result=book.advance(row['id'])
    assert result['state']=='operator_paused';book.planner.plan.assert_not_called()
    book.resume(row['id']);assert book.get(row['id'])['id']==row['id']
    assert book.advance(row['id'])['state']=='simulation_qualified'


def test_budget_resume_extends_same_child_not_an_identical_finished_budget(tmp_path):
    book=engine(tmp_path);row=book.create('Убрать носок',attempts=100)
    row['state']='budget_paused';row['tasks']=[dict(task_id='sock_basket',training_id='a'*12)]
    trainer=Trainer(tmp_path,'a'*12);trainer.save(dict(state='budget_paused',completed_attempts=96,options=dict(attempts=100)))
    book.save(row);book.resume(row['id'])
    saved=json.loads(trainer.path.read_text());assert saved['options']['attempts']==196
    assert book.get(row['id'])['tasks'][0]['training_id']=='a'*12


def test_new_ui_api_is_local_authenticated_and_has_no_act_or_manual_controls(tmp_path):
    app=create_app(tmp_path,engine_factory=lambda root:engine(root),start=False)
    with TestClient(app,base_url='http://127.0.0.1') as client:
        headers={'X-Explorer-Sim-Token':client.get('/api/session').json()['token']}
        assert client.post('/api/learn',json={'text':'Носок в корзину'}).status_code==403
        result=client.post('/api/learn',json={'text':'Носок в корзину'},headers=headers)
        assert result.status_code==202 and result.json()['state']=='queued'
        identifier=result.json()['id']
        assert client.post('/api/learn',json={'text':'x'},headers=headers|{'Origin':'http://evil.example'}).status_code==403
        assert client.get('/api/requests/'+identifier+'/package').status_code==409
        html=client.get('/').text
        assert 'id="learn"' in html and 'id="request"' in html
        assert 'request-steps' not in html and 'record' not in html and 'cycle-start' not in html


def test_room_training_is_simulation_scope_without_granting_real_map_acceptance():
    parsed=interpretation(('cookie_delivery',))
    text='Научись привозить печенье в упаковке из другой комнаты на поднос'
    old=grounded_plan(text,parsed,{'model':'test'})
    new=grounded_plan(text,parsed,{'model':'test'},simulation_only=True)
    assert not old['fully_supported'] and new['fully_supported']
    assert not new['physical_ready'] and not new['automatic_hardware_execution']
    assert compile_recipes(new)['recipes'][0]['layout']=='two_rooms_doorway'


def test_requested_scene_dimensions_add_extra_user_validation_without_removing_general_sets(tmp_path):
    from simulator.skill_training import SkillTraining
    trainer=SkillTraining(tmp_path,'c'*12)
    state=trainer.create(attempts=8,task_id='sock_basket',scene_recipe=recipe(basket_height_cm=30,basket_diameter_cm=25))
    assert len(state['validation_seeds'])==3 and all(len(group)==10 for group in state['validation_seeds'])
    assert len({s for group in state['validation_seeds'] for s in group})==30


def test_shallow_tray_requires_measured_rim_support_and_two_camera_positions():
    from simulator.surface_destination import observed_trays,combine_trays
    height,width=240,320;y,x=np.indices((height,width));z=np.zeros((height,width))
    z[(abs(x-160)<74)&(abs(y-120)<74)]=.008
    rim=((abs(x-160)<82)&(abs(y-120)<82))&((abs(x-160)>73)|(abs(y-120)>73));z[rim]=.035
    transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=.5
    frame=dict(frame_id='tray1',simulation_time=1.,intrinsics=dict(fx=500.,fy=500.,cx=160.,cy=120.),optical_to_world=transform.tolist())
    record=dict(frame=frame,depth=.5-z,robot_depth=np.full_like(z,np.inf))
    candidates=observed_trays(record);assert len(candidates)==1
    candidate=candidates[0]
    views=[dict(candidate=candidate,frame_id='tray1',camera_position=[0,0,.5])]
    with pytest.raises(__import__('simulator.rgbd_observation',fromlist=['ObservationRefused']).ObservationRefused):combine_trays(views)
    views.append(dict(candidate=dict(candidate,frame_id='tray2',simulation_time=1.1),frame_id='tray2',camera_position=[.03,0,.5]))
    assert combine_trays(views)['evidence']['multiview_consistent']
    record['depth'][:]=.5
    assert not observed_trays(record)


def test_explicit_user_dimensions_cannot_be_ignored_or_invented_by_language_model():
    from simulator.scene_recipes import explicit_scene_requests
    text='Убирай носок в круглую корзину высотой 30 см и диаметром 25 см'
    requests=explicit_scene_requests(text,['sock_basket'],[])
    assert requests==[dict(task_id='sock_basket',basket_height_cm=30.,basket_diameter_cm=25.)]
    assert explicit_scene_requests('Носок в корзину',['sock_basket'],[dict(task_id='sock_basket',basket_height_cm=8)])==[]
    changed=explicit_scene_requests(text+'\nУточнение пользователя: Диаметр корзины 30 см',['sock_basket'])
    assert changed[0]['basket_height_cm']==30 and changed[0]['basket_diameter_cm']==30
    assert explicit_scene_requests('Корзина высотой 0,3 м и диаметром 250 мм',['sock_basket'])==requests
    with pytest.raises(ValueError):explicit_scene_requests('Корзина высотой 29 см',['sock_basket'])


def test_failed_first_skill_does_not_skip_other_independent_requested_skills(tmp_path):
    class PartialTrainer(Trainer):
        def run(self,**kwargs):
            saved=json.loads(self.path.read_text())
            if saved['scene_recipe']['task_id']=='sock_basket':
                saved.update(state='needs_diagnosis',reason='Захват не подтверждён',simulation_ready=False);self.save(saved);return saved
            return super().run(**kwargs)
    book=engine(tmp_path,('sock_basket','toy_box'));book.trainer_factory=PartialTrainer
    row=book.create('Убрать носок и кубик',attempts=8);result=book.advance(row['id'])
    assert result['state']=='needs_diagnosis' and not result['simulation_ready']
    assert [t['state'] for t in result['tasks']]==['needs_diagnosis','simulation_qualified']
    assert not result.get('package')


def test_pending_pause_does_not_allow_resume_before_workers_stop(tmp_path):
    book=engine(tmp_path);row=book.create('Носок в корзину',attempts=8);book.active_id=row['id']
    assert book.pause(row['id'])['state']=='pausing'
    assert book.cancelled(row['id'])
    with pytest.raises(ValueError):book.resume(row['id'])


def test_semantic_training_reserves_memory_for_vision_subprocess_as_well(tmp_path,monkeypatch):
    import simulator.skill_training as training
    trainer=training.SkillTraining(tmp_path,'d'*12);state=trainer.create(attempts=8,scene_recipe=recipe())
    counts=[]
    class Pool:
        def __init__(self,**kwargs):counts.append(kwargs['max_workers'])
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(training,'ProcessPoolExecutor',Pool)
    assert trainer._batch(state,[],lambda:False,lambda s:None)==[]
    assert counts==[2] and state['resource_measurement']['reserved_bytes_per_worker']==int(3.2*2**30)


def test_compatible_old_parameter_search_is_only_an_unverified_warm_start(tmp_path):
    from simulator.skill_training import SkillTraining
    old=SkillTraining(tmp_path,'e'*12);state=old.create(attempts=8)
    state.update(generation=1,state='needs_diagnosis');state['incumbent']['pick_fraction']=.4;old.save(state)
    new=SkillTraining(tmp_path,'f'*12);started=new.create(attempts=8,scene_recipe=recipe())
    assert started['incumbent']['pick_fraction']==.4
    assert started['initialization']['unverified'] is True
    assert started['initialization']['old_episodes_counted_for_qualification'] is False
    assert not started['episodes'] and not started['simulation_ready']


def test_restart_recovers_running_request_as_explicit_resume_without_touching_queued_or_user_pause(tmp_path):
    book=engine(tmp_path)
    running=book.create('Носок в корзину',attempts=8);running['state']='running'
    running['tasks']=[dict(task_id='sock_basket',training_id='1'*12,progress=dict(active_workers=[dict(slot=0)]))]
    book.save(running);folder=tmp_path/'skill-training'/('1'*12);folder.mkdir(parents=True)
    (folder/'keep-checkpoint').write_text('unchanged')
    queued=book.create('Следующий навык',attempts=8);paused=book.create('Моя пауза',attempts=8);book.pause(paused['id'])
    book.recover_requests()
    assert book.get(running['id'])['state']=='interrupted'
    assert not book.get(running['id'])['tasks'][0]['progress']['active_workers']
    assert (folder/'pause.request').exists() and (folder/'keep-checkpoint').read_text()=='unchanged'
    assert book.get(queued['id'])['state']=='queued'
    assert book.get(paused['id'])['state']=='operator_paused'


def test_pause_holds_pending_queue_and_resume_does_not_remove_other_manual_pauses(tmp_path):
    book=engine(tmp_path);first=book.create('Первая задача',attempts=8);second=book.create('Следующая задача',attempts=8)
    manual=book.create('Отдельная ручная пауза',attempts=8);book.pause(manual['id']);book.resume_queue()
    book.pause(first['id']);assert book.pause_info()['owner_id']==first['id']
    waiting=book.advance(second['id']);assert waiting['state']=='waiting_for_current' and not any(t.get('scene') for t in waiting['tasks'])
    book.resume(first['id']);assert book.pause_info() is None
    assert book.get(manual['id'])['state']=='operator_paused'
    assert book.advance(first['id'])['state']=='simulation_qualified'


def test_worker_phase_comes_from_actual_recording_not_optimizer_group(tmp_path):
    from simulator.automatic_server import summarized
    row=dict(id='2'*12,state='running',task_id='sock_basket',completed_attempts=3,options=dict(attempts=100,hours=24),
        phase_group='place',active_workers=[dict(slot=0,episode_id='3'*12)])
    folder=tmp_path/'skill-training'/row['id']/'episodes'/('3'*12);folder.mkdir(parents=True)
    write_json(folder/'progress.json',dict(phase='skill_verify_hold'))
    result=summarized(row,tmp_path)
    assert result['phase']=='place' and result['active_workers'][0]['phase']=='skill_verify_hold'


def test_stale_progress_cannot_erase_a_pause_or_its_history(tmp_path):
    book=engine(tmp_path);row=book.create('Пауза без потери данных',attempts=8)
    stale=book.get(row['id']);stale['state']='running'
    book.active_id=row['id'];paused=book.pause(row['id']);pause_event=paused['events'][-1]
    book.save(stale) # A worker callback holding a dictionary from before pause.
    assert book.get(row['id'])['state']=='pausing'
    assert pause_event in book.get(row['id'])['events'] and book.cancelled(row['id'])
    book.event(stale,'operator_paused','Работники остановлены')
    book.active_id=None;book.resume(row['id'])
    assert not book.cancelled(row['id']) and book.get(row['id'])['state']=='queued'
    assert pause_event in book.get(row['id'])['events']
