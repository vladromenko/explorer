"""Ground language requests without letting proposed intent grant capabilities."""
import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from simulator.language_learning import Interpretation, LearningGoals, LocalLanguageModel, grounded_plan
from simulator.server import JobRequest, Runtime
from simulator.transfer import _validate_learning_goal, export_curriculum, import_curriculum
from test_simulator_server import FakeEnv, client, auth


def intent(**changes):
    values=dict(title='Носок в корзину',task_ids=['sock_basket'],supported=True,
        requested_handover='none',missing=[],clarification='',explanation='Учить захват и размещение ткани.')
    return Interpretation(**(values | changes))


def plan(**changes):
    return grounded_plan('Научись убирать носки в корзину',intent(**changes),{'model':'test'})


def test_owned_language_service_resolves_app_binary_and_disables_cloud(tmp_path, monkeypatch):
    binary=tmp_path/'App Resources/ollama'
    binary.parent.mkdir(); binary.write_text('test binary')
    link=tmp_path/'ollama'; link.symlink_to(binary)
    monkeypatch.setattr('simulator.language_learning.shutil.which',lambda name:str(link))
    process=Mock(); process.poll.return_value=None
    spawn=Mock(return_value=process)
    monkeypatch.setattr('simulator.language_learning.subprocess.Popen',spawn)
    model=LocalLanguageModel(tmp_path/'data')
    model.call=Mock(side_effect=[OSError('not running'),{'version':'test'}])
    try:
        model.ensure()
        args,kwargs=spawn.call_args
        assert args[0]==[str(binary.resolve()),'serve']
        assert kwargs['env']['OLLAMA_NO_CLOUD']=='1'
        assert kwargs['env']['OLLAMA_HOST']=='127.0.0.1:11435'
        assert kwargs['env']['OLLAMA_MODELS']==str(tmp_path/'data/llm-models')
    finally:model.close()


def test_language_clarification_is_a_separate_latest_user_turn(tmp_path):
    model=LocalLanguageModel(tmp_path)
    model.ensure=Mock();model.status=Mock(return_value={'available':True,'digest':'test'})
    model.call=Mock(return_value={'message':{'content':intent().model_dump_json()}})
    model.plan('Печенье в упаковке в руку\nУточнение пользователя: Лучше на поднос')
    body=model.call.call_args.args[1]
    assert [message['role'] for message in body['messages']]==['system','user','assistant','user']
    assert body['messages'][-1]['content']=='Лучше на поднос'
    assert body['messages'][1]['content']=='Печенье в упаковке в руку'


def test_text_never_grants_unimplemented_handover_or_hardware_acceptance():
    value=grounded_plan('Привези печенье в упаковке и дай мне в руку',
        intent(task_ids=['cookie_delivery'],requested_handover='supported_surface'),{'model':'test'})
    assert value['fully_supported'] is False
    assert value['requested_handover']=='human_hand' and value['missing']
    assert value['physical_ready'] is False and value['automatic_hardware_execution'] is False
    assert value['tasks'][0]['task_id']=='cookie_delivery'
    assert value['tasks'][0]['steps'][1]['mode']=='existing_real_navigation_requires_map'
    assert 'попробует обучить' in value['explanation'] and 'заранее не гарантирован' in value['explanation']


def test_unknown_skills_shell_and_fictitious_acceptance_rejected():
    with pytest.raises(ValidationError):
        intent(task_ids=['run_shell'])
    with pytest.raises(ValidationError):
        intent(hardware_accepted=True)
    with pytest.raises(ValueError):
        plan(task_ids=['sock_basket','sock_basket'])
    empty=plan(task_ids=[],supported=False,clarification='Что нужно убрать?')
    assert not empty['fully_supported'] and empty['missing']


def test_user_can_change_handover_goal_to_supported_surface_in_followup():
    value=grounded_plan('Привези печенье в упаковке и дай мне в руку\nУточнение пользователя: Лучше на поднос',
        intent(task_ids=['cookie_delivery'],requested_handover='supported_surface'),{'model':'test'})
    assert value['fully_supported'] is True and value['requested_handover']=='supported_surface'
    assert value['physical_ready'] is False


def test_unaccepted_real_multiroom_delivery_is_not_granted_by_language_model():
    value=grounded_plan('Привези печенье в упаковке из другой комнаты на поднос',
        intent(task_ids=['cookie_delivery'],requested_handover='supported_surface'),{'model':'test'})
    assert value['fully_supported'] is False
    assert any('между комнатами' in item for item in value['missing'])
    assert value['task_ids']==['cookie_delivery'] and value['physical_ready'] is False


def test_persistent_goal_keeps_partial_scope_and_failed_rollouts(tmp_path):
    book=LearningGoals(tmp_path)
    goal=book.create(plan(supported=False,requested_handover='human_hand',task_ids=['cookie_delivery']),{})
    goal['night_plan_id']='a'*12;book.save(goal)
    fresh=LearningGoals(tmp_path)
    task=dict(task_id='cookie_delivery',state='complete',simulation_qualified=True,stages={'validate':{'successes':10,'episodes':10}})
    result=fresh.list([{'id':'a'*12,'state':'complete','tasks':[task]}])[0]
    assert result['state']=='partial_simulation_checked' and result['physical_ready'] is False
    task['simulation_qualified']=False;task['stages']['validate']['successes']=0
    result=fresh.list([{'id':'a'*12,'state':'complete','tasks':[task]}])[0]
    assert result['state']=='needs_more_training' and result['results'][0]['successes']==0
    with pytest.raises(ValueError):fresh.get('../escape')


def test_plan_only_and_missing_parts_do_not_start_training(tmp_path):
    runtime=Runtime(FakeEnv,tmp_path)
    runtime.language.plan=Mock(return_value=(intent(task_ids=['cookie_delivery'],supported=False,
        requested_handover='human_hand',missing=['Нет передачи в ладонь']),{'model':'test'}))
    runtime.night_queue.create=Mock(side_effect=AssertionError('Must not train without the partial choice'))
    runtime.learning_cycles.create=Mock(side_effect=AssertionError('Must not train without the partial choice'))
    try:
        job=dict(id='b'*12,kind='assistant',status='running')
        runtime.run_job(job,JobRequest(request_text='Печенье в руку'))
        assert job['status']=='complete'
        assert job['result']['state']=='needs_decision'
        runtime.night_queue.create.assert_not_called()
        runtime.learning_cycles.create.assert_not_called()
        # Explicitly authorizing the available part reaches the actual trainer.
        runtime.learning_cycles.create.side_effect=ValueError('Reached training')
        selected=dict(id='c'*12,kind='assistant',status='running')
        runtime.run_job(selected,JobRequest(goal_id=job['goal_id'],allow_partial=True))
        assert selected['status']=='failed' and 'Reached training' in selected['message']
        runtime.learning_cycles.create.assert_called_once()
        runtime.night_queue.create.assert_not_called()
    finally:
        runtime.close()


def test_followup_reinterprets_original_request_without_resuming_old_goal(tmp_path):
    runtime=Runtime(FakeEnv,tmp_path)
    runtime.language.plan=Mock(return_value=(intent(),{'model':'test'}))
    try:
        initial=runtime.learning_goals.create(plan(),{})
        job=dict(id='d'*12,kind='assistant',status='running')
        runtime.run_job(job,JobRequest(goal_id=initial['id'],request_text='Корзина для белья',auto_train=False))
        assert job['status']=='complete'
        assert job['result']['parent_id']==initial['id']
        text=runtime.language.plan.call_args.args[0]
        assert initial['plan']['request'] in text and 'Уточнение пользователя: Корзина для белья' in text
        assert runtime.learning_goals.get(initial['id'])['night_plan_id'] is None
    finally:runtime.close()


def test_phase_preparation_resume_keeps_workshops_and_ignores_archived_act_state(tmp_path,monkeypatch):
    calls=[]
    def prepare(root,identifier,task,cancelled,progress):
        calls.append(identifier)
        if len(calls)==1:raise InterruptedError('operator pause')
        return dict(state='prepared',simulation_ready=False)
    monkeypatch.setattr('simulator.skill_workshop.prepare',prepare)
    runtime=Runtime(FakeEnv,tmp_path)
    try:
        goal=runtime.learning_goals.create(plan(),{})
        goal['night_plan_id']='a'*12;goal['learning_cycle_id']='b'*12
        runtime.learning_goals.save(goal)
        for index in range(2):
            job=dict(id=str(index)*12,kind='assistant',status='running')
            runtime.run_job(job,JobRequest(goal_id=goal['id'],learning_mode='skills'))
            saved=runtime.learning_goals.get(goal['id'])
            assert saved['state']==('paused' if index==0 else 'prepared')
            assert len(saved['skill_workshops'])==1
        assert calls[0]==calls[1]
        listed=runtime.learning_goals.list([dict(id='a'*12,state='paused',tasks=[])],
            [dict(id='b'*12,state='ready_for_transfer',tasks=[])])[0]
        assert listed['state']=='prepared' and listed['physical_ready'] is False
        assert listed['learning_mode']=='skills'
    finally:runtime.close()


def test_language_goal_metadata_survives_verified_import_without_execution(tmp_path):
    batch=dict(id='a'*12,tasks=[dict(task_id='sock_basket',state='failed',error='No demos',stages={})],learning_goal=plan())
    exported=export_curriculum(batch,tmp_path/'request.tar.gz')
    imported=import_curriculum(exported['path'],tmp_path/'robot',register=True)
    assert imported['automatic_execution_allowed'] is False
    paths=list((tmp_path/'robot/data/simulation-curricula').glob('*.json'))
    stored=json.loads(paths[0].read_text())['manifest']['learning_goal']
    assert stored['request']==batch['learning_goal']['request'] and stored['physical_ready'] is False
    with pytest.raises(ValueError):_validate_learning_goal(stored|{'physical_ready':True},['sock_basket'])
    with pytest.raises(ValueError):_validate_learning_goal(stored|{'requested_handover':'human_hand'},['sock_basket'])
    with pytest.raises(ValueError):_validate_learning_goal(stored,['toy_box'])


def test_new_autonomous_mode_reuses_learner_and_does_not_resume_archived_act(tmp_path,monkeypatch):
    calls=[]
    def run(self,*,cancelled,progress,resume):
        state=json.loads(self.path.read_text());calls.append((self.id,resume))
        state.update(state='operator_paused',reason='Пауза',simulation_ready=False)
        self.save(state);return state
    monkeypatch.setattr('simulator.skill_training.SkillTraining.run',run)
    runtime=Runtime(FakeEnv,tmp_path)
    try:
        goal=runtime.learning_goals.create(plan(),{})
        goal['learning_cycle_id']='b'*12;runtime.learning_goals.save(goal)
        for i in range(2):
            job=dict(id=str(i)*12,kind='assistant',status='running')
            runtime.run_job(job,JobRequest(goal_id=goal['id'],learning_mode='skill_search'))
        assert calls[0][0]==calls[1][0] and calls[0][1] is False and calls[1][1] is True
        saved=runtime.learning_goals.list([], [dict(id='b'*12,state='ready_for_transfer',tasks=[])])[0]
        assert saved['learning_mode']=='skill_search' and saved['state']=='operator_paused'
        assert not saved.get('archive') and saved['physical_ready'] is False
    finally:runtime.close()


def test_language_api_requires_local_session_and_bounds_identifiers(tmp_path):
    with client(tmp_path) as connection:
        headers=auth(connection)
        assert connection.post('/api/jobs/assistant',json={'request_text':'Носок'}).status_code==403
        assert connection.post('/api/jobs/assistant',json={'goal_id':'../escape'},headers=headers).status_code==422
        assert connection.get('/api/goals/not-valid/package').status_code==404
        assert connection.get('/api/state').json()['learning_goals']==[]


def test_goal_download_cannot_select_another_package_or_follow_external_path(tmp_path):
    from simulator.transfer import sha256
    with client(tmp_path) as connection:
        runtime=connection.app.state.runtime
        goal=runtime.learning_goals.create(plan(),{})
        package=tmp_path/'exports/my-request.tar.gz';package.write_bytes(b'test package')
        goal['archive']={'path':str(package),'sha256':sha256(package)}
        runtime.learning_goals.save(goal)
        unrelated=tmp_path/'exports/another-request.tar.gz';unrelated.write_bytes(b'unrelated')
        runtime.latest_package=unrelated
        assert connection.get('/api/goals/'+goal['id']+'/package').content==b'test package'
        package.write_bytes(b'changed')
        assert connection.get('/api/goals/'+goal['id']+'/package').status_code==404
        outside=tmp_path/'outside.tar.gz';outside.write_bytes(b'outside')
        goal['archive']={'path':str(outside),'sha256':sha256(outside)}
        runtime.learning_goals.save(goal)
        assert connection.get('/api/goals/'+goal['id']+'/package').status_code==404
