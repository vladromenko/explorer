import json
import time
import threading

from test_simulator_server import client,auth


def wait(connection,identifier):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        job=next(j for j in connection.get('/api/state').json()['jobs'] if j['id']==identifier)
        if job['status']!='running':return job
        time.sleep(.01)
    raise AssertionError('Job did not finish')


def test_short_skill_preparation_does_not_launch_act_or_export(tmp_path,monkeypatch):
    calls=[]
    def prepare(root,identifier,task,cancelled,progress):
        calls.append((identifier,task))
        folder=root/'skill-workshops'/identifier;folder.mkdir(parents=True)
        state=dict(id=identifier,task_id=task,created=time.time(),state='prepared',simulation_ready=False,physical_ready=False)
        (folder/'state.json').write_text(json.dumps(state))
        return state
    monkeypatch.setattr('simulator.skill_workshop.prepare',prepare)
    with client(tmp_path) as connection:
        response=connection.post('/api/jobs/skills',json={'task_ids':['toy_box']},headers=auth(connection))
        assert response.status_code==200
        job=wait(connection,response.json()['id'])
        assert job['status']=='complete'
        assert job['result']['simulation_ready'] is False
        assert job['result']['automatic_execution_allowed'] is False
        assert 'ещё не принято' in job['message']
        state=connection.get('/api/state?compact=true').json()
        assert state['skill_workshops'][0]['task_id']=='toy_box'
        assert not state['learning_cycles'] and not list((tmp_path/'exports').glob('*.tar.gz'))
        assert len(calls)==1 and all(j['kind']=='skills' for j in state['jobs'])


def test_diagnostic_trial_does_not_report_task_success_as_job_completion(tmp_path,monkeypatch):
    seen=[]
    def trial(env,root,identifier,task,seed,profile,cancelled,progress):
        seen.append((task,seed,profile))
        return dict(state='diagnosed',success=False,reason='Удержание не видно',simulation_ready=False)
    monkeypatch.setattr('simulator.skill_workshop.trial',trial)
    with client(tmp_path) as connection:
        response=connection.post('/api/jobs/skill_trial',json={'task_id':'sock_basket','seed':123,'scene_profile':'sock_variants_v1'},headers=auth(connection))
        job=wait(connection,response.json()['id'])
        assert job['status']=='complete' and job['result']['success'] is False
        assert 'Удержание не видно' in job['message'] and job['message']!='Готово'
        assert seen==[('sock_basket',123,'sock_variants_v1')]


def test_resume_preserves_workshop_identity(tmp_path,monkeypatch):
    seen=[]
    def prepare(root,identifier,*args):
        seen.append(identifier)
        return dict(state='prepared',simulation_ready=False)
    monkeypatch.setattr('simulator.skill_workshop.prepare',prepare)
    with client(tmp_path) as connection:
        response=connection.post('/api/jobs/skills',json={'task_ids':['sock_basket'],'workshop_id':'a'*12},headers=auth(connection))
        assert wait(connection,response.json()['id'])['status']=='complete'
        assert seen==['a'*12]


def test_autonomous_skill_learner_starts_without_act_or_robot_commands(tmp_path,monkeypatch):
    calls=[]
    def run(self,*,cancelled,progress,resume):
        saved=json.loads(self.path.read_text());calls.append((self.id,resume,saved['options']))
        progress(dict(saved,completed_attempts=8,active_workers=[]))
        saved.update(state='needs_diagnosis',reason='Нет улучшения',completed_attempts=8)
        self.save(saved);return saved
    monkeypatch.setattr('simulator.skill_training.SkillTraining.run',run)
    with client(tmp_path) as connection:
        response=connection.post('/api/jobs/skill_train',json=dict(task_ids=['sock_basket'],count=100,workers=4,hours=24),headers=auth(connection))
        assert response.status_code==200
        job=wait(connection,response.json()['id'])
        assert job['status']=='complete' and job['result']['state']=='needs_diagnosis'
        assert job['message']!='Готово' and not job['result']['simulation_ready']
        state=connection.get('/api/state').json()
        assert state['skill_trainings'][0]['completed_attempts']==8
        assert not state['learning_cycles'] and all(j['kind']=='skill_train' for j in state['jobs'])
        assert calls[0][1] is False and calls[0][2]['workers']==4


def test_stop_interrupts_skill_trial_but_focus_loss_does_not(tmp_path,monkeypatch):
    entered=threading.Event()
    def trial(env,root,identifier,task,seed,profile,cancelled,progress):
        entered.set()
        deadline=time.monotonic()+2
        while not cancelled() and time.monotonic()<deadline:time.sleep(.005)
        assert cancelled()
        return dict(state='paused',reason='Оператор остановил проверку',simulation_ready=False)
    monkeypatch.setattr('simulator.skill_workshop.trial',trial)
    with client(tmp_path) as connection:
        headers=auth(connection)
        response=connection.post('/api/jobs/skill_trial',json={'task_id':'sock_basket'},headers=headers)
        assert entered.wait(1)
        assert connection.post('/api/stop-manual',json={},headers=headers).status_code==200
        assert not connection.app.state.runtime.cancel_job.is_set()
        assert connection.post('/api/pause',json={},headers=headers).status_code==409
        assert connection.post('/api/reset',json={},headers=headers).status_code==409
        assert connection.post('/api/record/start',json={},headers=headers).status_code==409
        assert connection.post('/api/stop',json={},headers=headers).status_code==200
        job=wait(connection,response.json()['id'])
        assert job['status']=='cancelled' and job['result']['state']=='paused'
