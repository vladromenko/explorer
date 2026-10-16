"""One durable natural-language queue, with explicit simulator ownership.

Planning, scene generation, parameter learning and independent tests are distinct
persisted stages. Neither a language plan nor a finished budget grants readiness.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import urllib.request

from .learning import write_json,sha256
from .language_learning import LocalLanguageModel,grounded_plan
from .scene_recipes import compile_recipes,validate_recipe,explicit_scene_requests
from .skill_training import SkillTraining,identity,list_trainings

ACTIVE={'queued','planning','waiting_for_current','building_scene','training','validating','waiting_for_memory','packaging'}
PAUSED={'operator_paused','budget_paused','memory_paused','interrupted'}
TERMINAL={'needs_clarification','needs_diagnosis','validation_failed','simulation_qualified','failed'}


def legacy_get(route):
    with urllib.request.urlopen('http://127.0.0.1:8766'+route,timeout=2) as response:return json.load(response)


def legacy_post(route,body):
    token=legacy_get('/api/session')['token']
    request=urllib.request.Request('http://127.0.0.1:8766'+route,data=json.dumps(body).encode(),method='POST',
        headers={'Content-Type':'application/json','X-Explorer-Sim-Token':token})
    with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)


class AutomaticLearning:
    def __init__(self,root,*,planner=None,trainer_factory=SkillTraining,scene_builder=None):
        self.root=Path(root);self.folder=self.root/'automatic-requests';self.folder.mkdir(parents=True,exist_ok=True)
        self.planner=planner or LocalLanguageModel(self.root);self.trainer_factory=trainer_factory
        self.scene_builder=scene_builder or self.build_scene
        self.stop=threading.Event();self.thread=None;self.active_id=None;self.mutex=threading.RLock()

    def get(self,identifier):
        identity(identifier);value=json.loads((self.folder/(identifier+'.json')).read_text())
        if value.get('id')!=identifier:raise ValueError('Request identity mismatch')
        return value

    def save(self,value):
        if self.root.parent==Path('/Volumes/ExplorerSim') and not self.root.parent.is_mount():
            raise OSError('Диск ExplorerSim отключён; новые файлы не создаются')
        with self.mutex:
            path=self.folder/(identity(value['id'])+'.json')
            try:previous=json.loads(path.read_text())
            except FileNotFoundError:previous={}
            events={}
            for event in previous.get('events',[])+value.get('events',[]):
                events[(event['at'],event['stage'],event['message'])]=event
            value['events']=sorted(events.values(),key=lambda event:event['at'])
            if (self.folder/(value['id']+'.pause')).exists() and value['state'] in ACTIVE|{'running'}:
                value.update(state='pausing',reason='Останавливаю работников; записи сохраняются')
            value['updated']=time.time();write_json(path,value)

    def list(self):
        values=[]
        for path in self.folder.glob('*.json'):
            try:values.append(self.get(path.stem))
            except (OSError,ValueError,KeyError):continue
        return sorted(values,key=lambda r:r['created'],reverse=True)

    def create(self,text,*,attempts=100,hours=24.,parent_id=None):
        if not isinstance(text,str) or not text.strip() or len(text)>4000:raise ValueError('Напишите задачу, до 4000 символов')
        if type(attempts) is not int or not 8<=attempts<=1000 or not 0<hours<=24:raise ValueError('Некорректный бюджет')
        if parent_id:
            parent=self.get(parent_id)
            if parent['state'] not in ('needs_clarification','failed','needs_diagnosis','validation_failed'):
                raise ValueError('Уточнение создаётся для остановленного или неподдерживаемого плана')
            text=parent['text']+'\nУточнение пользователя: '+text
            if len(text)>4000:raise ValueError('Опишите полную задачу короче, до 4000 символов')
        value=dict(schema=1,id=secrets.token_hex(6),text=text.strip(),created=time.time(),state='queued',
            parent_id=parent_id,budget=dict(attempts=attempts,hours=float(hours)),tasks=[],events=[],
            simulation_ready=False,physical_ready=False,automatic_hardware_execution=False,
            reason='Просьба сохранена в очереди',elapsed_s=0.)
        with self.mutex:self.save(value)
        return value

    def event(self,value,stage,message):
        value['state']=stage;value['reason']=message
        value['events'].append(dict(at=time.time(),stage=stage,message=message));self.save(value)

    def pause_queue(self,owner_id=None):
        write_json(self.folder/'queue-pause.json',dict(paused=True,owner_id=owner_id,at=time.time()))

    def pause_info(self):
        path=self.folder/'queue-pause.json'
        try:return json.loads(path.read_text())
        except FileNotFoundError:return None

    def resume_queue(self):
        (self.folder/'queue-pause.json').unlink(missing_ok=True)

    def pause(self,identifier):
        with self.mutex:
            value=self.get(identifier)
            if value['state']=='simulation_qualified':return value
            self.pause_queue(identifier)
            (self.folder/(identifier+'.pause')).touch(exist_ok=True)
            self.event(value,'pausing' if self.active_id==identifier else 'operator_paused','Останавливаю работников; записи сохраняются')
        return value

    def resume(self,identifier):
        with self.mutex:
            value=self.get(identifier)
            if value['state'] not in PAUSED:raise ValueError('Эту задачу нельзя продолжить без разбора причины')
            if self.active_id==identifier:raise ValueError('Дождитесь полной остановки работников')
            if value['state']=='budget_paused':
                for task in value['tasks']:
                    trainer=self.trainer_factory(self.root,task['training_id'])
                    if trainer.path.is_file():
                        saved=json.loads(trainer.path.read_text())
                        if saved['state']=='budget_paused':
                            target=min(1000,saved['completed_attempts']+value['budget']['attempts'])
                            if target<=saved['completed_attempts']:raise ValueError('Достигнут лимит 1000 попыток; нужен разбор причины')
                            saved['options']['attempts']=max(saved['options']['attempts'],target);trainer.save(saved)
            # A fresh time allowance resumes the same tasks/checkpoints.
            (self.folder/(identifier+'.pause')).unlink(missing_ok=True)
            value['elapsed_s']=0.;value.pop('budget_exhausted',None);self.event(value,'queued','Продолжение сохранённой задачи');self.resume_queue()
        return value

    def cancelled(self,identifier):
        if self.stop.is_set() or not self.root.is_dir():return True
        return (self.folder/(identity(identifier)+'.pause')).exists() or self.get(identifier)['state'] in ('operator_paused','pausing')

    def simulator_busy(self):
        folder=self.root/'skill-training';folder.mkdir(exist_ok=True)
        with (folder/'owner.lock').open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
        return False

    def model_status(self):
        lock=Path(__file__).resolve().parents[1]/'config/grounding-model.json'
        try:
            expected=json.loads(lock.read_text());installed=json.loads((self.root/'models/grounding-dino-tiny/verified-install.json').read_text())
            vision=installed['revision']==expected['revision'] and installed['files']==expected['files']
            vision=vision and all((self.root/'models/grounding-dino-tiny'/name).is_file() for name in expected['files'])
        except (OSError,ValueError,KeyError):vision=False
        language=(self.root/'llm-models/manifests/registry.ollama.ai/library/qwen3.5/4b').is_file()
        from .semantic_vision import vision_python
        python=vision_python(self.root)
        return dict(vision=vision,language=language,vision_runtime=python.is_file(),
            ready=vision and language and python.is_file(),planner='Qwen3.5 4B',perception='GroundingDINO tiny',
            learner='Observed skill parameter policy search',universal_policy=False)

    def ensure_models(self,value):
        if self.model_status()['ready']:return
        self.event(value,'planning','Автоматическая установка недостающих моделей')
        script=Path(__file__).resolve().parents[1]/'bin/prepare-automatic-models.py'
        self.run_process([sys.executable,str(script)],self.folder/(value['id']+'-models.log'),value['id'],timeout=1800)
        if not self.model_status()['ready']:raise ValueError('Модели не прошли проверку установки')

    def run_process(self,args,log_path,identifier,*,timeout):
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('a') as log:
            p=subprocess.Popen(args,stdout=log,stderr=log);deadline=time.monotonic()+timeout
            try:
                while p.poll() is None:
                    if self.cancelled(identifier):raise InterruptedError('Остановлено')
                    if time.monotonic()>deadline:raise TimeoutError('Превышен конечный срок этапа; журнал: '+str(log_path))
                    time.sleep(.2)
                if p.returncode:raise ValueError('Этап завершился ошибкой; журнал: '+str(log_path))
            finally:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=3)
                    except subprocess.TimeoutExpired:p.kill();p.wait()

    def build_scene(self,value,task):
        folder=self.root/'automatic-scenes'/value['id']/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'recipe.json',task['recipe'])
        script=Path(__file__).resolve().parents[1]/'bin/build-automatic-scene.py'
        self.run_process([sys.executable,str(script),str(folder)],folder/'build.log',value['id'],timeout=90)
        manifest=json.loads((folder/'scene.json').read_text())
        if manifest['recipe_sha256']!=task['recipe']['sha256']:raise ValueError('Scene does not match requested recipe')
        return dict(folder=str(folder),manifest=manifest,preview=str(folder/'preview.jpg'))

    def cached_skill(self,recipe):
        from .skill_workshop import execution_fingerprint
        fingerprint=execution_fingerprint()['sha256']
        from .curriculum import report_passes
        for row in list_trainings(self.root):
            validation=row.get('validation',[])
            if len(validation)<2 or not all(report_passes(group) for group in validation):continue
            seeds=[r.get('seed') for group in validation for r in group.get('results',[])]
            training={r['seed'] for r in row.get('episodes',[]) if r.get('split','train')=='train'}
            if len(seeds)!=len(set(seeds)) or training.intersection(seeds):continue
            if (row.get('state')=='simulation_qualified' and row.get('simulation_ready') is True and
                (row.get('scene_recipe') or {}).get('sha256')==recipe['sha256'] and
                row.get('controller',{}).get('sha256')==fingerprint):
                artifact=row.get('archive') or {};path=Path(artifact.get('path','')).resolve()
                if artifact.get('verified') and path.is_relative_to(self.root/'exports') and path.is_file() and sha256(path)==artifact.get('sha256'):
                    return row
        return None

    def advance(self,identifier):
        value=self.get(identifier)
        if value['state'] in PAUSED|TERMINAL:return value
        started=time.monotonic();elapsed=value.get('elapsed_s',0.)
        def budget_cancelled():
            if elapsed+time.monotonic()-started>=value['budget']['hours']*3600:
                value['budget_exhausted']=True;return True
            return self.cancelled(identifier)
        try:
            self.ensure_models(value)
            if 'plan' not in value:
                from .resource_budget import memory_budget
                memory=memory_budget()
                if memory['available_bytes']<memory['total_bytes']*.15+3*2**30:
                    self.event(value,'waiting_for_memory','Ожидаю память для локального помощника');return value
                self.event(value,'planning','Разбираю просьбу и проверяю доступные действия')
                try:
                    parsed,provenance=self.planner.plan(value['text'],simulation_only=True)
                    if budget_cancelled():raise InterruptedError('Планирование остановлено')
                    plan=grounded_plan(value['text'],parsed,provenance,simulation_only=True)
                    plan['scene_requests']=explicit_scene_requests(value['text'],plan['task_ids'],plan.get('scene_requests',[]))
                    value['plan']=plan;compiled=compile_recipes(plan,plan['scene_requests']);value['compiled']=compiled
                except ValueError as error:
                    self.event(value,'needs_clarification','Не удалось однозначно задать сцену: '+str(error));return value
                if compiled['state']!='compiled':
                    self.event(value,'needs_clarification',compiled['question']);return value
                value['tasks']=[]
                for recipe in compiled['recipes']:
                    cached=self.cached_skill(recipe)
                    task=dict(task_id=recipe['task_id'],recipe=recipe,state='queued',training_id=secrets.token_hex(6))
                    if cached:task.update(state='simulation_qualified',training_id=cached['id'],reused=True,
                        result=dict(simulation_ready=True,archive=cached['archive'],reason='Использован совместимый проверенный навык'))
                    value['tasks'].append(task)
                self.save(value)
            if self.pause_info():
                self.event(value,'waiting_for_current','Общая очередь на паузе');return value
            if self.simulator_busy():
                if value['state']!='waiting_for_current':self.event(value,'waiting_for_current','Текущая очередь продолжает работу; эта просьба ждёт своей очереди')
                return value
            for task in value['tasks']:
                if self.pause_info():
                    self.event(value,'waiting_for_current','Общая очередь на паузе');return value
                if task['state']=='simulation_qualified':continue
                if budget_cancelled():raise InterruptedError('Пауза')
                if 'scene' not in task:
                    self.event(value,'building_scene','Создаю физическую сцену: '+task['recipe']['title'])
                    # Claim the same physics lock as both learner generations.
                    folder=self.root/'skill-training';folder.mkdir(exist_ok=True)
                    with (folder/'owner.lock').open('a') as owner:
                        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:
                            self.event(value,'waiting_for_current','Симулятор занят');return value
                        task['scene']=self.scene_builder(value,task)
                    self.save(value)
                trainer=self.trainer_factory(self.root,task['training_id'])
                remaining=max(.001,value['budget']['hours']-(elapsed+time.monotonic()-started)/3600)
                saved=trainer.create(attempts=value['budget']['attempts'],hours=min(24.,remaining),workers=4,
                    seed=int(value['id'],16)%1_000_000_000,task_id=task['task_id'],scene_recipe=task['recipe'])
                if saved['state'] in ('needs_diagnosis','validation_failed','failed'):
                    task['state']=saved['state'];task['result']=dict(simulation_ready=False,reason=saved.get('reason') or saved['state']);self.save(value);continue
                self.event(value,'training','Обучаю: '+task['recipe']['title'])
                def progress(state):
                    if budget_cancelled():raise InterruptedError('Пауза')
                    task['state']=state['state'];task['progress']=dict(completed=state['completed_attempts'],
                        total=state['options']['attempts'],generation=state['generation'],phase=state['phase_group'],
                        active_workers=state.get('active_workers',[]))
                    value['state']='validating' if state['state'] in ('validating','packaging') else state['state']
                    value['reason']=state.get('reason') or ('Обучаю: '+task['recipe']['title']);value['elapsed_s']=elapsed+time.monotonic()-started;self.save(value)
                saved['options']['hours']=min(24.,remaining);trainer.save(saved)
                result=trainer.run(cancelled=budget_cancelled,progress=progress,resume=True)
                task.update(state=result['state'],result=dict(simulation_ready=result['simulation_ready'],
                    archive=result.get('archive'),reason=result.get('reason')))
                if result['state']!='simulation_qualified':
                    reasons=Counter(r.get('reason') for r in result.get('episodes',[]) if r.get('status')=='complete' and r.get('success') is not True and r.get('reason'))
                    task['result']['failure_causes']=[dict(reason=reason,count=count) for reason,count in reasons.most_common(3)]
                    if result['state'] in ('needs_diagnosis','validation_failed','failed'):
                        self.save(value);continue
                    state='budget_paused' if value.get('budget_exhausted') else result['state']
                    self.event(value,state,result.get('reason') or state);return value
                self.save(value)
            failed=[task for task in value['tasks'] if task['state']!='simulation_qualified']
            if failed:
                self.event(value,'needs_diagnosis','Не все навыки прошли проверку: '+'; '.join(task['recipe']['title']+' — '+str(task.get('result',{}).get('reason',task['state'])) for task in failed));return value
            self.event(value,'packaging','Проверяю итоговый пакет всех навыков')
            value['package']=self.package(value);value['simulation_ready']=True
            self.event(value,'simulation_qualified','Все запрошенные задачи прошли независимые проверки в симуляции')
        except InterruptedError:
            self.event(value,'budget_paused' if value.get('budget_exhausted') else 'operator_paused','Вычисления остановлены; прогресс сохранён')
        except Exception as error:self.event(value,'operator_paused' if self.cancelled(identifier) else 'failed',str(error))
        finally:
            value['elapsed_s']=elapsed+time.monotonic()-started;self.save(value)
        return value

    def package(self,value):
        import tarfile
        folder=self.root/'automatic-results'/value['id'];folder.mkdir(parents=True,exist_ok=True)
        manifest=dict(schema=1,request=value['text'],plan=value['plan'],tasks=[],simulation_ready=True,
            physical_ready=False,automatic_hardware_execution=False,kind='observed_simulation_curriculum')
        sources={}
        for task in value['tasks']:
            artifact=task['result']['archive'];path=Path(artifact['path']).resolve()
            if (task['state']!='simulation_qualified' or not path.is_relative_to(self.root/'exports') or
                not artifact.get('verified') or sha256(path)!=artifact['sha256']):raise ValueError('Child package not qualified or changed')
            name=task['task_id']+'.tar.gz';sources[name]=path
            manifest['tasks'].append(dict(task_id=task['task_id'],file=name,sha256=artifact['sha256'],recipe=task['recipe']))
        write_json(folder/'manifest.json',manifest);sources['manifest.json']=folder/'manifest.json'
        hashes={n:sha256(p) for n,p in sources.items()}
        output=self.root/'exports'/('automatic-'+value['id']+'.tar.gz');temporary=output.with_suffix('.tmp')
        with tarfile.open(temporary,'w:gz') as archive:
            for name,path in sources.items():archive.add(path,arcname=name,recursive=False)
        seen=set()
        with tarfile.open(temporary,'r:gz') as archive:
            for member in archive:
                if member.name in seen or member.name not in hashes or not member.isfile():raise ValueError('Invalid package member')
                if hashlib.file_digest(archive.extractfile(member),'sha256').hexdigest()!=hashes[member.name]:raise ValueError('Package checksum mismatch')
                seen.add(member.name)
        if seen!=set(hashes):raise ValueError('Package incomplete')
        temporary.replace(output)
        return dict(path=str(output),sha256=sha256(output),verified=True,physical_ready=False)

    def recover_requests(self):
        """Called only after claiming the scheduler lease on a fresh boot."""
        for row in self.list():
            if row['state'] not in ('planning','building_scene','training','running','validating','packaging','waiting_for_memory','pausing'):continue
            state='operator_paused' if row['state']=='pausing' else 'interrupted'
            for task in row.get('tasks',[]):
                if task.get('progress'):task['progress']['active_workers']=[]
                identifier=task.get('training_id')
                if identifier:
                    folder=self.root/'skill-training'/identity(identifier)
                    if folder.is_dir():(folder/'pause.request').touch(exist_ok=True)
            self.event(row,state,'Служба была остановлена; продолжите сохранённую задачу без нового поиска')

    def start(self):
        if self.thread and self.thread.is_alive():return
        self.thread=threading.Thread(target=self.work,name='automatic-learning',daemon=True);self.thread.start()

    def work(self):
        owner=(self.folder/'scheduler.lock').open('a')
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:owner.close();return
        try:
            self.recover_requests()
            while not self.stop.is_set():
                if self.pause_info():self.stop.wait(.5);continue
                pending=[v for v in reversed(self.list()) if v['state'] in ACTIVE]
                if pending:
                    self.active_id=pending[0]['id'];self.advance(self.active_id);self.active_id=None
                self.stop.wait(1.)
        finally:fcntl.flock(owner,fcntl.LOCK_UN);owner.close()

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=10)
        self.planner.close()
