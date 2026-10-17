"""Durable simulation-only improvement cycles with independent final tests."""
import copy
import json
from pathlib import Path
import secrets
import time

from .curriculum import report_passes, task_spec
from .learning import write_json


class LearningCycles:
    def __init__(self,root):
        self.root=Path(root);self.folder=self.root/'learning-cycles';self.folder.mkdir(parents=True,exist_ok=True)
        for path in self.folder.glob('*.json'):
            try:
                value=json.loads(path.read_text())
                if value['state']=='running':
                    value.update(state='interrupted',reason='Приложение перезапущено; сохранение можно продолжить')
                    self.save(value)
            except (OSError,ValueError,KeyError,TypeError):pass

    def save(self,value):
        value['updated']=time.time();write_json(self.folder/(value['id']+'.json'),value)

    def get(self,identifier):
        if not isinstance(identifier,str) or len(identifier)!=12 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid learning cycle identity')
        return json.loads((self.folder/(identifier+'.json')).read_text())

    def list(self):
        output=[]
        for path in self.folder.glob('*.json'):
            try:output.append(json.loads(path.read_text()))
            except (ValueError,OSError):output.append(dict(id=path.stem,state='corrupt',tasks=[],created=path.stat().st_mtime))
        return sorted(output,key=lambda value:value.get('created',0),reverse=True)

    def create(self,tasks,count=100,hours=24,workers=4,scene_profile='sock_variants_v1',models=None,goal_id=None,steps=30000,
               teacher_profile='planned_release_v4'):
        if not tasks or len(set(tasks))!=len(tasks) or not 2<=count<=1000 or not 0<hours<=24 or workers not in (1,4,8):
            raise ValueError('Invalid automatic improvement parameters')
        if scene_profile not in ('baseline','sock_variants_v1'):raise ValueError('Unknown scene profile')
        if teacher_profile not in ('legacy_v1','planned_grasp_v3','planned_release_v4'):
            raise ValueError('Unknown teacher profile')
        if type(steps) is not int or not 1<=steps<=200000:raise ValueError('Invalid training step budget')
        for task in tasks:task_spec(task)
        value=dict(id=secrets.token_hex(6),created=time.time(),state='queued',hours=hours,
            options=dict(count=count,workers=workers,scene_profile=scene_profile,steps=steps,
                         teacher_profile=teacher_profile),goal_id=goal_id,
            tasks=[dict(task_id=task,state='queued',iteration=1,model_id=(models or {}).get(task),
                        reference_model_id=(models or {}).get(task),stages={},history=[]) for task in tasks],
            seed_base=600000000+secrets.randbelow(600000000),physical_ready=False,automatic_hardware_execution=False)
        self.save(value);return value

    def run(self,identifier,execute,cancelled,progress,cancellation_context=None):
        value=self.get(identifier)
        if value['state']=='ready_for_transfer':return value
        if value['state']=='needs_diagnosis':return value
        deadline=time.monotonic()+value['hours']*3600
        def stopped():return cancelled() or time.monotonic()>=deadline
        def check():
            if stopped():raise InterruptedError('Пауза по запросу пользователя' if cancelled() else 'Бюджет времени исчерпан; продолжите сохранённый цикл')
        def stage(task,name,options):
            if name in task['stages']:return task['stages'][name]
            check();value.update(phase=name,task_id=task['task_id']);task['state']='running'
            pending=task.get('pending')
            if not pending or pending.get('stage')!=name:
                pending=dict(stage=name,job_id=secrets.token_hex(6));task['pending']=pending
            if name in ('train','trials'):
                partial=self.root/('runs' if name=='train' else 'parallel-jobs')/pending['job_id']
                if partial.exists() and (name=='trials' or not (partial/'model/checkpoints/last').exists()):
                    pending['job_id']=secrets.token_hex(6)
            self.save(value);progress(value)
            result=execute(name,dict(options,job_id=pending['job_id'],task_id=task['task_id']),stopped)
            task['stages'][name]=result;task.pop('pending',None);self.save(value)
            return result
        value.pop('maintenance_token',None)
        value.update(state='running',reason=None,pause_kind=None);self.save(value)
        try:
            for index,task in enumerate(value['tasks']):
                while task['state']!='ready_for_transfer':
                    check()
                    seed=(value['seed_base']+index*10000000+task['iteration']*10000)%(2**31-100000)
                    options=dict(value['options'],seed=seed,batch_size=32,minimum_epochs=8)
                    corrective_generation=False
                    if task.get('model_id'):
                        reference=task.get('reference_model_id')
                        if reference and reference!=task['model_id']:
                            comparison=stage(task,'compare',dict(options,model_id=task['model_id'],
                                reference_model_id=reference,validation_seed=value['seed_base']+100))
                            task['last_comparison']=comparison
                            if not comparison['improved']:
                                if (value['options'].get('teacher_profile') in ('planned_grasp_v3','planned_release_v4')
                                    and task.get('corrective_teacher_evidence_sha256')
                                    and not task.get('corrective_round_used')):
                                    task['rejected_model_id']=task['model_id']
                                    task['model_id']=reference
                                    task['corrective_round_used']=True
                                    corrective_generation=True
                                    self.save(value)
                                else:
                                    value.update(state='needs_diagnosis',phase='diagnosis',
                                        reason='Новая модель не улучшила доставку на тех же диагностических сценах. Автоматический повтор обучения остановлен для разбора причин.')
                                    self.save(value);return value
                            else:task['reference_model_id']=task['model_id']
                        if not corrective_generation:
                            tuning=stage(task,'audit',dict(options,model_id=task['model_id'],validation_seed=seed+1000))
                            model=tuning['model_id']
                            report=stage(task,'validate',dict(options,model_id=model,count=10,validation_seed=seed+2000))
                            if report_passes(report):
                                final=stage(task,'final_validate',dict(options,model_id=model,count=10,validation_seed=seed+3000))
                            else:final=None
                            if final and report_passes(final):
                                package=stage(task,'export',dict(options,model_id=model))
                                task.update(state='ready_for_transfer',simulation_qualified=True,model_id=model,archive=package,
                                            validation=final,physical_ready=False)
                                task['history'].append(copy.deepcopy(task['stages']));self.save(value)
                            else:
                                failed=final or report
                                failed_seeds=[]
                                for position,row in enumerate(failed.get('results',[])):
                                    seed_value=row.get('domain_variation',{}).get('seed')
                                    if seed_value is None and position<len(failed.get('seeds',[])):seed_value=failed['seeds'][position]
                                    if row.get('success') is False and type(seed_value) is int:failed_seeds.append(seed_value)
                                task['failure_seeds']=failed_seeds
                    if task['state']!='ready_for_transfer':
                        # Previously failed test scenes become teaching data, never final test evidence.
                        failed=list(dict.fromkeys(task.get('failure_seeds',[])))[:options['count']]
                        fresh=[seed+4000+i for i in range(options['count']) if seed+4000+i not in failed]
                        scene_seeds=(failed+fresh)[:options['count']]
                        stage(task,'trials',dict(options,scene_seeds=scene_seeds))
                        trained=stage(task,'train',dict(options,warm_start_model_id=task.get('reference_model_id') or task.get('model_id')))
                        if not task.get('reference_model_id'):
                            task['reference_model_id']=task.get('model_id')
                        task['history'].append(copy.deepcopy(task['stages']))
                        task.update(model_id=trained['model_id'],iteration=task['iteration']+1,stages={},state='running')
                        self.save(value)
            value.update(state='ready_for_transfer',phase='ready',simulation_qualified=True,physical_ready=False)
        except InterruptedError as exc:
            context=cancellation_context() if cancelled() and cancellation_context else {}
            maintenance=context.get('kind')=='maintenance' and context.get('cycle_id')==identifier and context.get('token')
            value.update(state='paused',reason=str(exc),pause_kind='maintenance' if maintenance else 'operator' if cancelled() else 'budget_or_capacity')
            if maintenance:value['maintenance_token']=context['token']
        except Exception as exc:
            value.update(state='failed',reason=str(exc))
        finally:self.save(value);progress(value)
        return value
