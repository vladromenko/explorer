"""Versioned task curricula and durable, restartable simulation-only batches."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import secrets
import time

TASKS = {
    'sock_basket': dict(title='Носок → корзина', object_kind='sock', query='sock',
        destination='laundry', description='Найти, поднять ткань и отпустить в корзину.',
        scene='sock', rooms=False, destination_label='Корзина для белья'),
    'cookie_delivery': dict(title='Печенье в упаковке → поднос', object_kind='cookie', query='wrapped cookie',
        destination='user_tray', description='Проехать через дверь, взять упаковку и привезти на поднос возле человека.',
        scene='cookie', rooms=True, destination_label='Поднос рядом со мной'),
    'toy_box': dict(title='Лёгкий кубик → коробка', object_kind='cube', query='toy block',
        destination='toy_box', description='Поднять небольшой лёгкий кубик и убрать в коробку.',
        scene='cube', rooms=False, destination_label='Коробка для игрушек'),
    'cloth_basket': dict(title='Небольшая ткань → бельё', object_kind='cloth', query='small cloth',
        destination='laundry', description='Учиться обращению с мягкой тканью другого размера.',
        scene='cloth', rooms=False, destination_label='Корзина для белья'),
}


def task_spec(identifier):
    if identifier not in TASKS:
        raise ValueError('Неизвестная учебная задача: ' + str(identifier))
    return dict(id=identifier, **copy.deepcopy(TASKS[identifier]),
        recipe=['search', 'navigate', 'inspect', 'align', 'grasp', 'verify_hold',
                'transport', 'support', 'release', 'verify_place'],
        physical_acceptance=False, handover='supported_tray_only',
        contact_pad_model='estimated_flat_box' if TASKS[identifier]['scene'] in ('cookie','cube') else 'estimated_capsule',
        dynamics='estimated', transfer_requires_real_map_and_observed_validation=True)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def report_passes(report):
    passed = (report.get('heldout_independent') is True and report.get('episodes', 0) >= 10
              and report.get('successes') == report.get('episodes'))
    if report.get('scene_profile') == 'sock_variants_v1':
        from .domain_randomization import scene_coverage
        coverage = scene_coverage(report.get('results', []))
        passed = (passed and coverage['all_height_categories_present']
                  and coverage['distinct_seeds'] == report['episodes']
                  and coverage['user_basket_25cm_diameter_30cm_rim_episodes'] > 0)
    return passed


class NightQueue:
    """A single persisted batch; completed stages are immutable on resume.

    No automatic execution on application restart. An interrupted training stage
    is retried in a fresh run; failed datasets and logs remain available.
    """
    def __init__(self, root):
        self.folder = Path(root) / 'night-batches'
        self.folder.mkdir(parents=True, exist_ok=True)
        for path in self.folder.glob('*.json'):
            try:
                value = json.loads(path.read_text())
                if value.get('state') == 'running':
                    value.update(state='interrupted', reason='Приложение перезапущено', updated=time.time())
                    save_json(path, value)
            except (OSError,ValueError,AttributeError):
                pass

    def get(self, identifier):
        if not isinstance(identifier, str) or len(identifier) != 12 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Некорректный номер ночного задания')
        return json.loads((self.folder / (identifier + '.json')).read_text())

    def list(self):
        output=[]
        for path in self.folder.glob('*.json'):
            try:
                value=json.loads(path.read_text())
                if not isinstance(value,dict) or not {'id','created','tasks','state'}<=value.keys():
                    raise ValueError('Повреждён формат очереди')
                output.append(value)
            except (OSError,ValueError):
                output.append(dict(id=path.stem,created=path.stat().st_mtime,state='corrupt',tasks=[],reason='Повреждён файл очереди; данные сохранены: '+str(path)))
        return sorted(output,key=lambda item:item['created'],reverse=True)

    def create(self, tasks, count, steps, batch_size, hours, seed, rounds=3, scene_profile='baseline',workers=1):
        if scene_profile not in ('baseline', 'sock_variants_v1'):
            raise ValueError('Неизвестный профиль сцен')
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError('Выберите разные задачи для обучения')
        for task in tasks:
            task_spec(task)
        if not 2 <= count <= 1000 or not 1 <= steps <= 200000 or not 8 <= batch_size <= 32 or not 0 < hours <= 24 or not 1<=rounds<=5:
            raise ValueError('Параметры ночного задания вне диапазона')
        if type(workers) is not int or workers not in (1,4,8):
            raise ValueError('Выберите один, четыре или восемь исполнителей')
        value = dict(id=secrets.token_hex(6), created=time.time(), state='queued', hours=hours,
                     options=dict(count=count, steps=steps, batch_size=batch_size, seed=seed, rounds=rounds, scene_profile=scene_profile,workers=workers),
                     tasks=[dict(task_id=task, stages={}, history=[], round=1, state='queued') for task in tasks],
                     automatic_hardware_execution=False)
        save_json(self.folder / (value['id']+'.json'), value)
        return value

    def run(self, identifier, execute, cancelled, progress):
        value = self.get(identifier)
        if value['state'] == 'complete':
            return value
        deadline = time.monotonic() + value['hours']*3600
        def stopped():
            return cancelled() or time.monotonic() >= deadline
        def save():
            value['updated'] = time.time()
            save_json(self.folder / (identifier+'.json'), value)
        value.update(state='running', reason=None)
        save()
        try:
            for index, task in enumerate(value['tasks']):
                if task['state'] != 'complete':
                    task.update(state='running', error=None)
                    while task['round']<=value['options'].get('rounds',1) and task['state']=='running' and not task.get('training_complete'):
                        for stage in ('trials', 'train', 'validate', 'export'):
                            if stage not in task['stages']:
                                if stopped():
                                    raise InterruptedError('Лимит времени или остановка; очередь можно продолжить')
                                progress(f"{task_spec(task['task_id'])['title']} · раунд {task['round']} · {stage}")
                                options=dict(value['options'],seed=(value['options']['seed']+index*100000+task['round']*10000)%(2**31-1001))
                                options.update(count=min(1000,options['count']*task['round']),steps=min(200000,options['steps']*task['round']))
                                options['validation_seed']=20000000+index*100000+task['round']*10000
                                try:
                                    result = execute(stage, task, options, stopped)
                                    task['stages'][stage] = result
                                    save()
                                except InterruptedError:
                                    raise
                                except Exception as exc:
                                    task.update(state='failed', error=str(exc))
                                    save()
                                    break
                        if task['state'] != 'failed':
                            report=task['stages']['validate']
                            passed=report_passes(report)
                            task.setdefault('history',[]).append(copy.deepcopy(task['stages']))
                            if passed or task['round']==value['options'].get('rounds',1):
                                task.update(training_complete=True,simulation_qualified=passed)
                                task['stages']=max(task['history'],key=lambda stages:stages['validate'].get('successes',0))
                            else:
                                task.update(round=task['round']+1,stages={})
                            save()
                    if task.get('training_complete') and task['state']!='failed':
                        # Adaptive model selection needs one final, unused test
                        # set. A single model's original test already is blind.
                        if len(task['history'])>1:
                            options=dict(value['options'],validation_seed=500000000+index*100000)
                            for key,stage in (('final_validate','validate'),('final_export','export')):
                                if key not in task['stages']:
                                    if stopped():
                                        raise InterruptedError('Итоговая проверка приостановлена; продолжите очередь')
                                    progress(f"{task_spec(task['task_id'])['title']} · итоговая независимая проверка")
                                    try:
                                        task['stages'][key]=execute(stage,task,options,stopped)
                                        task['stages'][stage]=task['stages'][key]
                                        save()
                                    except InterruptedError:
                                        raise
                                    except Exception as exc:
                                        task.update(state='failed',error=str(exc))
                                        save()
                                        break
                        if task['state']!='failed':
                            report=task['stages']['validate']
                            task.update(state='complete',simulation_qualified=report_passes(report))
                            save()
            value.update(state='complete', failed_tasks=sum(t['state']=='failed' for t in value['tasks']))
        except InterruptedError as exc:
            value.update(state='paused', reason=str(exc))
        finally:
            save()
        return value
