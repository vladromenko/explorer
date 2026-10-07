"""One bounded experiment backend for web, CLI and Telegram. Analysis cannot actuate.

Recorded input is immutable per run; replay uses the same snapshot. Physical
skills keep their existing controllers and are never reachable through replay.
"""
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
import shutil
from arm_feedback import describe
from experiment_memory import ExperienceMemory


DEFINITIONS = [
 ('E01','Текущее положение руки','Интерфейс сервоприводов','Источники, свежесть и реальный путь чтения'),
 ('E02','Найти предмет','YOLO / GroundingDINO','Сравнение доступных наблюдений и глубины'),
 ('E03','Запомнить предмет и перемещение','DynaMem / EPM','ID, исправления владельца, история add/update/remove'),
 ('E04','Спросить память о прошлом','3D-Mem / 3DLLM-Mem','Релевантные ракурсы против последних кадров'),
 ('E05','Выбрать следующий ракурс','Активное восприятие','Рейтинг границ исследования по пути и энергии'),
 ('E06','Разобрать составную просьбу','BUMBLE / Hi Robot / VoLo / PARTNR','План зарегистрированных навыков с условиями'),
 ('E07','Проверить прогресс по истории','Robo-Dopamine 2.0','Подтверждённые этапы, ошибки и неизвестность'),
 ('E08','Сравнить прогноз и результат','V-JEPA 2 / X-MOBILITY','Кинематический прогноз и измеренная ошибка'),
 ('E09','Навести руку по изображению','ForceSight','Ошибка цели на кадре и допуски малой коррекции'),
 ('E10','Выбрать захват','Геометрический baseline / GraspGen-X','Кандидаты и причины геометрической отбраковки'),
 ('E11','Оценить закрытие захвата','ZeroTouch / RETAF / Reactive Diffusion Policy','План ограниченного закрытия и свежесть зрения'),
 ('E12','Подготовить данные контакта','ZeroTouch / TacImag / FELT / Forces for Free / Feel the Force','Проверка настоящих контактных меток и входов'),
 ('E13','Проверить удержание','Визуальное подтверждение исхода','Удержание, потеря видимости и неизвестный исход'),
 ('E14','Взять и положить','Последовательность навыков','Порядок подхода, контакта, подъёма и размещения'),
 ('E15','Путь с учётом руки','Nav2 / NavDP / X-NavDP / NaVILA','Габариты, выбранная цель и готовность навигации'),
 ('E16','Оценить показ или исправление','LeRobot / HIL-SERL / RECAP','Качество эпизодов и вмешательства человека'),
 ('E17','Проверить обученную политику','ACT / SmolVLA / RTC','Реальные версии ACT и результаты проверки'),
 ('E18','Выбрать, чему учить дальше','Arcadia / EPM-DDAFT','Очередь показов по неудачам и вмешательствам'),
]

# Operator inputs are separate from research names and never contain invented evidence.
OPERATOR_FORMS = {
    "E01": ("Робот", "none", "Показать углы и источник", "Проверит последние команды руки и наличие измеренной обратной связи."),
    "E02": ("Камера", "query", "Проверить видимые предметы", "Покажет свежие совпадения детектора. Для другой модели доступен отдельный поиск по названию."),
    "E03": ("Память", "query", "Сохранить текущий ракурс", "Сохранит свежие видимые объекты в памяти; повтор одного кадра не создаёт дубликат."),
    "E04": ("Память", "query", "Найти в памяти", "Сравнит подходящие прошлые наблюдения с последними ракурсами."),
    "E05": ("Карта", "none", "Показать направления исследования", "Рассчитает доступные границы карты и ранжирует направления без движения базы."),
    "E06": ("Задачи", "task", "Разобрать команду", "Составит порядок навыков и покажет, какие условия ещё нужны для исполнения."),
    "E07": ("Записанные данные", "events", "Проверить историю", "Оценит прогресс по подтверждённым событиям из вашей записи."),
    "E08": ("Записанные данные", "motion", "Посчитать ошибку движения", "Сравнит прогноз по скорости с двумя записанными позами. Новых команд шасси не отправляет."),
    "E09": ("Камера", "roi", "Проверить положение цели", "По выделенной области RGB-D вычислит центр и ошибку наведения."),
    "E10": ("Камера", "roi", "Рассчитать варианты захвата", "По выделенной области оценит глубину, размеры и две ориентации. Результат остаётся геометрической гипотезой."),
    "E11": ("Камера", "query", "Проверить условия закрытия", "Покажет цель, свежесть зрения и недостающие условия контроля контакта."),
    "E12": ("Записанные данные", "contact", "Проверить данные контакта", "Проверит поля и происхождение контактных меток перед обучением."),
    "E13": ("Записанные данные", "hold", "Проверить удержание", "Проверит последовательности RGB-D до и после подъёма; недостаточные доказательства дают неизвестный результат."),
    "E14": ("Задачи", "task", "Проверить план взять и положить", "Проверит порядок подхода, захвата, проверки удержания, перевозки и размещения."),
    "E15": ("Карта", "none", "Проверить готовность перевозки", "Покажет свежесть лидаров и допуски навигации с рукой и грузом."),
    "E16": ("Обучение", "none", "Проверить сохранённые показы", "Прочитает полные мобильные показы и исправления: результат, число кадров и пригодность."),
    "E17": ("Обучение", "none", "Показать модели и обучение", "Покажет настоящие задания ACT, backend и результаты проверки кандидатов."),
    "E18": ("Обучение", "none", "Подсказать следующий показ", "Построит рекомендации по сохранённым ошибкам и вмешательствам."),
}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fresh(stamp, now, ttl):
    return finite(stamp) and 0 <= now-stamp < ttl


def sequence_plan(text, flags):
    lowered = text.casefold()
    delivery = any(w in lowered for w in ('принес', 'привез', 'перенес', 'носк', 'печен', 'взять', 'полож', 'take', 'deliver'))
    names = ['observe', 'locate', 'navigate', 'align', 'grasp', 'verify_hold', 'carry', 'support', 'release', 'verify_place'] if delivery else ['observe', 'recall']
    requirements = {'navigate':['base_commissioned','mcu_watchdog_verified','lidar_tf_validated','localization_verified'],
                    'align':['camera_tf_validated','arm_commissioned'], 'grasp':['arm_commissioned','camera_tf_validated'],
                    'verify_hold':['camera_tf_validated'], 'carry':['base_commissioned','mcu_watchdog_verified','transport_pose_verified'],
                    'support':['camera_tf_validated'], 'release':['placement_verified']}
    steps = [dict(skill=n, requires=requirements.get(n, []),
                  missing=[f for f in requirements.get(n, []) if flags.get(f) is not True]) for n in names]
    return dict(steps=steps, executed=False, arbitrary_motor_commands=False,
                reason='Перенос требует подтверждённого захвата; отпускание — подтверждённой опоры',
                next_step='observe', task=text, parser='bounded_keyword_baseline')


class Experiments:
    def __init__(self, root, feedback_graph=None, frontiers=None, live_status=None):
        self.root = Path(root)
        self.folder = self.root/'data/experiments'
        self.folder.mkdir(exist_ok=True)
        self.memory = ExperienceMemory(self.root)
        self.lock = threading.RLock()
        self.generation = 0
        self.graph = feedback_graph or (lambda: {})
        self.frontiers = frontiers or (lambda: [])
        self.live_status = live_status or (lambda: {})
        self.active = None
        self.profile = 'Работа'
        self.version = read_json(self.root/'config/release.json').get('version', 'lab-20260928')

    def snapshot(self):
        live = self.live_status()
        return dict(at=time.time(), state=read_json(self.root/'data/status.json'),
            perception=read_json(self.root/'data/perception.json'), search=live.get('search', {}),
            policy_preview=live.get('policy_preview', {}),
            map=read_json(self.root/'data/map.json'), learning=live.get("learning",read_json(self.root/'data/learning-status.json')),
            mobile_policy=live.get("mobile_policy",{}),
            graph=self.graph(), joint_state=read_json(self.root/'data/arm-state.json'),
            map_session=read_json(self.root/'data/map_session.json'),
            memory_objects=self.memory.objects(),curriculum=self.memory.curriculum(),
            episodes=[read_json(p) for folder in ("demonstrations","mobile-demonstrations")
                      for p in (self.root/"data"/folder).glob("*/episode.json")])

    def capture(self):
        import numpy as np
        folder=self.root/'data/experiment-captures';folder.mkdir(exist_ok=True)
        if len(list(folder.glob('*.npz')))>=64:
            raise ValueError('Лимит 64 снимков для ручного выбора области; сохраните нужные записи')
        # Idle perception intentionally runs at 0.5 Hz. Wait for its next genuine
        # synchronized frame rather than widening freshness or waking all models.
        deadline=time.monotonic()+3
        sample=None
        while sample is None and time.monotonic()<deadline:
            try:
                with np.load(self.root/"data/rgbd-snapshot.npz",allow_pickle=False) as frame:
                    if fresh(float(frame["stamp"]),time.time(),1.5):
                        sample={key:frame[key].copy() for key in frame.files}
            except FileNotFoundError:pass
            if sample is None:time.sleep(.05)
        if sample is None:raise ValueError("Нет свежего совмещённого RGB-D за 3 секунды")
        ident=uuid.uuid4().hex
        np.savez_compressed(folder/(ident+".npz"),**sample)
        h,w=sample["rgb"].shape[:2]
        return dict(id=ident,stamp=float(sample["stamp"]),width=w,height=h)

    def capture_path(self, ident):
        if not isinstance(ident,str) or len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):
            raise ValueError('Некорректный ID снимка')
        path=self.root/'data/experiment-captures'/(ident+'.npz')
        if not path.is_file():raise ValueError('Снимок не найден')
        return path

    def capture_geometry(self, params):
        import numpy as np
        from lab_geometry import candidates
        with np.load(self.capture_path(params['capture_id']),allow_pickle=False) as frame:
            stamp=float(frame['stamp'])
            result=candidates(frame['depth'],frame['k'],params.get('bbox'),stamp,stamp,stamp)
        result.update(capture_id=params['capture_id'],capture_stamp=stamp,age_s=time.time()-stamp,
                      analysis_of_frozen_frame=True,physical_execution_allowed=False)
        return result

    def catalog(self):
        state=read_json(self.root/'data/status.json')
        flags=state.get('commissioning', {})
        missing=[f for f in ('base_commissioned','mcu_watchdog_verified','camera_tf_validated','arm_commissioned') if flags.get(f) is not True]
        return dict(version=self.version, profile=self.profile, active=self.active,
            experiments=[dict(id=i, name=n, inspiration=source, implementation=details,
                implementation_type='engineering_baseline', code_available=True, model_trained=False,
                physically_verified=False, admitted_modes=['observe','replay','shadow'],
                physical_admitted=False, physical_blocked_by=missing,
                prepare='Робот неподвижен. Для сравнения выберите прежний запуск.',
                group=OPERATOR_FORMS[i][0],input_kind=OPERATOR_FORMS[i][1],
                action_label=OPERATOR_FORMS[i][2],purpose=OPERATOR_FORMS[i][3],
                requires_input=OPERATOR_FORMS[i][1] in ("events","motion","contact","hold","roi"),
                measures=details, next_step='Откройте результат: там указаны условия конкретного действия')
                for i,n,source,details in DEFINITIONS])

    def preflight(self, experiment, mode, params):
        if experiment not in {d[0] for d in DEFINITIONS}:
            raise ValueError('Неизвестный эксперимент')
        if mode not in ('observe','replay','shadow'):
            raise ValueError('Физический режим лаборатории не допущен; используйте проверенные ручные режимы руки')
        if len(json.dumps(params, allow_nan=False)) > 12000:
            raise ValueError('Слишком большие параметры')
        if mode == 'replay' and not params.get('run_id'):
            raise ValueError('Для replay нужен сохранённый запуск')
        if not params.get('run_id'):
            state=read_json(self.root/'data/status.json')
            if not fresh(state.get('at'),time.time(),2):
                raise ValueError('Нет свежего состояния робота')
        blocked=[]
        if mode!="replay":
            if experiment=="E07" and not isinstance(params.get("events"),list):
                blocked.append("Вставьте список events из записанной истории")
            elif experiment=="E07" and (len(params["events"])>100 or any(not isinstance(row,dict) for row in params["events"])):
                blocked.append("events должен содержать до 100 объектов событий")
            if experiment=="E08":
                for name in ("before","after"):
                    pose=params.get(name)
                    if not isinstance(pose,dict) or not all(finite(pose.get(key)) for key in ("x","y","yaw")):
                        blocked.append("Нужна записанная поза "+name+": x, y, yaw")
                if not finite(params.get("duration_s")) or not 0<params["duration_s"]<=10:
                    blocked.append("Укажите длительность в секундах от 0 до 10")
                velocity=params.get("velocity")
                if not isinstance(velocity,list) or len(velocity)!=3 or not all(finite(value) for value in velocity):
                    blocked.append("Нужна записанная команда velocity: vx, vy, wz")
            if experiment=="E12" and (not isinstance(params.get("samples"),list) or not params["samples"]):
                blocked.append("Вставьте реальные samples с контактными метками")
            elif experiment=="E12" and (len(params["samples"])>100 or any(not isinstance(row,dict) for row in params["samples"])):
                blocked.append("samples должен содержать от 1 до 100 объектов записей")
            if experiment=="E13" and any(not isinstance(params.get(name),list) or len(params[name])<3 for name in ("before","after")):
                blocked.append("Нужны минимум три наблюдения before и after")
            if experiment in ("E09","E10") and params.get("capture_id"):
                bbox=params.get("bbox")
                if not isinstance(bbox,list) or len(bbox)!=4 or not all(finite(value) for value in bbox):
                    blocked.append("Выделите предмет прямоугольником на сохранённом кадре")
        return dict(ready=not blocked,blocked_by=blocked,motor_access=False,mode=mode,experiment=experiment)

    def cancel(self):
        with self.lock:
            self.generation += 1
            if self.active:
                self.active['cancel_requested']=True
        return dict(cancelled=True, physical_freeze_claimed=False)

    def get(self, run_id):
        if not isinstance(run_id,str) or len(run_id)!=32 or any(c not in '0123456789abcdef' for c in run_id):
            raise ValueError('Некорректный ID запуска')
        path=self.folder/(run_id+'.json')
        if not path.is_file():
            raise ValueError('Запуск не найден')
        return json.loads(path.read_text())

    def recent(self):
        paths=sorted(self.folder.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]
        return [dict(id=r['id'],experiment=r['experiment'],mode=r['mode'],state=r['state'],
                     at=r['at'],summary=r.get('result',{}).get('summary'),elapsed_ms=r.get('elapsed_ms'))
                for r in (read_json(p) for p in paths)]

    def start(self, experiment, mode, params, request_id):
        check=self.preflight(experiment,mode,params)
        if not check["ready"]:raise ValueError("; ".join(check["blocked_by"]))
        if not isinstance(request_id,str) or not 16<=len(request_id)<=80:
            raise ValueError('Нужен уникальный идентификатор запроса')
        digest=hashlib.sha256(json.dumps([experiment,mode,params],sort_keys=True,allow_nan=False).encode()).hexdigest()
        with self.lock:
            for p in self.folder.glob('*.json'):
                old=read_json(p)
                if old.get('request_id')==request_id:
                    if old['request_digest']!=digest:
                        raise ValueError('Повторный ID с другими параметрами')
                    return old
            if self.active:
                raise ValueError('Другой эксперимент выполняется')
            if len(list(self.folder.glob('*.json')))>=1000:
                raise ValueError('Лимит 1000 запусков: сохраните экспорт; данные не удалены')
            generation=self.generation
            run=dict(id=uuid.uuid4().hex,request_id=request_id,request_digest=digest,at=time.time(),
                     experiment=experiment,mode=mode,params=params,version=self.version,state='running',
                     motor_access=False,interventions=[])
            self.active=run
        begin=time.monotonic()
        try:
            previous=self.get(params['run_id']) if params.get('run_id') else None
            if previous and previous["experiment"]!=experiment:
                raise ValueError("Выберите прошлый запуск этого же эксперимента")
            effective_params=previous.get("effective_params",previous["params"]) if previous and mode=="replay" else params
            snapshot=previous['inputs'] if previous else self.snapshot()
            if not previous and experiment=='E05':snapshot['frontiers']=self.frontiers()
            if not previous and experiment=='E04':snapshot['retrieval']=self.memory.retrieve(str(params.get('query','')))
            if not previous and experiment in ('E09','E10') and params.get('capture_id'):
                snapshot['geometry']=self.capture_geometry(params)
            run['inputs']=snapshot
            run['input_sha256']=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
            run['calibration']=snapshot['state'].get('commissioning',{})
            run["effective_params"]=effective_params
            run['result']=self.evaluate(experiment,snapshot,effective_params,mode)
            with self.lock:
                run['state']='cancelled' if generation!=self.generation else 'completed'
                if run['state']=='cancelled':
                    run['result']['summary']='Отменено; результата нельзя использовать для действия'
            run['elapsed_ms']=round((time.monotonic()-begin)*1000,2)
        except (ValueError,OSError,KeyError,TypeError,AttributeError) as exc:
            run.update(state='failed',result=dict(summary=str(exc)),elapsed_ms=round((time.monotonic()-begin)*1000,2))
        finally:
            with self.lock:
                self.active=None
                path=self.folder/(run['id']+'.json')
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(run,ensure_ascii=False,allow_nan=False))
                temporary.replace(path)
        return run

    def evaluate(self, ident, snap, params, mode):
        s=snap['state']; p=snap['perception']; flags=s.get('commissioning',{})
        now=snap['at']; query=str(params.get('query','')).strip()[:500]
        visible=p.get('objects',[]) if fresh(p.get('image_stamp',p.get('at')),now,2) else []
        evidence=dict(frame_stamp=p.get('image_stamp'),sensor_age=s.get('sensor_age'),
                      measured_arm_angles=False,physical_success_verified=False)
        if ident=='E01':
            arm=describe(s,snap['graph'],now)
            return dict(summary="Источник углов: измерения приводов." if arm["measured"].get("available") is True else
                        "Показаны расчётные команды руки. Измеренные углы robotio не предоставляет.",
                        arm=arm,evidence=evidence)
        if ident=='E02':
            matches=[o for o in visible if not query or query.casefold() in str(o.get('label','')).casefold()]
            return dict(summary=f'Свежих совпадений детектора: {len(matches)}. Метка остаётся гипотезой.',
                objects=matches,alternative=snap.get('search'),coordinates='camera',mask_available=False,
                same_frame_comparison=False,next_step="Для другого словаря нажмите «Поиск по названию» в этой карточке",evidence=evidence)
        if ident=='E03':
            changed=None
            if mode=='observe' and fresh(p.get('image_stamp',p.get('at')),now,2):
                changed=self.memory.remember_view(p,str(snap.get('map_session',{}).get('id','unknown')))
            return dict(summary='Ракурс сохранён. ID назначаются при явном добавлении; одинаковые предметы не сливаются.' if changed else 'Память прочитана без изменения.',
                observation=changed,objects=[o for o in snap.get('memory_objects',[]) if query.casefold() in o['label'].casefold()],automatic_disappearance=False,evidence=evidence)
        if ident=='E04':
            return dict(summary='Сравнение поиска по меткам и последних ракурсов на одном наборе.',**snap.get('retrieval',{}))
        if ident=='E05':
            candidates=snap.get('frontiers',[])
            if isinstance(candidates,dict):
                candidates=candidates.get('candidates',candidates.get('frontiers',[]))
            ranked=[]
            for c in candidates[:40]:
                distance=c.get('path_distance_m',c.get('distance_m',c.get('distance',0)))
                if finite(distance) and distance>=0:
                    novelty=float(c.get('frontier_cells',c.get('size',c.get('cells',1))))
                    ranked.append(dict(candidate=c,score=novelty/(1+distance),estimated_path_m=distance,
                                       energy_cost_calibrated=False))
            ranked.sort(key=lambda c:c['score'],reverse=True)
            return dict(summary='Предложение ракурса без движения; прирост информации измеряется только после нового наблюдения.',
                ranked=ranked[:6],mission_budget=dict(max_views=6,max_seconds=180,no_gain_limit=2),executed=False)
        if ident in ('E06','E14'):
            return dict(summary='План проверен по порядку действий. Физическое выполнение не запущено.',
                        **sequence_plan(query or 'взять и положить',flags))
        if ident=='E07':
            history=params.get('events',[])
            if not isinstance(history,list) or len(history)>100:
                raise ValueError('Нужно не более 100 событий')
            verified=[e for e in history if e.get('verified') is True and e.get('source') in ('operator','measured_sensor')]
            state='unknown'
            if verified:
                state='failure' if verified[-1].get('outcome')=='failure' else 'progress' if len({e.get('stage') for e in verified})>1 else 'no_confirmed_progress'
            return dict(summary='Прогресс: '+state,progress=state,confirmed_events=verified,
                single_frame_baseline='unknown',occlusion_is_failure=False,closed_gripper_is_success=False)
        if ident=='E08':
            a=params.get('before');b=params.get('after');duration=params.get('duration_s');v=params.get('velocity')
            if not a or not b or not finite(duration) or not 0<duration<=10 or not isinstance(v,list) or len(v)!=3 or not all(finite(x) for x in v):
                return dict(summary='Нужна пара записанных поз, длительность и команда скорости; прогноз не выдуман.',
                    required=['before {x,y,yaw}','after {x,y,yaw}','duration_s','velocity [vx,vy,wz]'],executed=False)
            if not all(finite(x.get(k)) for x in (a,b) for k in ('x','y','yaw')):
                raise ValueError('Позиции должны содержать конечные x, y, yaw')
            theta=a['yaw'];vx,vy,w=v
            dx=(vx*math.cos(theta)-vy*math.sin(theta))*duration
            dy=(vx*math.sin(theta)+vy*math.cos(theta))*duration
            predicted=dict(x=a['x']+dx,y=a['y']+dy,yaw=theta+w*duration)
            error=math.hypot(predicted['x']-b['x'],predicted['y']-b['y'])
            return dict(summary=f'Ошибка локального прогноза: {error:.4f} м.',prediction=predicted,observed=b,error_m=error,
                independent_observation=params.get('observation_source') in ('external_camera','lidar_registration'),
                method='first_order_body_velocity_kinematics',uncertainty='not_calibrated',executed=False)
        if ident in ('E09','E10','E11'):
            if ident in ('E09','E10') and snap.get('geometry'):
                return dict(snap['geometry'],evidence=evidence)
            target=next((o for o in visible if not query or query.casefold() in o.get('label','').casefold()),None)
            if target is None:
                return dict(summary='Нет свежей распознанной цели; коррекция и закрытие запрещены.',evidence=evidence)
            bbox=target.get('bbox',[]);xyz=target.get('position')
            if ident=='E09':
                center=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2] if len(bbox)==4 else None
                return dict(summary='Центр цели доступен; двигаться к нему можно только после подтверждения TF и глубины.',
                    target_pixel=center,xyz_camera=xyz,target=target,closed_loop_executed=False,
                    blocked_by=[k for k in ('camera_tf_validated','arm_commissioned') if flags.get(k) is not True],evidence=evidence)
            if ident=='E10':
                candidates=[dict(roll_deg=roll,rank=n+1,source='geometric_orientation_hypothesis',
                    feasible=None,rejected_by=['Нет подтверждённого преобразования camera→base и калибровки раскрытия'],
                    approach_collision_checked=False,servo_target=None) for n,roll in enumerate((0,45,-45,90))]
                return dict(summary='Геометрические ориентации показаны; исполнимость не подтверждена.',
                    candidates=candidates,depth=xyz,aperture_m=None,learned_model=False)
            return dict(summary='План визуального закрытия; физический контакт не подтверждён.',
                sequence=['limited_close','fresh_observation','check_contact_or_unknown'],maximum_steps=5,
                maximum_joint_step_deg=2,force_newtons=None,contact='unknown',
                blocked_by=['Нет подтверждённого tracking предмета относительно пальцев'],evidence=evidence)
        if ident=='E12':
            samples=params.get('samples',[])
            fields=('wrist_rgb','gripper_state','local_gravity','contact_label','label_source')
            invalid=[dict(index=i,missing=[k for k in fields if k not in row]) for i,row in enumerate(samples[:100])
                     if any(k not in row for k in fields) or row.get('label_source') not in ('calibrated_sensor','validated_tactile_dataset')]
            return dict(summary='Проверка схемы реальных контактных данных; силовая модель не установлена.',
                samples=len(samples),invalid=invalid,required_fields=fields,force_model_available=False,
                next_step='Собрать синхронные данные с проверенными контактными метками или предоставить совместимые валидированные веса')
        if ident=='E13':
            from grasp_verification import verify_lift
            result=verify_lift(params.get('before',[]),params.get('after',[]),flags.get('camera_tf_validated') is True)
            return dict(summary='Удержание: '+result['outcome'],**result)
        if ident=='E15':
            return dict(summary='Проверка готовности навигации с грузом; движение не запущено.',
                body_velocity_axes=['vx','vy','wz'],planar_obstacles=s.get('lidar'),
                upper_obstacle_coverage='not_validated',arm_envelope='not_validated',
                blocked_by=[k for k in ('base_commissioned','lidar_tf_validated','mcu_watchdog_verified','localization_verified','transport_pose_verified') if flags.get(k) is not True],
                next_step='Подтвердить остановку контроллера и габарит транспортной позы',executed=False)
        if ident=='E16':
            episodes=snap.get('episodes',[])
            rows=[dict(id=e.get('id'),task=e.get('name'),steps=e.get("samples",len(e.get('steps',[]))),outcome=e.get('outcome'),
                       quality=e.get("quality"),state=e.get("state"),skill_id=e.get("skill_id"),kind=e.get("kind"),
                       measured=e.get('joint_positions_measured',False),source=e.get('source')) for e in episodes]
            return dict(summary=f'Найдено {len(rows)} показов. Оцените ошибки и вмешательства для очереди обучения.',
                episodes=rows,recording_interface="Обучение → Обучаться → Записать показ",memory_curriculum=snap.get('curriculum',{}))
        if ident=='E17':
            return dict(summary='Фактическое состояние ACT. SmolVLA и RTC не заявлены как установленные политики.',
                learning=snap['learning'],policy_preview=snap.get('policy_preview',{}),mobile_policy=snap.get("mobile_policy",{}),
                alternatives=[dict(name='SmolVLA',available=False,next_step='Адаптация к этому телу, данные и измерение RAM'),
                              dict(name='RTC',available=False,next_step='Совместимая flow-политика с префиксом исполняемого блока')],
                execute_interface="Обучение → Проверить навык")
        suggestions=[]
        for episode in snap.get("episodes",[]):
            quality=episode.get("quality") or {}
            if episode.get("outcome") in ("failure","unknown") or quality.get("usable") is False:
                suggestions.append(dict(episode=episode.get("id"),task=episode.get("name"),
                    outcome=episode.get("outcome"),reason=quality.get("reason") or episode.get("reason"),
                    next_step="Запишите исправленный полный показ и оцените результат на вкладке «Обучение»"))
        return dict(summary="Рекомендации построены по сохранённым оценкам и качеству показов.",
                    mobile_demonstration_suggestions=suggestions,**snap.get('curriculum',{}))
