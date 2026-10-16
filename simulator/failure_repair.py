"""Evidence-led synthesis of small controller programs, never arbitrary code.

Diagnostics may read the independent outcome only after an episode stops.
Actors receive a checked program, robot-joint priors and sensors, NOT outcomes,
scene IDs, labels, hidden object coordinates or a language model's commands.
Each proposed program must beat a control on the same training scenes before
promotion. Final qualification is still a separate unchanged test.
"""
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path

OPERATORS={
    'history_ik_seeds':'Повторно решить кинематику с проверенными прежними положениями руки',
    'cartesian_feedback':'Исправлять ошибку положения по измеренной кинематике перед смыканием',
    'surface_axis_grasp':'Ориентировать пальцы поперёк измеренного участка ткани',
    'extended_hold_probe':'Найти проверочное движение также в боковых направлениях',
    'probe_feasible_view':'Проверить достижимость обзора вместе с последующим контрольным движением',
    'visibility_planned_lift':'Спланировать высокий подъём с видимым участком ткани на всём пути',
    'reacquire_surface':'Получить новый обзор и выбрать заново наблюдаемый участок ткани',
}
CAUSES={
    'ik_infeasible':('Недостижимая траектория захвата',('history_ik_seeds','cartesian_feedback')),
    'probe_infeasible':('Нет достижимого проверочного движения',('extended_hold_probe','probe_feasible_view','history_ik_seeds')),
    'missed_material':('Ткань физически не поднята',('surface_axis_grasp','cartesian_feedback')),
    'held_but_unseen':('Физическое удержание есть, камера его не подтверждает',('visibility_planned_lift','probe_feasible_view','extended_hold_probe')),
    'hold_lost_or_occluded':('После подъёма потеряно удержание либо его видимость',('visibility_planned_lift','surface_axis_grasp')),
    'placement_unseen':('Физическое размещение не подтверждено изображением',('history_ik_seeds',)),
    'surface_occluded':('Выбранный участок ткани закрыт после подъезда',('reacquire_surface','history_ik_seeds')),
    'vision_timeout':('Модель зрения не успела обработать кадр',()),
    'compute_deadline':('Вычисление не уложилось во время этапа',()),
    'vision_backend_failure':('Сбой процесса модели зрения',()),
    'unclassified':('Причина пока не установлена',()),
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def program(value=None):
    if value is None:value={'schema':1,'operators':[]}
    if (not isinstance(value,dict) or set(value)!={'schema','operators'} or type(value['schema']) is not int or
        value['schema']!=1 or not isinstance(value['operators'],list) or len(value['operators'])>len(OPERATORS) or
        any(type(x) is not str or x not in OPERATORS for x in value['operators']) or
        len(set(value['operators']))!=len(value['operators'])):
        raise ValueError('Invalid controller repair program')
    return dict(schema=1,operators=sorted(value['operators']))


def enabled(value,operator):return operator in program(value)['operators']


def diagnose_report(report):
    execution=report.get('execution',{});truth=report.get('independent_physics_outcome',{})
    stage=execution.get('stage',report.get('phase'));reason=str(report.get('reason',''))
    infrastructure=infrastructure_failure(report)
    if infrastructure:return infrastructure,'identified'
    if 'участок ткани закрыт' in reason:return 'surface_occluded','identified'
    # Explicit solver failure identifies a missing feasible trajectory, not
    # a material miss. A camera refusal by itself identifies neither slipping
    # nor an empty grip; preserve that uncertainty in the diagnosis.
    if 'Проверочное движение недостижимо' in reason:return 'probe_infeasible','identified'
    if any(s in reason for s in ('точного захвата','полного смыкания','IK:','кинематик')):
        return 'ik_infeasible','identified'
    if stage=='verify_place' and truth.get('success') is True:return 'placement_unseen','identified'
    if stage in ('verify_hold','carry','support'):
        if truth.get('grasp_contact') is True and truth.get('lifted') is True:
            return 'held_but_unseen','identified' if 'Камера' in reason or 'кадр' in reason else 'hypothesis'
        if truth.get('lifted') is False:return 'missed_material','identified'
        if truth.get('lifted') is True:return 'hold_lost_or_occluded','hypothesis'
    return 'unclassified','unknown'


def infrastructure_failure(report):
    reason=str(report.get('reason',''))
    if any(text in reason for text in ('Модель зрения не ответила','vision worker timeout','vision service did not start')):
        return 'vision_timeout'
    if 'Stage deadline expired' in reason:return 'compute_deadline'
    if any(text in reason for text in ('Vision worker disconnected','vision service failed','Grounding model checksum mismatch',
        'Vision backend unavailable','Semantic frame mismatch')):
        return 'vision_backend_failure'
    return None


def diagnosis(rows,root):
    root=Path(root).resolve();cases=[]
    for row in rows:
        if row.get('status') not in ('complete','infrastructure_failed') or row.get('success') is True:continue
        try:
            path=Path(row['report']).resolve()
            if not path.is_relative_to(root):raise ValueError('Report outside storage')
            content=path.read_bytes();report=json.loads(content)
        except (KeyError,OSError,ValueError,TypeError):continue
        code,certainty=diagnose_report(report)
        cases.append(dict(code=code,certainty=certainty,episode_id=row.get('episode_id'),
            report=str(path),sha256=hashlib.sha256(content).hexdigest(),
            stage=report.get('execution',{}).get('stage',report.get('phase')),reason=report.get('reason')))
    counts=Counter(c['code'] for c in cases)
    return dict(schema=1,kind='post_stop_evidence_diagnosis',cases=cases,
        causes=[dict(code=code,title=CAUSES[code][0],count=count,operators=list(CAUSES[code][1]))
            for code,count in counts.most_common()],
        runtime_actions=[dict(cause=code,action='shared_pinned_vision_and_same_scene_retry_with_lower_parallelism',
            motion_reward_excluded=True) for code in counts if code in ('vision_timeout','compute_deadline','vision_backend_failure')],
        sensor_absence_is_not_empty_grip=True,acceptance_changed=False)


def repair_target(base,candidate,analysis):
    changed=set(program(base)['operators'])^set(program(candidate)['operators'])
    for cause in analysis.get('causes',[]):
        if changed.intersection(cause['operators']):return cause['title']
    return 'Проверка подсказок из прежних записей'


def repair_proposals(state,analysis):
    """Causal trials change PROGRAM STRUCTURE with numeric settings fixed."""
    current=program(state.get('controller_program'));base=digest(current)
    context=state.get('curriculum_pairs_passed',0)+100*state.get('validation_round',0)
    tried={(x['base'],x['candidate'],x['context']) for x in state.get('repair_trials',[])}
    names=[]
    for cause in analysis.get('causes',[]):
        for name in cause['operators']:
            if name not in names:names.append(name)
    if state.get('experience',{}).get('ik_seeds') and 'history_ik_seeds' not in names:names.append('history_ik_seeds')
    candidates=[current]
    modifications=[(name,) for name in names]+list(combinations(names,2))
    for change in modifications:
        if 'history_ik_seeds' in change and not state.get('experience',{}).get('ik_seeds'):continue
        ops=set(current['operators'])
        # Removal tests can repair a previously helpful operator that becomes
        # harmful in a new context. This is not an irreversible flag cascade.
        ops.symmetric_difference_update(change);choice=program(dict(schema=1,operators=sorted(ops)))
        if (base,digest(choice),context) in tried:continue
        candidates.append(choice)
        if len(candidates)==4:break
    if len(candidates)==1:return None
    while len(candidates)<4:candidates.append(current)
    return candidates


def record_trial(state,programs,selected,improved):
    base=digest(programs[0]);context=state.get('curriculum_pairs_passed',0)+100*state.get('validation_round',0)
    for candidate in programs[1:]:
        if candidate==programs[0]:continue
        state.setdefault('repair_trials',[]).append(dict(generation=state['generation'],base=base,
            candidate=digest(candidate),program=candidate,context=context,
            promoted=improved and candidate==programs[selected],comparison='paired_same_seed_same_parameters'))
    if improved:state['controller_program']=program(programs[selected])
