"""Evidence-driven curriculum and automatic capability graduation.

Operator labels are useful training data but never physical acceptance.  Each
automatic unlock is derived from immutable verifier records and can be revoked
when its evidence disappears or no longer matches the current map/calibration.
"""
import hashlib
import json
import math
from pathlib import Path
import time


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError,TypeError):return {} if default is None else default


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');tmp.replace(path)


class AutonomyGraduation:
    def __init__(self,root):
        self.root=Path(root);self.config=read(self.root/'config/autonomy-graduation.json')
        self.folder=self.root/'data/autonomy-evidence';self.folder.mkdir(parents=True,exist_ok=True)

    def records(self,kind=None):
        rows=[]
        for path in sorted(self.folder.glob('*.json')):
            value=read(path)
            if value.get('schema')!='explorer_autonomy_evidence_v1':continue
            if kind and value.get('kind')!=kind:continue
            if value.get('hardware_executed') is not True or value.get('simulation') is True:continue
            if value.get('verifier_outcome') not in ('success','failure','unsafe'):continue
            value['_path']=path.relative_to(self.root).as_posix();value['_sha256']=digest(path);rows.append(value)
        return rows

    def episodes(self):
        rows=[]
        for path in sorted((self.root/'data/mobile-demonstrations').glob('*/episode.json')):
            value=read(path)
            if value.get('state')=='complete' and set(value.get('stages',[]))=={'travel_to_object','grasp','carry','place'}:
                rows.append(value)
        return rows

    def jobs(self):
        return [read(p) for p in sorted((self.root/'data/learning-jobs').glob('*/job.json'))]

    @staticmethod
    def progress(value,required):return dict(value=value,required=required,complete=value>=required)

    def localization(self):
        c=self.config['localization'];rows=[r for r in self.records('localization_return') if r['verifier_outcome']=='success']
        errors=[r.get('translation_error_m') for r in rows];yaws=[r.get('yaw_error_deg') for r in rows]
        valid=bool(rows) and all(type(v) in (int,float) and math.isfinite(v) for v in errors+yaws)
        errors=errors if valid else [];yaws=yaws if valid else []
        headings=len({r.get('start_heading_bucket') for r in rows if r.get('start_heading_bucket') in ('front','left','right','back')})
        med=lambda x:sorted(x)[len(x)//2] if x else None
        metrics=dict(returns=len(rows),headings=headings,translation_median_m=med(errors),translation_max_m=max(errors) if errors else None,
                     yaw_median_deg=med(yaws),yaw_max_deg=max(yaws) if yaws else None)
        passed=(len(rows)>=c['minimum_returns'] and headings>=c['minimum_headings'] and
                max(errors)<=c['maximum_translation_error_m'] and med(errors)<=c['maximum_median_translation_error_m'] and
                max(yaws)<=c['maximum_yaw_error_deg'] and med(yaws)<=c['maximum_median_yaw_error_deg']) if errors else False
        return dict(id='localization',name='Повторная локализация',state='accepted' if passed else 'training',metrics=metrics,
            progress=[self.progress(len(rows),c['minimum_returns']),self.progress(headings,c['minimum_headings'])],
            next_action='Вернитесь к отмеченной точке с другого направления и нажмите «Зафиксировать возврат»; система сравнит pose и независимое совпадение лидаров.',
            verification='Ошибки pose, совпадение двух лидаров, свежая карта и три различных направления старта.',unlocks=['3D-память на карте','автономное исследование'],evidence=rows)

    def grasp(self):
        c=self.config['visual_grasp'];episodes=self.episodes();task_success=[e for e in episodes if e.get('outcome')=='success']
        corrections=[e for e in episodes if e.get('outcome')=='failure' or e.get('human_interventions',0)>0]
        lifts=[r for r in self.records('retained_lift') if r['verifier_outcome']=='success']
        places=[r for r in self.records('verified_place') if r['verifier_outcome']=='success']
        aperture=[r for r in self.records('gripper_aperture_calibration') if r['verifier_outcome']=='success' and
                  all(type(r.get(k)) in (int,float) and math.isfinite(r[k]) for k in ('open_deg','sock_close_deg','open_aperture_mm'))]
        conditions={(r.get('object_class'),r.get('condition')) for r in lifts if r.get('object_class') and r.get('condition')}
        recent=self.records('retained_lift')[-20:];rate=sum(r['verifier_outcome']=='success' for r in recent)/len(recent) if recent else 0
        passed=(len(aperture)>=c['minimum_aperture_calibrations'] and len(episodes)>=c['minimum_complete_demonstrations'] and len(corrections)>=c['minimum_failures_or_interventions'] and
                len(lifts)>=c['minimum_verified_lifts'] and len(places)>=c['minimum_verified_places'] and
                len(conditions)>=c['minimum_object_conditions'] and rate>=c['minimum_recent_success_rate'])
        metrics=dict(aperture_calibrations=len(aperture),complete_demonstrations=len(episodes),operator_successes=len(task_success),failures_or_interventions=len(corrections),
                     verified_retained_lifts=len(lifts),verified_places=len(places),object_conditions=len(conditions),recent_verified_success_rate=rate)
        return dict(id='visual_grasp',name='Визуально подтверждённый захват robotio',state='accepted' if passed else 'training',metrics=metrics,
            progress=[self.progress(len(aperture),c['minimum_aperture_calibrations']),self.progress(len(episodes),c['minimum_complete_demonstrations']),self.progress(len(corrections),c['minimum_failures_or_interventions']),
                      self.progress(len(lifts),c['minimum_verified_lifts']),self.progress(len(places),c['minimum_verified_places']),self.progress(len(conditions),c['minimum_object_conditions'])],
            next_action='Записывайте полный показ. После закрытия поднимите предмет и удерживайте не менее секунды в поле камеры; после отпускания отведите захват. Неудачи и ваши вмешательства тоже сохраняйте.',
            verification='RGB-D отслеживает тот же предмет относительно захвата, подтверждает отрыв от пола, удержание и размещение. Одна отметка оператора не открывает допуск.',
            unlocks=['замкнутый визуальный захват','проверка политики на роботе'],evidence=aperture+lifts+places)

    def policy(self,grasp):
        c=self.config['policy'];jobs=self.jobs()
        validated=[j for j in jobs if j.get('state') in ('validated','validated_offline') and j.get('validation',{}).get('improves_hold_baseline') is True and j.get('validation',{}).get('held_out_fraction',0)>=c['minimum_held_out_fraction']]
        trials=self.records('policy_trial');success=[r for r in trials if r['verifier_outcome']=='success'];unsafe=[r for r in trials if r['verifier_outcome']=='unsafe']
        passed=grasp['state']=='accepted' and bool(validated) and len(trials)>=c['minimum_supervised_trials'] and len(success)>=c['minimum_verified_successes'] and len(unsafe)<=c['maximum_unsafe_trials']
        return dict(id='policy',name='Обученная политика захвата и перевозки',state='accepted' if passed else 'training',
            metrics=dict(validated_training_jobs=len(validated),supervised_trials=len(trials),verified_successes=len(success),unsafe_trials=len(unsafe)),
            progress=[self.progress(len(validated),1),self.progress(len(trials),c['minimum_supervised_trials']),self.progress(len(success),c['minimum_verified_successes'])],
            next_action='После достаточных показов запустите обучение. Затем выполните 10 запусков с наблюдением; визуальный проверяющий сам запишет успех, срыв или небезопасный исход.',
            verification='Отложенная выборка улучшает baseline; затем минимум 8 из 10 физических запусков подтверждены зрением, небезопасных исходов нет.',
            unlocks=['самостоятельное выполнение выученного навыка'],evidence=trials)

    def delivery(self,localization,policy):
        c=self.config['delivery'];cycles=[r for r in self.records('delivery_cycle') if r['verifier_outcome']=='success' and set(c['required_stages']).issubset(r.get('stages',[]))]
        passed=localization['state']=='accepted' and policy['state']=='accepted' and len(cycles)>=c['minimum_verified_cycles']
        return dict(id='delivery',name='Автономный pick-and-deliver',state='accepted' if passed else 'training',metrics=dict(verified_cycles=len(cycles)),
            progress=[self.progress(len(cycles),c['minimum_verified_cycles'])],
            next_action='После допусков локализации и политики выполните три полных наблюдаемых цикла: поиск → поездка → захват → удержание → перевозка → размещение.',
            verification='Каждая стадия имеет свежие сенсоры; захват и размещение подтверждены независимо от ACK команды.',unlocks=['полная автономная доставка'],evidence=cycles)

    def sync(self,items):
        accepted={i['id'] for i in items if i['state']=='accepted'}
        commissioning=read(self.root/'config/commissioning.json')
        desired=dict(localization_verified='localization' in accepted,gripper_calibrated='visual_grasp' in accepted,
                     visual_closed_loop_arm_verified='visual_grasp' in accepted,learned_policy_verified='policy' in accepted,
                     autonomous_delivery_verified='delivery' in accepted)
        changed=any(commissioning.get(k)!=v for k,v in desired.items())
        if changed:commissioning.update(desired);atomic(self.root/'config/commissioning.json',commissioning)
        grasp=next((i for i in items if i['id']=='visual_grasp' and i['state']=='accepted'),None)
        if grasp:
            aperture=[r for r in self.records('gripper_aperture_calibration') if r['verifier_outcome']=='success'][-1]
            atomic(self.root/'config/gripper-accepted.json',dict(schema_version=1,execution_authorized=True,
                aperture_mm_calibrated=True,open_deg=aperture['open_deg'],sock_close_deg=aperture['sock_close_deg'],
                open_aperture_mm=aperture['open_aperture_mm'],feedback='visual_closed_loop',physical_validation_record=aperture['_path'],
                evidence_sha256=aperture['_sha256']))
        else:
            path=self.root/'config/gripper-accepted.json';current=read(path)
            if current.get('feedback')=='visual_closed_loop' and current.get('execution_authorized') is True:
                current['execution_authorized']=False;current['revoked_reason']='Доказательства учебного допуска больше не действительны';atomic(path,current)
        acceptance={i['id']:dict(state=i['state'],evidence_sha256={r['_path']:r['_sha256'] for r in i['evidence']}) for i in items}
        atomic(self.root/'data/autonomy-graduation-status.json',dict(at=time.time(),accepted=sorted(accepted),records=acceptance))
        return changed

    def status(self):
        localization=self.localization();grasp=self.grasp();policy=self.policy(grasp);delivery=self.delivery(localization,policy)
        items=[localization,grasp,policy,delivery];self.sync(items)
        for item in items:item.pop('evidence',None)
        return dict(at=time.time(),automatic_unlocks=True,arm_feedback_measured=False,
                    arm_feedback_note='Robotio не измеряет суставы; для ограниченной автономности используется отдельный проверяемый визуальный контур.',
                    accepted=[i['id'] for i in items if i['state']=='accepted'],items=items)
