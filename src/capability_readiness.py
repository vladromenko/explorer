"""Capability-scoped readiness with explicit evidence-producing workflows."""
import json
from pathlib import Path
import time


CAPABILITIES={
 'OBSERVE_LOCAL':dict(runtime=('camera',),evidence=(),permissions=(),policy=None,operate=False),
 'SEARCH_OBJECT':dict(runtime=('camera','grounding'),evidence=(),permissions=(),policy=None,operate=False),
 'LEARN_GRASP_LOCAL':dict(runtime=('controller','camera','arm','stationary','power'),
    evidence=('handeye','local_grasp_profile'),permissions=('arm_motion','target_contact'),policy=None,operate=False),
 'LEARN_PLACE_LOCAL':dict(runtime=('controller','camera','arm','stationary','power'),
    evidence=('handeye','local_grasp_profile','local_reset_profile'),permissions=('arm_motion','target_contact'),policy=None,operate=False),
 'LEARN_PUSH_LOCAL':dict(runtime=('controller','camera','arm','stationary','power'),
    evidence=('handeye','training_zone'),permissions=('arm_motion','target_contact','scene_preparation'),policy=None,operate=False),
 'EXPLORE_LOCAL':dict(runtime=('controller','lidars','odometry','navigation_runtime','power'),
    evidence=('map',),permissions=('base_motion',),policy=None,operate=False),
 'NAVIGATE':dict(runtime=('controller','lidars','odometry','navigation_runtime','power'),
    evidence=('map','localization'),permissions=('base_motion',),policy=None,operate=True),
 'PICK_LOCAL':dict(runtime=('controller','camera','arm','stationary','power'),
    evidence=('handeye','local_grasp_profile'),permissions=('arm_motion','target_contact'),policy='grasp',operate=True),
 'PLACE_LOCAL':dict(runtime=('controller','camera','arm','stationary','power'),
    evidence=('handeye','local_grasp_profile','local_reset_profile'),permissions=('arm_motion','target_contact'),policy='place',operate=True),
 'PICK_AND_DELIVER':dict(runtime=('controller','camera','arm','lidars','odometry','navigation_runtime','power'),
    evidence=('handeye','localization','delivery_setup'),permissions=('base_motion','arm_motion','target_contact'),policy='delivery',operate=True),
 'AUTONOMOUS_DAY':dict(runtime=('controller','power'),evidence=('day_profile',),permissions=(),policy=None,operate=False),
}

WORKFLOWS={
 'handeye':'Использовать уже принятую привязку камеры к arm4; при несовпадении остановить только манипуляцию.',
 'local_grasp_profile':'В ручном режиме показать открытое и удерживающее предмет положения захвата и сохранить командные углы как экспериментальный профиль. Миллиметры и сила не придумываются.',
 'local_reset_profile':'Выбрать видимую локальную область возврата предмета; робот проверит опору, отделение и отвод руки.',
 'training_zone':'Выбрать на кадре ограниченную область и разрешённые лёгкие учебные предметы.',
 'map':'Создать или выбрать карту штатным SLAM.',
 'localization':'Запустить guided localization validation: несколько возвратов, lidar matching и pose error без изменения TF.',
 'delivery_setup':'Сохранить принятые search/destination places и один физически проверенный сквозной сценарий.',
 'day_profile':'Создать ограниченный Autonomous Day profile с окном времени, зонами, объектами и бюджетами.',
}


def read(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError,TypeError):return {}


def fresh(value,now,limit):
    at=value.get('at')
    return isinstance(at,(int,float)) and 0<=now-at<limit


class CapabilityReadiness:
    def __init__(self,root):self.root=Path(root)

    def facts(self,now=None):
        now=time.time() if now is None else now
        status=read(self.root/'data/status.json');perception=read(self.root/'data/perception.json')
        power=read(self.root/'data/power.json');handeye=read(self.root/'config/handeye-accepted.json')
        commissioning=read(self.root/'config/commissioning.json');arm=read(self.root/'data/arm-state.json')
        navigation=read(self.root/'data/navigation-health.json');planning=read(self.root/'data/planning-health.json')
        permissions=read(self.root/'data/autonomy-permissions.json');day=read(self.root/'config/autonomous-day.json')
        graduation=read(self.root/'data/autonomy-graduation-status.json').get('accepted',[])
        scopes=set(permissions.get('scopes',[])) if permissions.get('expires_at',0)>now else set()
        sensors=status.get('sensor_age',{}) if fresh(status,now,2) else {}
        arm_ready=(arm.get('boot_id')==status.get('boot_id') and
                   arm.get('phase')=='command_elapsed_observation_required' and
                   arm.get('at',0)>read(self.root/'data/arm-telemetry-fault.json').get('at',0))
        return dict(
          runtime=dict(
            controller=(fresh(status,now,2) and sensors.get('odom',99)<.5),
            camera=(fresh(perception,now,2) and perception.get('image_stamp',0)>now-2),
            grounding=(self.root/'.venv-learning/bin/python').exists(),
            arm=arm_ready,
            stationary=(status.get('stop_latched') is True or status.get('base_hold_confirmed') is True) and
                       all(abs(v)<.001 for v in status.get('velocity',[99,99,99])),
            power=fresh(power,now,4) and power.get('state') in ('NORMAL','IDLE','PERFORMANCE'),
            lidars=all(sensors.get(name,99)<.6 for name in ('scan0','scan1')),
            odometry=sensors.get('odom',99)<.5,
            navigation_runtime=(fresh(navigation,now,4) and navigation.get('stage')=='active' and
                                fresh(planning,now,4) and planning.get('stage')=='active')),
          evidence=dict(
            handeye=(handeye.get('execution_authorized') is True and bool(handeye.get('physical_validation_record'))),
            local_grasp_profile=bool(read(self.root/'config/local-grasp-profile.json').get('experimental_execution_authorized')),
            local_reset_profile=bool(read(self.root/'config/local-reset-profile.json').get('experimental_execution_authorized')),
            training_zone=bool(read(self.root/'config/local-training-zone.json').get('accepted_by_user')),
            map=bool(commissioning.get('lidar_tf_validated')),
            localization='localization' in graduation or commissioning.get('localization_verified') is True,
            delivery_setup=(self.root/'config/delivery-acceptance.json').exists(),
            day_profile=day.get('enabled') is True,
          ),permissions=scopes,accepted=set(graduation),day_profile=day,
          provenance=dict(joint_state='command_estimate',battery_soc='unavailable',camera_world_geometry='accepted_stationary_handeye'))

    def status(self,now=None):
        facts=self.facts(now);rows=[]
        for name,spec in CAPABILITIES.items():
            runtime_missing=[key for key in spec['runtime'] if facts['runtime'].get(key) is not True]
            evidence_missing=[key for key in spec['evidence'] if facts['evidence'].get(key) is not True]
            permission_missing=[key for key in spec['permissions'] if key not in facts['permissions']]
            policy=spec['policy'];policy_validated=policy is None or policy in facts['accepted']
            available=not runtime_missing and not evidence_missing
            experimental_ready=available and not permission_missing
            operate_accepted=experimental_ready and (policy_validated if spec['operate'] else True)
            workflow=[dict(id=key,instruction=WORKFLOWS[key]) for key in evidence_missing]
            rows.append(dict(id=name,runtime_ready=not runtime_missing,runtime_missing=runtime_missing,
                evidence_ready=not evidence_missing,evidence_missing=evidence_missing,
                experimental_permission=not permission_missing,permission_missing=permission_missing,
                capability_available=available,experimental_ready=experimental_ready,
                learned_policy_validated=policy_validated,production_accepted=operate_accepted,
                evidence_workflow=workflow,requires_localization='localization' in spec['evidence'],
                requires_destination='delivery_setup' in spec['evidence']))
        return dict(at=time.time(),capabilities=rows,provenance=facts['provenance'],
                    separation=['hardware_runtime','experimental_permission','capability_evidence','learned_policy','production_acceptance'])
