"""Persistent, bounded Autonomous Day profile and activity selection."""
import json
from pathlib import Path
import threading
import time


class AutonomousDayPlanner:
    def choose(self, status, permission, curriculum, resources):
        if permission.get('expires_at',0)<=time.time():return dict(activity='pause',reason='permission_expired')
        if status.get('manual_takeover') or status.get('emergency_stop'):return dict(activity='pause',reason='manual_or_emergency_stop')
        if resources.get('disk_free_gb',0)<2:return dict(activity='ask_help',reason='storage_reserve_low',question='Освободите 2 ГБ на Jetson')
        capabilities=set(permission.get('allowed_capabilities',[]))
        if capabilities&{'EXPLORE_LOCAL','NAVIGATE'} and status.get('localization_valid') is not True:
            return dict(activity='relocalize',reason='pose_unknown',budget_s=120)
        if 'LEARN_GRASP_LOCAL' in capabilities and status.get('local_grasp_ready') is True and status.get('queued_grasp_workflow'):
            return dict(activity='learn_grasp_local',reason='permitted_queued_experiment',
                        workflow_id=status['queued_grasp_workflow'],budget_s=permission.get('maximum_session_s',300))
        queue=curriculum.get('queue',[])
        if queue and queue[0].get('priority',0)>.6:
            return dict(activity='collect_demonstration_request',reason='high_failure_or_intervention_rate',item=queue[0])
        if status.get('training_ready') and resources.get('ram_available_mb',0)>1800:
            return dict(activity='train_candidate',reason='dataset_ready_and_resources_available')
        if status.get('unknown_object_query') and permission.get('base_motion'):
            return dict(activity='bounded_exploration',reason='target_absent_from_memory',budget_s=300)
        return dict(activity='observe_and_update_memory',reason='no_safe_physical_task_ready',budget_s=60)

    def budget(self, permission, requested_seconds):
        remaining=max(0,float(permission.get('expires_at',0))-time.time())
        return min(float(requested_seconds),remaining,float(permission.get('maximum_session_s',8*3600)))


class AutonomousDayRuntime:
    def __init__(self,root,status_provider,resource_provider,curriculum_provider=lambda:{},dispatch=None):
        self.root=Path(root);self.profile_path=self.root/'config/autonomous-day.json'
        self.status_path=self.root/'data/autonomous-day-status.json';self.status_provider=status_provider
        self.resource_provider=resource_provider;self.curriculum_provider=curriculum_provider;self.dispatch=dispatch
        self.planner=AutonomousDayPlanner();self.stop_event=threading.Event();self.last_dispatch=0.
        threading.Thread(target=self.loop,daemon=True).start()

    def profile(self):
        try:return json.loads(self.profile_path.read_text())
        except (OSError,ValueError,TypeError):return {'enabled':False}

    def save(self,profile):
        required=('start_hour','end_hour','allowed_capabilities','allowed_objects','allowed_zones',
                  'max_attempts','max_continuous_motion_s','minimum_voltage_v','storage_quota_gb')
        if any(key not in profile for key in required):raise ValueError('Incomplete Autonomous Day profile')
        if not 0<=profile['start_hour']<=23 or not 1<=profile['end_hour']<=24 or profile['start_hour']>=profile['end_hour']:
            raise ValueError('Invalid Autonomous Day time window')
        if not 1<=profile['max_attempts']<=500 or not 10<=profile['max_continuous_motion_s']<=1800:
            raise ValueError('Invalid Autonomous Day budgets')
        value=dict(profile,schema_version=1,updated_at=time.time())
        temporary=self.profile_path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));temporary.replace(self.profile_path)
        return self.status()

    def active(self,profile,now=None):
        now=time.localtime() if now is None else now
        return profile.get('enabled') is True and profile.get('start_hour',24)<=now.tm_hour<profile.get('end_hour',0)

    def status(self):
        profile=self.profile()
        try:state=json.loads(self.status_path.read_text())
        except (OSError,ValueError,TypeError):state={}
        return dict(profile=profile,active=self.active(profile),runtime=state,persistent=True,
                    browser_required=False,stop_priority='absolute',manual_takeover='pause_and_reobserve')

    def loop(self):
        while not self.stop_event.wait(5):
            try:self.tick()
            except (OSError,ValueError,KeyError,TypeError) as exc:self.write({'activity':'pause','reason':str(exc)})

    def tick(self):
        profile=self.profile()
        if not self.active(profile):return self.write({'activity':'pause','reason':'profile_disabled_or_outside_window'})
        status=self.status_provider();resources=self.resource_provider();voltage=status.get('battery_voltage_v')
        if type(voltage) not in (int,float) or voltage<profile['minimum_voltage_v']:
            return self.write({'activity':'low_power_rest','reason':'minimum_voltage_reserve'})
        permission=dict(expires_at=time.time()+10,base_motion='EXPLORE_LOCAL' in profile['allowed_capabilities'],
                        allowed_capabilities=profile['allowed_capabilities'],
                        maximum_session_s=profile['max_continuous_motion_s'])
        decision=self.planner.choose(status,permission,self.curriculum_provider(),resources)
        decision['profile_updated_at']=profile.get('updated_at');decision['allowed_capabilities']=profile['allowed_capabilities']
        if self.dispatch is not None and decision['activity']!='pause' and time.monotonic()-self.last_dispatch>30:
            result=self.dispatch(decision,profile);decision['dispatch']=result;self.last_dispatch=time.monotonic()
        return self.write(decision)

    def write(self,value):
        record=dict(value,at=time.time());temporary=self.status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False));temporary.replace(self.status_path)
        return record
