"""Bounded activity selection for one explicitly permitted autonomous session."""
import time


class AutonomousDayPlanner:
    def choose(self, status, permission, curriculum, resources):
        if permission.get('expires_at',0)<=time.time():return dict(activity='pause',reason='permission_expired')
        if status.get('manual_takeover') or status.get('stop_latched'):return dict(activity='pause',reason='manual_or_stop')
        if resources.get('disk_free_gb',0)<2:return dict(activity='ask_help',reason='storage_reserve_low',question='Освободите 2 ГБ на Jetson')
        if status.get('localization_valid') is not True:
            return dict(activity='relocalize',reason='pose_unknown',budget_s=120)
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
