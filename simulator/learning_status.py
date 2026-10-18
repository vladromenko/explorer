"""Small persisted learning summaries for the browser, independent of a session."""
from functools import lru_cache
import json
from pathlib import Path
import time

from .curriculum import task_spec


def compact_value(value):
    if isinstance(value,dict):
        return {key:compact_value(item) for key,item in value.items()
            if key not in ('summaries','history','training_seeds','checkpoint_sha256','model_checkpoint_sha256','source_checkpoint_sha256')}
    if isinstance(value,list):
        if value and all(isinstance(item,str) for item in value) and len(value)>40:return len(value)
        return [compact_value(item) for item in value]
    return value


@lru_cache(maxsize=4)
def demonstration_counts(root,interval,scene_profile='baseline'):
    output={}
    for path in (Path(root)/'demonstrations').glob('*/episode.json'):
        try:
            value=json.loads(path.read_text())
            task=value.get('task_id','sock_basket')
            matches=task not in ('sock_basket','cloth_basket') or value.get('scene_profile','baseline')==scene_profile
            finished=(value.get('attempt_completed') is True or value.get('state')=='complete' or
                      value.get('attempt_completed') is None and value.get('finished_at') is not None and
                      type(value.get('frames')) is int and value['frames']>0 and type(value.get('success')) is bool)
            if matches and finished:
                task=value.get('task_id','sock_basket');count=output.setdefault(task,dict(attempts=0,successful=0))
                count['attempts']+=1
                count['successful']+=int(value.get('physics_verified_success',value.get('success')) is True)
        except (OSError,ValueError):pass
    return output


def learning_status(root,snapshot):
    cycles=snapshot.get('learning_cycles',[]);batches=snapshot.get('night_batches',[])
    current=next((value for value in cycles if value['state']=='running'),cycles[0] if cycles else None)
    active=next((job for job in reversed(snapshot.get('jobs',[])) if job['status']=='running' and job['kind'] not in ('cycle','night','assistant')),None)
    plan=current or (batches[0] if batches else None)
    task_id=((active or {}).get('details') or {}).get('skill_id') or (plan or {}).get('task_id')
    if not task_id and plan and plan.get('tasks'):task_id=plan['tasks'][0]['task_id']
    task_id=task_id or 'sock_basket'
    task=next((item for item in (plan or {}).get('tasks',[]) if item['task_id']==task_id),{})
    model=snapshot.get('model') or {}
    previous=next((item.get('final_validate') or item.get('validate') for item in reversed(task.get('history',[]))
                   if item.get('final_validate') or item.get('validate')),{})
    report=task.get('validation') or task.get('stages',{}).get('final_validate') or task.get('stages',{}).get('validate') or previous or {}
    successes=report.get('successes',model.get('successes',0));episodes=report.get('episodes',model.get('episodes',0))
    qualified=current.get('state')=='ready_for_transfer' if current else task.get('simulation_qualified',False)
    status='running' if active or current and current['state']=='running' else 'ready_for_transfer' if qualified else 'needs_more_training'
    if current and current['state'] in ('paused','interrupted','failed','corrupt','needs_diagnosis'):status=current['state']
    details=(active or {}).get('details') or {}
    saved=task.get('stages',{}).get('train') or {}
    budget=details.get('training_budget') or saved.get('training_budget')
    output=dict(state=status,task_id=task_id,title=task_spec(task_id)['title'],cycle_id=current['id'] if current else None,
        plan_id=plan['id'] if plan else None,phase=active['kind'] if active else (current or {}).get('phase','finished'),
        message=(active or {}).get('message') or (current or {}).get('reason'),iteration=task.get('iteration',task.get('round',1)),
        progress=(active or {}).get('progress',0),steps_completed=details.get('steps_completed',saved.get('steps_completed')),
        training_budget=budget,eta_s=details.get('eta_s'),worker_progress=details if active and active['kind'] in ('audit','trials') else None,
        demonstrations=demonstration_counts(str(root),int(time.monotonic()//5),(plan or {}).get('options',{}).get('scene_profile','baseline')).get(task_id,dict(attempts=0,successful=0)),
        successes=successes,episodes=episodes,required_successes=10,simulation_ready=bool(qualified),physical_ready=False,
        model_id=task.get('model_id') or model.get('id'),archive=task.get('archive'),
        comparison=task.get('last_comparison'),
        teacher_profile=(plan or {}).get('options',{}).get('teacher_profile','legacy_v1'),
        resume_model_id=task.get('pending',{}).get('job_id') if task.get('pending',{}).get('stage')=='train' else None)
    from .neural_recovery import current as recovery_current
    recovery=recovery_current(root)
    if recovery and (recovery.get('alive') or recovery['cycle_id']==output['cycle_id']):
        state=recovery['state']
        if state in ('running','queued') and not recovery['alive']:state='interrupted'
        details=recovery.get('details',{})
        output.update(state=state,phase=recovery.get('phase','diagnosis'),message=recovery.get('reason') or recovery.get('message'),
            cycle_id=recovery['cycle_id'],model_id=Path(recovery['model']).name,
            iteration=recovery.get('iteration',1),external_recovery_id=recovery['id'],
            steps_completed=details.get('steps_completed'),training_budget=details.get('training_budget'),eta_s=details.get('eta_s'),
            progress=100*recovery.get('completed',0)/max(1,recovery.get('total',1)),
            simulation_ready=recovery.get('simulation_ready',False),archive=recovery.get('archive'),physical_ready=False,
            worker_progress={'active_workers':recovery.get('views',[])},
            recovery_counts={'completed':recovery.get('completed',0),'total':recovery.get('total',0)})
        if state=='pausing':output['message']='Останавливаем исполнителей и сохраняем контрольную точку; диск пока не извлекайте'
    return output
