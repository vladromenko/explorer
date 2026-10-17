"""Measured parallelism for the observed learner, outside its policy/reward.

Replay identical TRAINING inputs, never holdouts, under the existing owner.
No second queue, reduced physics, shortened trajectory or learning credit.
"""
import copy
import json
from pathlib import Path
import secrets
import time

import psutil

from .resource_budget import memory_budget
from .worker_runtime import WorkerStalled

GIB=2**30


def apply_quality_scale(state,workers):
    """Extra capacity buys more diverse material within the same time limit."""
    if not state.get('scene_recipe'):return
    base=state.get('initial_attempt_budget',state['options']['attempts'])
    target=min(2000,max(base,int(base*workers/4)))
    state['quality_attempt_budget']=max(state.get('quality_attempt_budget',base),target)
    state['options']['attempts']=max(state['options']['attempts'],target)
    state['minimum_curriculum_pairs']=max(state.get('minimum_curriculum_pairs',16),int(16*workers/4))
    state['resource_objective']='more_diverse_full_fidelity_training_material'


def reservation(peak):
    # The model server remains a separate shared resident process. Actor peak
    # includes its renderer; add 25% and 256 MiB for future harder episodes.
    return max(int(1.5*GIB),int(peak*1.25+256*2**20))


def projected_workers(memory,peak,cores):
    spare=memory['available_bytes']-memory['total_bytes']*.15
    return max(1,min(8,cores,int(spare/reservation(peak))))


def signature(row):
    # Success counters alone cannot prove that parallelism preserved behavior.
    return tuple(json.dumps(row.get(k),sort_keys=True) for k in
        ('seed','candidate','status','success','phase_credit','failure_group','reason',
         'controller_program','domain_variation','trajectory_sha256'))


def assessment(baseline,probe,workers):
    a=baseline['results'];b=probe['results']
    order=lambda r:(r['candidate'],r['seed'])
    matched=(len(a)==len(b)==8 and all(r.get('trajectory_sha256') for r in a+b) and
        all(signature(x)==signature(y) for x,y in zip(sorted(a,key=order),sorted(b,key=order))))
    speedup=baseline['wall_s']/max(.001,probe['wall_s'])
    reasons=[]
    if not matched:reasons.append('Trajectories or full-fidelity outcomes differ')
    if probe['minimum_memory_fraction']<.15:reasons.append('Less than 15% memory headroom')
    if probe['actual_peak_workers']!=workers:reasons.append('Requested parallelism was not reached')
    if speedup<1.2:reasons.append('Less than 20% measured speedup')
    return dict(accepted=not reasons,reasons=reasons,workers=workers,speedup=speedup,
        exact_trajectory_equivalence=bool(matched),minimum_memory_fraction=probe['minimum_memory_fraction'])


def calibrate(trainer,state,specs,cancelled,progress):
    """One finite, isolated replay experiment after a warm full training batch."""
    if (state['options']['workers']!=4 or state.get('capacity_calibration',{}).get('state') in ('completed','not_useful') or
            len(specs)!=8 or any(s.get('split','train')!='train' for s in specs) or
            (trainer.folder/'update.request').exists() or trainer.pause.exists()):return
    measured=[r.get('peak_rss_bytes',0) for r in state['episodes'] if r.get('status')=='complete']
    peak=max(measured,default=0)
    if not peak:return # Older evidence without real residency measurements cannot size new workers.
    memory=memory_budget();cores=psutil.cpu_count(logical=False) or 1
    target=projected_workers(memory,peak,cores)
    if target<=4:
        state['capacity_calibration']=dict(state='waiting_for_headroom',projected_workers=target,
            memory=memory,measured_actor_peak_bytes=peak,reason='Four workers retained until RAM permits a comparison')
        trainer.save(state);return
    token=secrets.token_hex(6) # A paused or repeated probe never overwrites earlier measurements.
    root=trainer.folder/'capacity-calibration'/token
    results=[]
    for workers in (4,target):
        if cancelled() or trainer.pause.exists():raise InterruptedError('Пауза измерения производительности')
        memory=memory_budget()
        if projected_workers(memory,peak,cores)<workers:
            state['capacity_calibration']=dict(state='waiting_for_headroom',projected_workers=target,
                reason='Memory changed before replay; no extra actors started')
            trainer.save(state);return
        run=type(trainer)(trainer.root,trainer.id)
        run.folder=root/str(workers);run.folder.mkdir(parents=True,exist_ok=True)
        run.path=run.folder/'state.json';run.pause=trainer.pause
        bench=copy.deepcopy(state)
        bench.update(episodes=[],completed_attempts=0,active_workers=[],effective_workers=workers,
            options=dict(state['options'],workers=workers),state='running',
            measured_worker_reservation_bytes=reservation(peak),capacity_replay=True,
            minimum_memory_fraction=1.,actual_peak_workers=0,infrastructure_events=[])
        inputs=[dict(s,episode_id=secrets.token_hex(6),split='capacity_benchmark') for s in specs]
        for s in inputs:s['folder']=str(run.folder/'episodes'/s['episode_id'])
        state.update(capacity_calibration=dict(state='measuring',workers=workers,folder=str(run.folder)))
        trainer.save(state)
        def update(current):
            state['active_workers']=current['active_workers']
            trainer.save(state)
            progress(dict(state,phase_group='capacity_benchmark'))
        started=time.monotonic()
        try:rows=run._batch_once(bench,inputs,cancelled,update)
        except WorkerStalled as error:
            state.update(active_workers=[],calibrated_workers=4,effective_workers=4,
                capacity_calibration=dict(state='not_useful',reason=str(error),folder=str(root),
                    excluded_from_learning_and_acceptance=True))
            trainer.save(state);return
        result=dict(results=rows,wall_s=time.monotonic()-started,
            minimum_memory_fraction=bench['minimum_memory_fraction'],actual_peak_workers=bench['actual_peak_workers'])
        results.append(result)
        (run.folder/'measurement.json').write_text(json.dumps(result,indent=2))
    checked=assessment(*results,target)
    state.update(active_workers=[],measured_worker_reservation_bytes=reservation(peak),
        calibrated_workers=target if checked['accepted'] else 4,
        effective_workers=target if checked['accepted'] else 4,
        capacity_calibration=dict(state='completed',**checked,folder=str(root),
            excluded_from_learning_and_acceptance=True,baseline_wall_s=results[0]['wall_s'],probe_wall_s=results[1]['wall_s']))
    apply_quality_scale(state,state['calibrated_workers'])
    trainer.save(state)
