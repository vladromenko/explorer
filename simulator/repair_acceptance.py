"""Causal repair confirmation cannot override physics or final qualification."""
import math


def confirms_nonregression(rows,phase):
    for row in rows:
        credits=row.get('phase_credit',{})
        numbers=[row.get('score'),credits.get('pick'),credits.get(phase)]
        if (row.get('status')!='complete' or type(row.get('success')) is not bool or
            any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=100 for v in numbers)):
            return False,'invalid_confirmation_evidence'
    grouped={i:[r for r in rows if r.get('candidate')==i] for i in (0,1)}
    if len(rows)!=4 or any(len(r)!=2 for r in grouped.values()):return False,'incomplete_confirmation'
    seeds={r.get('seed') for r in grouped[0]}
    if len(seeds)!=2 or {r.get('seed') for r in grouped[1]}!=seeds:return False,'unpaired_confirmation'
    for seed in seeds:
        old=next(r for r in grouped[0] if r['seed']==seed);new=next(r for r in grouped[1] if r['seed']==seed)
        if old.get('success') is True and new.get('success') is not True:return False,'regressed_full_task'
        before,after=old.get('phase_credit',{}),new.get('phase_credit',{})
        if before.get('pick',0)>=9 and after.get('pick',0)<9:return False,'regressed_verified_pick'
        if after.get(phase,-1)<before.get(phase,0):return False,'regressed_blocked_phase'
    if min(r.get('score',-1) for r in grouped[1])<min(r.get('score',0) for r in grouped[0]):
        return False,'regressed_worst_task_outcome'
    return True,'replicated_on_new_training_scenes'
