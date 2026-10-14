"""Independent full-physics tuning scenes; this module cannot qualify a policy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .learning import write_json,read_bundle
from .parallel_trials import memory_free_percent, stop_children

MODES=('queue20','queue10','queue5','ensemble1')


def choose_mode(rows,baseline='queue20'):
    counts={mode:sum(row['success'] is True for row in rows if row['mode']==mode) for mode in MODES}
    errors={mode:sum(bool(row.get('error')) for row in rows if row['mode']==mode) for mode in MODES}
    complete={mode:sum(row['mode']==mode for row in rows) for mode in MODES}
    if len(set(complete.values()))!=1 or not next(iter(complete.values())):
        raise ValueError('Every tuning mode must have the same scenes')
    eligible=[mode for mode in MODES if not errors[mode]]
    if not eligible:raise ValueError('All tuning modes had execution errors')
    best=max(eligible,key=lambda mode:(counts[mode],mode==baseline,-MODES.index(mode)))
    return dict(mode=best,successes=counts[best],episodes=complete[best],counts=counts,
                diagnostic_only=True,qualification=False)


def audit(model,seeds,folder,workers=4,cancelled=None,progress=None):
    if workers not in (1,4,8) or not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError('Invalid tuning workers or duplicate scenes')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    bundle,_=read_bundle(model)
    pending=list(enumerate(seeds));active={};results=[];started=time.monotonic();sampled=0.
    try:
        while pending or active:
            if cancelled and cancelled():raise InterruptedError('Проверка обратной связи приостановлена')
            if time.monotonic()-sampled>=2:
                sampled=time.monotonic()
                if memory_free_percent()<10:raise InterruptedError('Недостаточно свободной памяти для проверки')
            while pending and len(active)<workers:
                index,seed=pending.pop(0);output=folder/f'{index:04d}.json'
                if output.is_file():
                    saved=json.loads(output.read_text())
                    if (saved.get('running') is False and saved.get('seeds')==[seed] and len(saved.get('results',[]))==len(MODES)
                            and saved.get('source_checkpoint_sha256')==bundle['checkpoint_sha256']):
                        results.extend(saved['results'])
                    else:
                        output.rename(folder/f'{index:04d}-interrupted-{time.time_ns()}.json')
                        pending.insert(0,(index,seed))
                else:
                    log=(folder/f'{index:04d}.log').open('ab')
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
                    try:
                        child=subprocess.Popen([sys.executable,'-m','simulator.feedback_audit','--model',str(model),
                            '--output',str(output),'--seeds',str(seed),'--modes',*MODES],stdout=log,stderr=log,
                            env=env,start_new_session=True)
                    except Exception:
                        log.close();raise
                    slot=next(slot for slot in range(workers) if slot not in {item['slot'] for item in active.values()})
                    active[index]=dict(child=child,log=log,output=output,started=time.monotonic(),slot=slot)
            for index,item in list(active.items()):
                code=item['child'].poll()
                if code is not None:
                    item['log'].close()
                    if code!=0:raise RuntimeError(f'Ошибка проверки; журнал {folder}/{index:04d}.log')
                    saved=json.loads(item['output'].read_text())
                    if saved.get('running') or len(saved['results'])!=len(MODES):raise RuntimeError('Tuning evidence is incomplete')
                    results.extend(saved['results']);del active[index]
                elif time.monotonic()-item['started']>1800:raise TimeoutError('Tuning scene exceeded 30 minutes')
            status=dict(completed=len(results),total=len(seeds)*len(MODES),workers=workers,wall_s=time.monotonic()-started,
                active_workers=[dict(index=index,slot=item['slot'],pid=item['child'].pid) for index,item in active.items()])
            write_json(folder/'progress.json',status)
            if progress:progress(status)
            if pending or active:time.sleep(.2)
    finally:
        stop_children(active)
    decision=choose_mode(results)
    report=dict(model_id=Path(model).name,seeds=seeds,results=results,decision=decision,
                diagnostic_only=True,qualification=False,physics_reduced=False,wall_s=time.monotonic()-started)
    write_json(folder/'result.json',report)
    return report
