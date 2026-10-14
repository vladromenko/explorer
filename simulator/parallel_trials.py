"""Independent full-fidelity teacher processes; no hardware interfaces."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid


def save(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temporary.replace(path)


def worker(spec_path):
    # Import the environment in the child's main thread. Each process owns its
    # MuJoCo data, OpenGL context, output folder and random scene seed.
    from .environment import ExplorerEnv
    spec=json.loads(Path(spec_path).read_text())
    env=ExplorerEnv()
    started=time.monotonic()
    try:
        episode=env.make_episode_trial(spec['folder'],seed=spec['seed'],
            task_id=spec['task_id'],scene_profile=spec['scene_profile'])
        if episode.get('attempt_completed') is not True:
            raise RuntimeError('Worker did not finish its attempt')
        # The fingerprint lets a benchmark compare actual issued trajectories,
        # not just success counters; wall-clock timestamps intentionally differ.
        import hashlib
        digest=hashlib.sha256()
        rows=Path(spec['folder'])/'samples.jsonl'
        for line in rows.read_text().splitlines():
            row=json.loads(line)
            digest.update(json.dumps({k:row.get(k) for k in
                ('observation','action','simulation_time','phase','simulated_outcome')},sort_keys=True).encode())
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes=peak if sys.platform=='darwin' else peak*1024
        save(spec['result'],dict(episode=episode,wall_s=time.monotonic()-started,
            peak_rss_bytes=peak_bytes,trajectory_sha256=digest.hexdigest(),pid=os.getpid()))
    finally:
        env.close()


def stop_children(active):
    for item in active.values():
        child=item['child']
        if child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline=time.monotonic()+5
    for item in active.values():
        child=item['child']
        try:
            child.wait(timeout=max(.01,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        item['log'].close()


def collect(root, job_id, count, first_seed, task_id='sock_basket', scene_profile='sock_variants_v1',
            workers=4, recovered=None, cancelled=None, progress=None, attempt_timeout_s=900, seeds=None):
    if type(workers) is not int or workers not in (1,4,8):
        raise ValueError('Choose one, four or eight independent workers')
    if type(count) is not int or not 1<=count<=1000 or type(first_seed) is not int or first_seed<0:
        raise ValueError('Invalid parallel attempt range')
    seeds=list(seeds) if seeds is not None else list(range(first_seed,first_seed+count))
    if len(seeds)!=count or len(set(seeds))!=count or any(type(seed) is not int or not 0<=seed<2**31 for seed in seeds):
        raise ValueError('Choose distinct valid scene seeds for every attempt')
    if not job_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in job_id):
        raise ValueError('Invalid parallel job identity')
    from .curriculum import task_spec
    task_spec(task_id)
    if scene_profile not in ('baseline','sock_variants_v1'):
        raise ValueError('Unknown scene profile')
    root=Path(root).resolve()
    manifests=dict(recovered or {})
    if any(type(index) is not int or not 0<=index<count for index in manifests):
        raise ValueError('Recovered attempt index is outside this job')
    coordination=root/'parallel-jobs'/job_id
    coordination.mkdir(parents=True,exist_ok=False)
    pending=[index for index in range(count) if index not in manifests]
    active={}
    metadata={}
    started=time.monotonic()
    try:
        while pending or active:
            if cancelled and cancelled():
                raise InterruptedError('Parallel generation cancelled')
            while pending and len(active)<workers:
                import shutil
                if shutil.disk_usage(root).free<8*1024**3:
                    raise InterruptedError('Less than 8 GiB free; generation paused')
                index=pending.pop(0)
                folder=root/'demonstrations'/f'trial-{job_id}-{index:04d}'
                if folder.exists():
                    raise ValueError('Do not overwrite an existing attempt')
                result=coordination/f'{index:04d}-result.json'
                spec=coordination/f'{index:04d}-request.json'
                save(spec,dict(folder=str(folder),result=str(result),seed=seeds[index],
                               task_id=task_id,scene_profile=scene_profile))
                log=(coordination/f'{index:04d}.log').open('wb')
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                         MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONUNBUFFERED='1')
                try:
                    child=subprocess.Popen([sys.executable,'-m','simulator.parallel_trials','worker',str(spec)],
                        stdout=log,stderr=log,env=env,start_new_session=True)
                except Exception:
                    log.close()
                    raise
                active[index]=dict(child=child,log=log,result=result,started=time.monotonic())
            for index,item in list(active.items()):
                code=item['child'].poll()
                if code is not None:
                    item['log'].close()
                    if code!=0 or not item['result'].is_file():
                        raise RuntimeError(f'Worker {index} failed; see {coordination}/{index:04d}.log')
                    output=json.loads(item['result'].read_text())
                    episode=output['episode']
                    if episode.get('seed')!=seeds[index] or episode.get('attempt_completed') is not True:
                        raise RuntimeError('Worker result does not match its assigned attempt')
                    manifests[index]=episode
                    metadata[index]={k:output[k] for k in ('wall_s','peak_rss_bytes','trajectory_sha256','pid')}
                    del active[index]
                elif time.monotonic()-item['started']>attempt_timeout_s:
                    raise TimeoutError(f'Parallel attempt {index} exceeded its time budget')
            status=dict(workers=workers,completed=len(manifests),total=count,
                        active_attempts=[index+1 for index in active],wall_s=time.monotonic()-started)
            save(coordination/'progress.json',status)
            if progress:
                progress(status)
            if pending or active:
                time.sleep(.1)
    finally:
        stop_children(active)
    result=dict(workers=workers,wall_s=time.monotonic()-started,
                episodes=[manifests[index] for index in range(count)],
                measurements={str(index):value for index,value in metadata.items()})
    save(coordination/'result.json',result)
    return result


def memory_free_percent():
    import re
    query=subprocess.run(['/usr/bin/memory_pressure','-Q'],capture_output=True,text=True,check=True)
    match=re.search(r'System-wide memory free percentage:\s*(\d+)%',query.stdout)
    if not match:
        raise RuntimeError('Cannot measure macOS memory pressure')
    return int(match.group(1))


def assess(serial, parallel, total_memory, cpu_count, minimum_free_percent,workers=4):
    serial_rows,parallel_rows=serial['episodes'],parallel['episodes']
    matched=(len(serial_rows)==len(parallel_rows)>=workers and all(
        a['seed']==b['seed'] and a['success']==b['success'] and a.get('error')==b.get('error')
        and a['frames']==b['frames'] and serial['measurements'][str(index)]['trajectory_sha256']==
            parallel['measurements'][str(index)]['trajectory_sha256']
        for index,(a,b) in enumerate(zip(serial_rows,parallel_rows))))
    # Count only simultaneous workers; eight attempts on four workers have two
    # waves, not eight simultaneous resident processes.
    peaks=sorted((item['peak_rss_bytes'] for item in parallel['measurements'].values()),reverse=True)
    peak_sum=sum(peaks[:workers])
    speedup=serial['wall_s']/parallel['wall_s']
    reasons=[]
    if not matched:reasons.append('Physical trajectories differ between serial and parallel execution')
    if any(row.get('error') and 'Численная нестабильность' in row['error'] for row in parallel_rows):
        reasons.append('Physics instability')
    if cpu_count<workers:reasons.append('Fewer processor cores than workers')
    share=.35 if workers==4 else .55
    if peak_sum>total_memory*share:reasons.append('Workers exceed their RAM budget')
    if minimum_free_percent<15:reasons.append('Insufficient memory headroom during benchmark')
    if speedup<1.3:reasons.append('Less than 1.3x measured generation speedup')
    return dict(fits=not reasons,reasons=reasons,workers=workers,trajectory_equivalence=matched,
                speedup=speedup,worker_peak_rss_sum_bytes=peak_sum,
                total_memory_bytes=total_memory,cpu_count=cpu_count,
                minimum_free_percent=minimum_free_percent)


def eight_projection(four, baseline_free_percent, current_free_percent, cpu_count):
    # Doubling the measured loss of headroom is conservative when mesh pages
    # are shared, and prevents starting eight processes into known pressure.
    projected=min(current_free_percent,baseline_free_percent)-2*max(0,baseline_free_percent-four['minimum_free_percent'])
    return dict(projected_free_percent=projected,
                can_probe=four['fits'] and projected>=15 and cpu_count>=8)


def choose_workers(four,eight):
    if (eight and eight.get('fits') is True and four['fits'] and
            eight['speedup']/four['speedup']>=1.2):
        return eight
    return four


def benchmark(root, report_path, seed=92010001):
    # Caller must wait for the active round; this benchmark does not share the
    # server's environment, datasets or final unseen qualification scenes.
    root,report_path=Path(root),Path(report_path)
    if report_path.exists():raise ValueError('Preserve the original resource evidence')
    root.mkdir(parents=True,exist_ok=True)
    serial=collect(root,'benchmark-serial',8,seed,workers=1)
    def measured_collect(workers,name):
        free=[memory_free_percent()]
        last_sample=[-2.]
        def measure(status):
            if status['wall_s']-last_sample[0]>=2.:
                free.append(memory_free_percent())
                last_sample[0]=status['wall_s']
                if free[-1]<10:
                    raise InterruptedError('Memory headroom below 10%; benchmark workers stopped')
        return collect(root,name,8,seed,workers=workers,progress=measure),free
    baseline_free=memory_free_percent()
    try:
        parallel,free=measured_collect(4,'benchmark-four')
    except InterruptedError as exc:
        report=dict(assessment={'fits':False,'workers':4,'reasons':[str(exc)]},
                    serial=serial,created=time.time(),benchmark_only=True,
                    physics_reduced=False,automatic_hardware_execution=False)
        save(report_path,report)
        return report
    memory=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    cpus=int(subprocess.check_output(['sysctl','-n','hw.physicalcpu'],text=True))
    four=assess(serial,parallel,memory,cpus,min(free))
    projection=eight_projection(four,baseline_free,memory_free_percent(),cpus)
    eight=None
    eight_run=None
    if projection['can_probe']:
        try:
            eight_run,eight_free=measured_collect(8,'benchmark-eight')
            eight=assess(serial,eight_run,memory,cpus,min(eight_free),workers=8)
        except InterruptedError as exc:
            eight=dict(fits=False,workers=8,reasons=[str(exc)])
    else:
        eight=dict(fits=False,workers=8,reasons=['Insufficient projected memory headroom or failed four-worker baseline'])
    report=dict(serial=serial,parallel=parallel,eight_parallel=eight_run,
                four_assessment=four,eight_assessment=eight,eight_projection=projection,
                assessment=choose_workers(four,eight),
                created=time.time(),benchmark_only=True,physics_reduced=False,
                image_capture_and_recording=True,automatic_hardware_execution=False)
    save(report_path,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    w=commands.add_parser('worker');w.add_argument('spec',type=Path)
    b=commands.add_parser('benchmark');b.add_argument('--root',type=Path,required=True)
    b.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='worker':worker(args.spec)
    else:print(json.dumps(benchmark(args.root,args.report)['assessment'],ensure_ascii=False))


if __name__=='__main__':main()
