#!/usr/bin/env python3
"""Measure four/eight workers and start only after the existing queue is idle."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from simulator.parallel_trials import benchmark,save

URL='http://127.0.0.1:8766'
SUPPORT=Path.home()/'Library/Application Support/ExplorerSimulator'
DATA=Path('/Volumes/ExplorerSim/ExplorerSim')


def get(path):
    with urllib.request.urlopen(URL+path,timeout=10) as stream:
        return json.load(stream)


def transition_ready(state,plan_id):
    plan=next((p for p in state.get('night_batches',[]) if p['id']==plan_id),None)
    if plan is None:raise ValueError('Original queue not found')
    active=[j for j in state.get('jobs',[]) if j.get('status')=='running']
    view=state.get('state') or {}
    command=view.get('command',[0.]*9)
    target=view.get('target',command)
    moving=(any(abs(float(value))>1e-4 for value in command[6:]) or
            any(abs(float(a)-float(b))>.1 for a,b in zip(command[:6],target[:6])))
    return (plan.get('state')=='complete' and not active and not state.get('recording') and not moving),plan


def installed_server_command(command,expected_python):
    # macOS ps displays the executable path without quoting embedded spaces.
    # Match that entire path before parsing only the remaining arguments.
    prefix=expected_python+' '
    if not command.startswith(prefix):return False
    try:arguments=shlex.split(command[len(prefix):])
    except ValueError:return False
    return arguments[:2]==['-m','simulator.server']


def run(plan_id,dry_run=False):
    if len(plan_id)!=12 or any(c not in '0123456789abcdef' for c in plan_id):
        raise ValueError('Invalid original queue identity')
    snapshot=get('/api/state')
    ready,plan=transition_ready(snapshot,plan_id)
    marker=DATA/f'parallel-transition-{plan_id}.json'
    if marker.is_file():
        saved=json.loads(marker.read_text())
        if saved.get('state') in ('launched','capacity_failed'):
            return saved
        next_plan=next((p for p in snapshot.get('night_batches',[]) if p['id']==saved.get('next_plan_id')),None)
        if next_plan and next_plan.get('state') in ('running','complete','paused'):
            result=dict(saved,state='launched',recovered=True,workers=next_plan.get('options',{}).get('workers',4))
            if not dry_run:save(marker,result)
            return result
    if not ready:return dict(state='waiting',original_plan=plan_id,message='Current round and final evaluation remain active')
    if dry_run:return dict(state='ready',original_plan=plan_id)
    if not DATA.is_dir():raise ValueError('External training disk is unavailable')
    report_path=DATA/'parallel-capacity.json'
    if marker.is_file() and report_path.is_file():
        report=json.loads(report_path.read_text())
    else:
        if report_path.exists():
            report_path=DATA/f'parallel-capacity-{uuid.uuid4().hex[:8]}.json'
        probe_root=DATA/'benchmarks'/f'four-workers-{uuid.uuid4().hex[:8]}'
        os.environ.setdefault('EXPLORER_SIM_MESH_DIR','/Volumes/ExplorerSim/assets/m3_reference/meshes')
        report=benchmark(probe_root,report_path)
        save(DATA/'parallel-capacity.json',report)
        save(marker,dict(state='benchmarked',original_plan=plan_id,capacity_report=str(report_path)))
    if report['assessment']['fits'] is not True:
        outcome=dict(state='capacity_failed',original_plan=plan_id,assessment=report['assessment'])
        save(marker,outcome)
        return outcome
    workers=report['assessment']['workers']
    ready,_=transition_ready(get('/api/state'),plan_id)
    if not ready:return dict(state='waiting',message='New work was started during benchmark; no restart')
    # Atomically install just simulator source/contract/docs. Never run the app
    # installer here: it boots out services even when they are still training.
    for relative in ('simulator/server.py','simulator/curriculum.py','simulator/learning.py',
                     'simulator/transfer.py','simulator/parallel_trials.py','simulator/feedback_audit.py',
                     'src/mobile_policy_contract.py','docs/SIMULATOR-AND-TRANSFER.ru.md'):
        target=SUPPORT/'source'/relative
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
        shutil.copy2(REPO/relative,temporary)
        temporary.replace(target)
    pid_text=subprocess.check_output(['/usr/sbin/lsof','-n','-P','-iTCP:8766','-sTCP:LISTEN','-t'],text=True)
    pids=set(pid_text.split())
    if len(pids)!=1:raise RuntimeError('Cannot identify the single local simulator owner')
    pid=int(next(iter(pids)))
    command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip()
    expected=str(SUPPORT/'runtime/bin/python')
    if not installed_server_command(command,expected):
        raise RuntimeError('Port owner is not the installed Explorer simulator')
    # This second guard prevents stopping a newly launched learner or recording.
    if not transition_ready(get('/api/state'),plan_id)[0]:
        return dict(state='waiting',message='Queue became busy; no restart')
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+35
    stopped=False
    while time.monotonic()<deadline and not stopped:
        try:os.kill(pid,0)
        except ProcessLookupError:stopped=True
        if not stopped:time.sleep(.5)
    if not stopped:raise RuntimeError('Idle simulator has not stopped; another owner was not launched')
    with (SUPPORT/'service.log').open('ab') as out,(SUPPORT/'service-error.log').open('ab') as err:
        subprocess.Popen(['/bin/bash',str(SUPPORT/'source/bin/start-simulator.command')],
                         stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    deadline=time.monotonic()+60
    state=None
    while time.monotonic()<deadline and state is None:
        try:state=get('/api/state')
        except OSError:time.sleep(.5)
    if state is None:raise RuntimeError('Updated simulator did not become available')
    if any(j.get('status')=='running' for j in state.get('jobs',[])):
        raise RuntimeError('Another job started; no duplicate learning queue')
    # Bind the launch to a durable identity. If the HTTP response is lost, the
    # next call recovers by this plan ID rather than creating a second queue.
    from simulator.curriculum import NightQueue
    intent=json.loads(marker.read_text())
    if not intent.get('next_plan_id'):
        next_plan=NightQueue(DATA).create([t['task_id'] for t in plan['tasks']],
            count=100,steps=30000,batch_size=16,hours=24,seed=plan['options']['seed']+10000000,
            rounds=1,scene_profile=plan['options'].get('scene_profile','baseline'),workers=workers)
        intent['next_plan_id']=next_plan['id']
        save(marker,intent)
    token=get('/api/session')['token']
    request=urllib.request.Request(URL+'/api/jobs/night',method='POST',
        data=json.dumps({'plan_id':intent['next_plan_id'],'workers':workers}).encode(),
        headers={'Content-Type':'application/json','X-Explorer-Sim-Token':token})
    with urllib.request.urlopen(request,timeout=15) as stream:job=json.load(stream)
    result=dict(state='launched',original_plan=plan_id,next_plan_id=intent['next_plan_id'],
                job_id=job['id'],workers=workers,assessment=report['assessment'])
    save(marker,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.plan,args.dry_run),ensure_ascii=False))


if __name__=='__main__':main()
