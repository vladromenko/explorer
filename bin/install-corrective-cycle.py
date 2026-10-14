#!/usr/bin/env python3
"""Install one committed simulator fix after the current learner finishes.

Only local simulator processes are controlled. No robot or training checkpoint
is modified. A durable marker and PID lock prevent duplicate installations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/Volumes/ExplorerSim/ExplorerSim')
SUPPORT=Path.home()/'Library/Application Support/ExplorerSimulator'
URL='http://127.0.0.1:8766'


def get(path):
    with urllib.request.urlopen(URL+path,timeout=10) as response:return json.load(response)


def post(path,value):
    token=get('/api/session')['token']
    request=urllib.request.Request(URL+path,method='POST',data=json.dumps(value).encode(),
        headers={'Content-Type':'application/json','X-Explorer-Sim-Token':token})
    with urllib.request.urlopen(request,timeout=15) as response:return json.load(response)


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));temporary.replace(path)


def boundary(snapshot,cycle,train_id,trained_state=None):
    if cycle.get('pause_kind')=='operator':return 'operator_paused'
    if cycle.get('state')=='ready_for_transfer':return 'already_qualified'
    job=next((job for job in snapshot.get('jobs',[]) if job['id']==train_id),None)
    if job and job['status'] in ('failed','cancelled'):return 'training_failed'
    recorded=any(task.get('model_id')==train_id and task.get('pending',{}).get('stage')!='train'
                 for task in cycle.get('tasks',[]))
    done=(job and job['status']=='complete') or (not job and trained_state in ('validated_simulation','trained_simulation_unvalidated'))
    if done and recorded:return 'ready'
    return 'waiting_for_training'


def training_state(train_id):
    path=DATA/'runs'/train_id/'job.json'
    return json.loads(path.read_text()).get('state') if path.is_file() else None


def teacher_evidence(profile):
    locations={
        'planned_grasp_v3':DATA/'diagnostics/teacher-close-20261014/planned-paired.json',
        'planned_release_v4':DATA/'diagnostics/teacher-release-independent-20261014/paired.json',
    }
    path=locations[profile]
    value=json.loads(path.read_text())
    rows=value['results'];by_seed={}
    for row in rows:
        by_seed.setdefault(row['seed'],{})[row['profile']]=row
    if (len(rows)!=20 or len(by_seed)!=10 or any(set(pair)!={'legacy_v1',profile} for pair in by_seed.values())
        or not value.get('diagnostic_only') or value.get('qualification') or not value.get('physics_unchanged')):
        raise ValueError('Corrective teacher comparison is incomplete')
    original=sum(pair['legacy_v1']['success'] for pair in by_seed.values())
    planned=sum(pair[profile]['success'] for pair in by_seed.values())
    def acceptable(row):
        error=row.get('error')
        return not error or (row.get('success') is False and row.get('outcome',{}).get('phase')=='lift'
            and error=='Контактное удержание при подъёме не подтверждено')
    regressions=sum(pair['legacy_v1']['success'] and not pair[profile]['success'] for pair in by_seed.values())
    if planned<=original or regressions or any(not acceptable(pair[profile]) for pair in by_seed.values()):
        raise ValueError('Corrective teacher did not improve real recorded episodes')
    for seed,pair in by_seed.items():
        for profile,row in pair.items():
            episode=path.parent/f'{profile}-{seed}'/'episode.json'
            saved=json.loads(episode.read_text())
            if (saved.get('seed')!=seed or saved.get('teacher_profile')!=profile
                or saved.get('success')!=row['success'] or saved.get('attempt_completed') is not True):
                raise ValueError('Corrective teacher evidence differs from saved physical episode')
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original=original,planned=planned,
        episodes=10,regressions=regressions,teacher_profile=profile)


def simulator_pids():
    processes=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    entries=[line.strip().split(None,1) for line in processes if line.strip()]
    return [int(parts[0]) for parts in entries if len(parts)==2 and parts[1].endswith('-m simulator.server --port 8766')]


def start_service(source):
    with (SUPPORT/'service.log').open('ab') as log,(SUPPORT/'service-error.log').open('ab') as errors:
        return subprocess.Popen(['/bin/bash',str(source/'bin/start-simulator.command')],cwd=source,
            stdout=log,stderr=errors,start_new_session=True)


def wait_service(seconds=60):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        try:get('/api/state?compact=true');return True
        except OSError:time.sleep(1)
    return False


def install(cycle_id,train_id,reference,revision,watch=False,teacher_profile='planned_grasp_v3'):
    if not Path('/Volumes/ExplorerSim').is_mount():raise ValueError('ExplorerSim disk is not mounted')
    marker=DATA/'maintenance'/f'corrective-{cycle_id}.json';marker.parent.mkdir(parents=True,exist_ok=True)
    if marker.exists() and json.loads(marker.read_text()).get('state')=='installed':return {'state':'already_installed'}
    lock=marker.with_suffix('.pid')
    if lock.exists():
        try:os.kill(int(lock.read_text()),0);return {'state':'installer_already_running'}
        except (ProcessLookupError,ValueError):lock.unlink()
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:return {'state':'installer_already_running'}
    with os.fdopen(fd,'w') as handle:handle.write(str(os.getpid()))
    state=dict(state='waiting_for_training',cycle_id=cycle_id,train_id=train_id,reference_model_id=reference,revision=revision)
    changed=[];cycle_backup=None;server_stopped=False
    save(marker,state)
    try:
        snapshot=get('/api/state?compact=true')
        cycle=json.loads((DATA/'learning-cycles'/(cycle_id+'.json')).read_text())
        decision=boundary(snapshot,cycle,train_id,training_state(train_id))
        while watch and decision=='waiting_for_training':
            time.sleep(15)
            snapshot=get('/api/state?compact=true')
            cycle=json.loads((DATA/'learning-cycles'/(cycle_id+'.json')).read_text())
            decision=boundary(snapshot,cycle,train_id,training_state(train_id))
        if decision!='ready':state['state']=decision;save(marker,state);return state
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if actual!=revision or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise ValueError('Install requires the exact tested clean Git revision')
        evidence=teacher_evidence(teacher_profile)
        state['teacher_evidence']=evidence
        def following_training():
            return any(job['status']=='running' and job.get('kind')=='train' and job['id']!=train_id
                for job in snapshot.get('jobs',[]))
        while following_training():
            state['state']='waiting_for_following_training';save(marker,state)
            if not watch:return state
            time.sleep(15)
            snapshot=get('/api/state?compact=true')
            cycle=json.loads((DATA/'learning-cycles'/(cycle_id+'.json')).read_text())
            if cycle.get('pause_kind')=='operator':
                state['state']='operator_paused';save(marker,state);return state
            if cycle.get('state')=='ready_for_transfer':
                state['state']='already_qualified';save(marker,state);return state
        state.update(state='stopping_following_stage');save(marker,state)
        if any(job['status']=='running' for job in snapshot.get('jobs',[])):
            post('/api/cancel',{})
        deadline=time.monotonic()+90
        while any(job['status']=='running' for job in get('/api/state?compact=true').get('jobs',[])):
            if time.monotonic()>deadline:raise TimeoutError('Following stage did not stop; original service retained')
            time.sleep(1)
        # The target learner must still be complete, not merely at a checkpoint.
        trained=json.loads((DATA/'runs'/train_id/'job.json').read_text())
        if trained['state'] not in ('validated_simulation','trained_simulation_unvalidated'):
            raise ValueError('Current training has not completed; refusing restart')
        cycle=json.loads((DATA/'learning-cycles'/(cycle_id+'.json')).read_text())
        if cycle.get('pause_kind')=='operator':
            state['state']='operator_paused';save(marker,state);return state
        servers=simulator_pids()
        if len(servers)!=1:raise ValueError('Expected exactly one local simulator server')
        os.kill(servers[0],signal.SIGTERM)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            try:os.kill(servers[0],0)
            except ProcessLookupError:break
            time.sleep(.5)
        else:raise TimeoutError('Original service did not shut down cleanly')
        server_stopped=True
        cycle_path=DATA/'learning-cycles'/(cycle_id+'.json')
        cycle_backup=marker.parent/f'{cycle_id}-before-correction.json'
        shutil.copy2(cycle_path,cycle_backup)
        cycle=json.loads(cycle_path.read_text())
        for task in cycle['tasks']:
            if task.get('model_id')==train_id:
                task['reference_model_id']=reference
                task['corrective_teacher_evidence_sha256']=evidence['sha256']
                if task.get('pending',{}).get('stage') in ('audit','validate','final_validate','compare','trials','export'):
                    task.pop('pending',None)
        cycle['options']['teacher_profile']=teacher_profile
        cycle.update(state='interrupted',pause_kind='maintenance',reason='Установлено сравнение моделей и исправление подготовки показов')
        save(cycle_path,cycle)
        source=SUPPORT/'source';backup=marker.parent/('source-'+revision[:12]);backup.mkdir(exist_ok=True)
        for directory in ('simulator','bin','docs'):
            for path in (ROOT/directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and 'learning-live' not in path.parts:
                    relative=path.relative_to(ROOT);target=source/relative
                    if target.exists():
                        previous=backup/relative;previous.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,previous)
                    else:previous=None
                    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.installing')
                    shutil.copy2(path,temporary);temporary.replace(target);changed.append((target,previous))
        (source/'installed-training-revision').write_text(revision+'\n')
        process=start_service(source)
        if not wait_service():raise TimeoutError('Updated simulator did not become available')
        job=post('/api/jobs/cycle',{'cycle_id':cycle_id})
        state.update(state='installed',service_pid=process.pid,job_id=job['id'],finished_at=time.time())
        save(marker,state);return state
    except Exception as error:
        rollback_error=None
        if server_stopped:
            try:
                for pid in simulator_pids():os.kill(pid,signal.SIGTERM)
                time.sleep(2)
                for target,previous in reversed(changed):
                    if previous:shutil.copy2(previous,target)
                    else:target.unlink(missing_ok=True)
                if cycle_backup and cycle_backup.is_file():shutil.copy2(cycle_backup,DATA/'learning-cycles'/(cycle_id+'.json'))
                start_service(SUPPORT/'source')
                if not wait_service():raise TimeoutError('Original simulator service could not restart')
            except Exception as failure:rollback_error=str(failure)
        state.update(state='failed',error=str(error),rollback_error=rollback_error)
        save(marker,state);raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle',required=True);parser.add_argument('--after-train',required=True)
    parser.add_argument('--reference',required=True);parser.add_argument('--revision',required=True)
    parser.add_argument('--teacher-profile',choices=('planned_grasp_v3','planned_release_v4'),default='planned_grasp_v3')
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    for value in (args.cycle,args.after_train,args.reference):
        if len(value)!=12 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid job identity')
    print(json.dumps(install(args.cycle,args.after_train,args.reference,args.revision,args.watch,args.teacher_profile),ensure_ascii=False))
