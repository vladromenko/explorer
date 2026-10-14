#!/usr/bin/env python3
"""Read/resume one authorized simulator cycle; never connect to a real robot."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulator.parallel_trials import memory_free_percent
from simulator.learning import read_bundle, sha256
from src.simulation_task_contract import simulation_qualification
from simulator.curriculum import report_passes

URL='http://127.0.0.1:8766'
DATA=Path('/Volumes/ExplorerSim/ExplorerSim')


def get(path):
    with urllib.request.urlopen(URL+path,timeout=10) as response:return json.load(response)


def check(identifier,resume=False):
    if len(identifier)!=12 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid cycle ID')
    if not DATA.is_dir():return dict(state='disk_missing',action_required=True)
    try:snapshot=get('/api/state?compact=true')
    except OSError:return dict(state='service_unavailable',action_required=True)
    path=DATA/'learning-cycles'/(identifier+'.json');cycle=json.loads(path.read_text())
    active=[job for job in snapshot.get('jobs',[]) if job['status']=='running']
    if cycle['state']=='ready_for_transfer':
        checks=[]
        for task in cycle['tasks']:
            folder=DATA/'runs'/task['model_id'];bundle,_=read_bundle(folder)
            gate=simulation_qualification(folder,bundle)
            paired=task.get('stages',{})
            first=paired.get('validate',{});final=paired.get('final_validate',{})
            different=bool(first.get('seeds')) and set(first['seeds']).isdisjoint(final.get('seeds',[]))
            weights=bundle['checkpoint_sha256']['model.safetensors']
            same_weights=first.get('model_weights_sha256')==final.get('model_weights_sha256')==weights
            verified=gate['simulation_qualified'] and report_passes(first) and report_passes(final) and different and same_weights
            checks.append(dict(task_id=task['task_id'],verified=verified,qualification=gate))
        archive=cycle.get('archive') or {}
        if not archive and active:return dict(state='running',phase='export',active_jobs=len(active))
        archive_path=Path(archive.get('path','')).resolve()
        archive_verified=archive_path.is_relative_to(DATA/'exports') and archive_path.is_file() and sha256(archive_path)==archive.get('sha256')
        return dict(state='ready_for_transfer' if archive_verified and checks and all(row['verified'] for row in checks) else 'evidence_failed',
                    checks=checks,archive=archive,archive_verified=archive_verified,physical_ready=False)
    if active:return dict(state='running',phase=cycle.get('phase'),active_jobs=len(active))
    if cycle['state']=='paused' and cycle.get('pause_kind')=='operator':return dict(state='operator_paused')
    if cycle['state']=='failed':return dict(state='failed',reason=cycle.get('reason'))
    if cycle['state'] not in ('queued','paused','interrupted'):return dict(state=cycle['state'])
    # A killed UI service can leave a learner/scene process finishing its work.
    # Never reopen the same checkpoint while that owner is still alive.
    commands=subprocess.check_output(['ps','-axo','command'],text=True).splitlines()
    pending=[task.get('pending',{}).get('job_id') for task in cycle['tasks']]
    for command in commands:
        is_worker=('-m lerobot.scripts.lerobot_train ' in command or '-m simulator.train_entry ' in command
            or '-m simulator.feedback_audit ' in command or '-m simulator.parallel_trials worker ' in command)
        if is_worker and any(identifier and identifier in command for identifier in pending):
            return dict(state='waiting_for_existing_worker')
    free=memory_free_percent()
    if free<25:return dict(state='waiting_for_memory',minimum_free_percent=free)
    if not resume:return dict(state='resumable',phase=cycle.get('phase'))
    token=get('/api/session')['token']
    request=urllib.request.Request(URL+'/api/jobs/cycle',method='POST',
        data=json.dumps({'cycle_id':identifier}).encode(),headers={'Content-Type':'application/json','X-Explorer-Sim-Token':token})
    with urllib.request.urlopen(request,timeout=15) as response:job=json.load(response)
    return dict(state='resumed',cycle_id=identifier,job_id=job['id'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cycle',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();print(json.dumps(check(args.cycle,args.resume),ensure_ascii=False))
