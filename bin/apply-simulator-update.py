#!/usr/bin/env python3
"""Authorized program maintenance, preserving data and explicit operator pauses.

New versions drain a complete comparison. The first upgrade from older versions
uses their existing cooperative pause and preserves partial attempts as well.
Never controls the real robot or deletes training data.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from simulator.learning import write_json
from simulator.skill_training import identity
from simulator.repair_upgrade import actor_children

URL='http://127.0.0.1:8767'
DATA=Path('/Volumes/ExplorerSim/ExplorerSim')


def get(route):
    with urllib.request.urlopen(URL+route,timeout=10) as response:return json.load(response)


def post(route):
    token=get('/api/session')['token']
    request=urllib.request.Request(URL+route,data=b'{}',method='POST',headers={
        'Content-Type':'application/json','X-Explorer-Sim-Token':token})
    with urllib.request.urlopen(request,timeout=10) as response:return json.load(response)


def finish_legacy_pause(root,identifier,receipt,reason):
    """Only reinterpret the maintenance pause whose exact receipt we own."""
    request_path=root/'automatic-requests'/(identifier+'.json')
    queue_path=root/'automatic-requests/queue-pause.json'
    if not queue_path.is_file() or json.loads(queue_path.read_text())!=receipt:
        raise ValueError('Пауза пользователя изменилась; автоматическое продолжение запрещено')
    value=json.loads(request_path.read_text())
    if value['state']!='operator_paused':raise ValueError('Исполнители ещё не сохранили остановку')
    record=root/'maintenance'/('authorized-update-'+identifier+'.json')
    if not record.exists():write_json(record,dict(previous_request=value,pause_receipt=receipt,reason=reason,at=time.time()))
    value.update(state='maintenance_paused',reason=reason,pending_program_update=True,updated=time.time())
    value['events'].append(dict(stage='maintenance_paused',message=reason,at=time.time()))
    for task in value.get('tasks',[]):
        if task['state']!='simulation_qualified':task['state']='maintenance_paused'
    write_json(request_path,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--reason',required=True)
    options=parser.parse_args();identifier=identity(options.request)
    if not DATA.parent.is_mount():raise ValueError('External disk missing')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Save tested code in Git before installing')
    snapshot=get('/api/state')
    if snapshot.get('queue_pause'):raise ValueError('User paused the queue; not changing it')
    target=next(r for r in snapshot['requests'] if r['id']==identifier)
    if target['state'] in ('operator_paused','interrupted'):raise ValueError('User pause cannot be auto-resumed')
    lease=(DATA/'maintenance'/('update-'+identifier+'.lock')).open('a')
    with lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:post('/api/requests/'+identifier+'/prepare_update')
        except urllib.error.HTTPError as error:
            if error.code!=404:raise
            if target['state'] not in ('running','training','validating','waiting_for_memory'):
                raise ValueError('Old runtime is not in a maintainable running state')
            post('/api/requests/'+identifier+'/pause');receipt=get('/api/state')['queue_pause']
            if not receipt or receipt.get('owner_id')!=identifier:raise ValueError('Maintenance pause not acknowledged')
            deadline=time.monotonic()+45
            while True:
                s=get('/api/state');row=next(r for r in s['requests'] if r['id']==identifier)
                if s.get('queue_pause')!=receipt:raise ValueError('Operator changed pause during maintenance')
                installed=json.loads((DATA/'maintenance/automatic-install.json').read_text())
                if row['state']=='operator_paused' and not actor_children(installed['source']):break
                if time.monotonic()>deadline:raise RuntimeError('Workers did not drain; retained pause and data, no forced kill')
                time.sleep(.5)
            finish_legacy_pause(DATA,identifier,receipt,options.reason)
            if get('/api/state').get('queue_pause')!=receipt:raise ValueError('Operator pause changed before handoff')
            post('/api/queue/resume')
        subprocess.run([sys.executable,str(ROOT/'bin/stage-causal-repair.py'),'--request',identifier],cwd=ROOT,check=True)


if __name__=='__main__':main()
