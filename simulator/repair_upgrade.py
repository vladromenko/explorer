"""One-shot, disk-aware handoff after the CURRENT learner releases ownership.

This is part of the installed training program, not a Codex reminder. While
workers run it only reads completed reports. It never pauses a training job,
clears an operator pause, sends robot commands, or kills a physics child.
"""
import argparse
import fcntl
import hashlib
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

import psutil

from .failure_repair import diagnosis,program,repair_proposals
from .learning import write_json

URL='http://127.0.0.1:8767'
ACTIVE={'queued','running','planning','building_scene','training','validating','packaging',
        'waiting_for_current','waiting_for_memory','pausing'}


def get(route):
    with urllib.request.urlopen(URL+route,timeout=5) as response:return json.load(response)


def decision(snapshot,identifier):
    if snapshot.get('queue_pause'):return 'operator_pause'
    target=next((r for r in snapshot.get('requests',[]) if r['id']==identifier),None)
    if target is None:return 'missing_request'
    if any(r['state'] in ACTIVE for r in snapshot['requests']):return 'active'
    if any(r.get('active_workers') for r in snapshot.get('trainings',[])):return 'active'
    if target['state'] in ('operator_paused','interrupted'):return 'operator_pause'
    if target['state']=='memory_paused':
        memory=snapshot.get('memory',{})
        total=memory.get('total_bytes',0);available=memory.get('available_bytes',0)
        return 'ready' if total>0 and available/total>=.25 else 'waiting_for_memory'
    if target['state']=='simulation_qualified':return 'qualified'
    if target['state'] in ('budget_paused','needs_diagnosis','validation_failed','failed','maintenance_paused'):return 'ready'
    return 'active'


def source_verified(config):
    source=Path(config['source']).resolve()
    for name,digest in config['files'].items():
        file=(source/name).resolve()
        if not file.is_relative_to(source) or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('Staged source checksum mismatch: '+name)
    return source


def actor_children(source):
    source=str(Path(source).resolve());actors=[]
    for p in psutil.process_iter(['pid','cmdline']):
        args=' '.join(p.info['cmdline'] or [])
        if not any(name in args for name in ('multiprocessing.spawn','simulator-grounding.py','simulator.worker_runtime')):continue
        try:cwd=p.cwd()
        except (psutil.Error,OSError):continue
        if cwd==source:actors.append(p.info['pid'])
    return actors


def launch_service(source,support):
    source=Path(source);support=Path(support)
    environment=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'src'),
        EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes',
        HF_HOME='/Volumes/ExplorerSim/cache/huggingface',HF_HUB_DISABLE_TELEMETRY='1',
        PYTORCH_ENABLE_MPS_FALLBACK='1',TOKENIZERS_PARALLELISM='false')
    with (support/'automatic-service.log').open('ab') as log,(support/'automatic-service-error.log').open('ab') as errors:
        service=subprocess.Popen([str(support/'runtime/bin/python'),'-m','simulator.automatic_server'],
            cwd=source,env=environment,stdout=log,stderr=errors,start_new_session=True)
    try:
        for _ in range(120):
            try:get('/api/session');return service
            except OSError:
                if service.poll() is not None:raise RuntimeError('Service exited before startup')
                time.sleep(.5)
        raise RuntimeError('Service did not answer')
    except BaseException:
        # Called only while holding the physical owner lease, so this newly
        # created service cannot have acquired workers or received continuation.
        if service.poll() is None:
            service.terminate()
            try:service.wait(timeout=10)
            except subprocess.TimeoutExpired:raise RuntimeError('Unresponsive service retained; no forced kill')
        raise


def activate(config,marker_path):
    root=Path(config['root']);source=source_verified(config);support=Path(config['support'])
    if not root.parent.is_mount():return None
    snapshot=get('/api/state');state=decision(snapshot,config['request'])
    if state not in ('ready','qualified'):return None
    old=json.loads((root/'maintenance/automatic-install.json').read_text())
    if actor_children(old['source']):return None
    with (root/'skill-training/owner.lock').open('a') as owner:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return None
        # Recheck after claiming physical ownership; a new user job takes
        # precedence over this upgrade. No pause/cancel endpoint is used.
        if decision(get('/api/state'),config['request']) not in ('ready','qualified'):return None
        old_process=psutil.Process(old['pid'])
        if ('simulator.automatic_server' not in old_process.cmdline() or
            Path(old_process.cwd()).resolve()!=Path(old['source']).resolve()):raise ValueError('Service identity changed')
        backup=root/'maintenance'/('before-causal-repair-'+config['revision'][:12]);backup.mkdir(exist_ok=True)
        shutil.copy2(root/'maintenance/automatic-install.json',backup/'automatic-install.json')
        app=Path.home()/'Applications/Explorer Simulator.app/Contents/MacOS/launch'
        if app.is_file():shutil.copy2(app,backup/'launch')
        write_json(marker_path,dict(config,state='activating',previous_source=old['source'],at=time.time()))
        from .semantic_cache import stop_servers
        if stop_servers(root):raise RuntimeError('Idle vision workers did not exit; update retained for diagnosis')
        old_process.send_signal(signal.SIGTERM)
        for _ in range(100):
            try:get('/api/session')
            except OSError:break
            time.sleep(.1)
        else:raise RuntimeError('Idle service did not stop; no escalation performed')
        try:service=launch_service(source,support)
        except Exception as error:
            restored=launch_service(old['source'],support)
            write_json(root/'maintenance/automatic-install.json',dict(old,pid=restored.pid,
                rollback_after_failed_upgrade=True,at=time.time()))
            raise RuntimeError('New service rejected; previous service restored: '+str(error)) from error
        write_json(root/'maintenance/automatic-install.json',dict(state='installed',revision=config['revision'],
            source=str(source),pid=service.pid,files=config['files'],backup=str(backup),url=URL+'/',at=time.time(),
            legacy_service_restarted=False,legacy_actor_modified=False))
        launcher='''#!/bin/bash
set -euo pipefail
if ! mount | /usr/bin/grep -Fq ' on /Volumes/ExplorerSim ('; then
 /usr/bin/osascript -e 'display alert "Подключите диск ExplorerSim"'
 exit 0
fi
if ! /usr/bin/curl --fail --silent --max-time 1 {url}/api/session >/dev/null; then
 cd {source}
 export EXPLORER_SIM_MESH_DIR=/Volumes/ExplorerSim/assets/m3_reference/meshes
 export PYTHONPATH={pythonpath}
 nohup {python} -m simulator.automatic_server >>{log} 2>>{errors} </dev/null &
fi
for attempt in {{1..90}}; do
 if /usr/bin/curl --fail --silent --max-time 1 {url}/api/session >/dev/null; then
  open '{url}/'
  exit 0
 fi
 sleep 0.5
done
'''.format(url=URL,source=shlex.quote(str(source)),pythonpath=shlex.quote(str(source)+':'+str(source/'src')),
            python=shlex.quote(str(support/'runtime/bin/python')),log=shlex.quote(str(support/'automatic-service.log')),
            errors=shlex.quote(str(support/'automatic-service-error.log')))
        temporary=app.with_suffix('.tmp');temporary.write_text(launcher);temporary.chmod(0o755);temporary.replace(app)
    # Release the physics lease BEFORE the new service receives a continuation.
    if state=='qualified':return dict(state='qualified',request=config['request'],no_duplicate_training=True)
    latest=get('/api/state')
    if latest.get('queue_pause'):return dict(state='operator_pause',reason='Operator paused during upgrade; not resumed')
    token=get('/api/session')['token']
    request=urllib.request.Request(URL+'/api/requests/'+config['request']+'/resume',data=b'{}',method='POST',
        headers={'Content-Type':'application/json','X-Explorer-Sim-Token':token})
    with urllib.request.urlopen(request,timeout=10) as response:result=json.load(response)
    return dict(state='launched',request=result['id'],parent=config['request'],pid=service.pid,
        old_episodes_counted_for_qualification=False)


def run(marker):
    marker=Path(marker);config=json.loads(marker.read_text())
    with marker.with_suffix('.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if config.get('state') in ('launched','qualified','failed'):return
        root=Path(config['root']);last=-1
        while True:
            if not root.parent.is_mount():
                write_json(marker,dict(config,state='waiting_for_disk',at=time.time()));time.sleep(20);continue
            try:
                snapshot=get('/api/state');next_step=decision(snapshot,config['request'])
                row=next((r for r in snapshot['requests'] if r['id']==config['request']),{})
                if row.get('superseded_by'):
                    installed=json.loads((root/'maintenance/automatic-install.json').read_text())
                    if installed.get('revision')==config['revision']:
                        write_json(marker,dict(config,state='launched',request=row['superseded_by'],
                            parent=config['request'],recovered_idempotent_handoff=True,at=time.time()));return
                rows=[]
                for task in row.get('tasks',[]):
                    path=root/'skill-training'/task['training_id']/'state.json'
                    if path.is_file():rows+=json.loads(path.read_text()).get('episodes',[])
                count=len(rows)
                if count!=last:
                    analysis=diagnosis(rows[-128:],root)
                    memory_path=root/'maintenance/historical-joint-memory-sock_basket.json'
                    memory={}
                    if memory_path.is_file():
                        from .experience_memory import checked_memory
                        memory=checked_memory(memory_path)
                    analysis['pending_programs']=repair_proposals(dict(controller_program=program(),experience=memory),analysis)
                    analysis['physical_tests_deferred_until_owner_free']=True
                    write_json(root/'maintenance'/('causal-diagnosis-'+config['request']+'.json'),analysis);last=count
                if next_step in ('ready','qualified'):
                    result=activate(config,marker)
                    if result:
                        write_json(marker,dict(config,**result,at=time.time()));return
                write_json(marker,dict(config,state='waiting_for_active' if next_step=='active' else next_step,
                    observed_completed=count,at=time.time()))
            except (OSError,ValueError,KeyError,RuntimeError,psutil.Error) as error:
                write_json(marker,dict(config,state='failed',reason=str(error),at=time.time()));return
            time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('marker',type=Path)
    run(parser.parse_args().marker)
