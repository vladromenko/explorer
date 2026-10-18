#!/usr/bin/env python3
"""One durable full-room follow-up; leaves current ACT training untouched."""
import argparse
import fcntl
import io
import json
import os
import re
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulator.learning import write_json,sha256


def alive(value):
    import psutil
    try:
        args=psutil.Process(value['pid']).cmdline()
        return 'simulator.room_validation' in args and value['id'] in args
    except (psutil.Error,KeyError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'))
    parser.add_argument('--status',action='store_true');parser.add_argument('--pause',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.root;marker=root/'room-learning/current.json'
    if not args.status and not root.parent.is_mount():raise ValueError('Connect the existing ExplorerSim disk')
    queue_lease=None
    if not args.status:
        marker.parent.mkdir(parents=True,exist_ok=True);queue_lease=(marker.parent/'queue.lock').open('a')
        fcntl.flock(queue_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads(marker.read_text()) if marker.exists() else None
    if old and not re.fullmatch('[0-9a-f]{12}',old['id']):raise ValueError('Invalid existing room cycle marker')
    folder=root/'room-learning'/old['id'] if old else None
    state=json.loads((folder/'state.json').read_text()) if folder and (folder/'state.json').exists() else old
    if args.status:print(json.dumps(state or dict(state='not_queued'),ensure_ascii=False));return
    if args.pause:
        if folder is None:raise ValueError('No existing full-room cycle')
        (folder/'pause.request').touch();print(json.dumps(dict(pause_requested=True,id=old['id'])));return
    if state and alive(state):print(json.dumps(state,ensure_ascii=False));return
    if state and not args.resume:
        print(json.dumps(state,ensure_ascii=False));return
    if not root.parent.is_mount():raise ValueError('Connect the existing ExplorerSim disk')
    if args.resume:
        if state is None or state.get('state') not in ('operator_paused','waiting_for_local_learning','running'):
            raise ValueError('Only a paused or interrupted room cycle can be resumed; diagnose failed results first')
        import psutil
        for process in psutil.process_iter(['cmdline']):
            try:cmd=process.info['cmdline'] or []
            except (psutil.NoSuchProcess,psutil.AccessDenied):continue
            if 'simulator.room_validation' in cmd and any(mode in cmd for mode in ('trial','worker')) and any(str(folder) in arg for arg in cmd):
                raise ValueError('A remaining room worker still owns this cycle; preserve its files')
        (folder/'pause.request').unlink(missing_ok=True);identifier=old['id'];state['state']='queued';write_json(folder/'state.json',state)
    else:
        identifier=secrets.token_hex(6);folder=root/'room-learning'/identifier;folder.mkdir(parents=True)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Save checked code in Git before freezing a background cycle')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    support=Path.home()/'Library/Application Support/ExplorerSimulator';source=support/('room-source-'+revision[:12])
    if not source.exists():
        archive=subprocess.check_output(['git','archive',revision,'simulator','src','config','bin','docs'],cwd=ROOT)
        with tempfile.TemporaryDirectory(prefix='room-freeze-',dir=support) as temporary:
            stage=Path(temporary)/'source';stage.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive)) as stream:stream.extractall(stage,filter='data')
            from simulator.room_policy import RUNTIME_FILES
            write_json(stage/'source-revision.json',dict(revision=revision,files={name:sha256(stage/name) for name in RUNTIME_FILES}));stage.rename(source)
    env=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'src'),
        EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (folder/'background.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-m','simulator.room_validation','--root',str(root),'--id',identifier],
            cwd=source,env=env,stdout=log,stderr=log,start_new_session=True)
    write_json(marker,dict(id=identifier,pid=child.pid,code_revision=revision,source=str(source)))
    print(json.dumps(dict(id=identifier,pid=child.pid,state='queued',current_learning_interrupted=False,physical_ready=False)))


if __name__=='__main__':main()
