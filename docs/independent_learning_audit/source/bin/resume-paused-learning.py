#!/usr/bin/env python3
"""Explicit operator resume on a checked revision, retaining every prior file."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulator.learning import write_json,prepared_training_available


def validate_states(local,room):
    if local.get('state')!='paused' or local.get('active_pids') or local.get('alive'):
        raise ValueError('Resume needs a fully stopped operator-paused local owner')
    if room.get('state')!='operator_paused' or room.get('active_pids') or room.get('alive'):
        raise ValueError('Resume needs the existing fully stopped room cycle')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'));args=parser.parse_args()
    root=args.root
    if not root.parent.is_mount():raise ValueError('Connect ExplorerSim first')
    import psutil
    from simulator.neural_recovery import current as local_current
    from simulator.room_control import current as room_current,control
    local,room=local_current(root),room_current(root)
    if not local or not room:raise ValueError('Existing saved local and room cycles are required')
    validate_states(local,room)
    modules={'simulator.neural_recovery','simulator.train_entry','lerobot.scripts.lerobot_train',
             'simulator.room_validation','simulator.live_room_diagnostic','simulator.checkpoint_monitor'}
    for process in psutil.process_iter(['cmdline']):
        command=process.info['cmdline'] or []
        if '-m' in command and command[command.index('-m')+1] in modules:
            raise ValueError('A remaining compute owner exists; no second writer may start')
    if psutil.virtual_memory().available/psutil.virtual_memory().total<.25:
        raise ValueError('Resume needs at least 25% available unified memory')
    for path in (root/'skill-training/owner.lock',root/'neural-recovery/owner.lock',root/'room-learning'/room['id']/'owner.lock'):
        with path.open('a') as lease:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    training=root/'runs'/local['training_model_id']
    plan=dict(local_id=local['id'],room_id=room['id'],training_id=training.name,
        prepared_export_reuse=prepared_training_available(training),new_queue=False,files_deleted=False)
    if not args.apply:print(json.dumps(plan));return
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit checked code before resuming')
    maintenance=root/'maintenance';maintenance.mkdir(exist_ok=True)
    with (maintenance/'explicit-resume.lock').open('a') as operation:
        fcntl.flock(operation,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([sys.executable,str(ROOT/'bin/install-neural-launcher.py')],check=True,capture_output=True,text=True)
        installed=json.loads((Path.home()/'Library/Application Support/ExplorerSimulator/neural-launch.json').read_text())
        spec=importlib.util.spec_from_file_location('resume_source',ROOT/'bin/queue-room-cleanup.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        revision,room_source=module.freeze_source()
        if installed['revision']!=revision:raise ValueError('Installed revisions differ')
        validate_states(local_current(root),room_current(root))
        marker=root/'room-learning/current.json';old_marker=json.loads(marker.read_text())
        receipt=maintenance/('explicit-resume-'+str(time.time_ns())+'.json')
        queue=root/'automatic-requests/queue-pause.json'
        write_json(receipt,dict(local=local,room=room,room_marker=old_marker,
            automatic_pause=json.loads(queue.read_text()) if queue.exists() else None,
            plan=plan,revision=revision,at=time.time()))
        saved={k:v for k,v in local.items() if k!='alive'}
        saved.setdefault('code_upgrades',[]).append(dict(previous=saved.get('runtime_source'),
            next=installed['source'],reason='Explicit resume with verified prepared-export reuse and command refusal evidence',receipt=str(receipt)))
        saved.update(runtime_source=installed['source'],code_revision=revision)
        write_json(root/'neural-recovery'/local['id']/'state.json',saved)
        write_json(marker,dict(old_marker,source=str(room_source),code_revision=revision))
        result=control(root,'resume')
        write_json(receipt.with_name(receipt.stem+'-launched.json'),dict(plan=plan,result=result,at=time.time()))
        print(json.dumps(dict(plan,result=result,receipt=str(receipt))))


if __name__=='__main__':main()
