#!/usr/bin/env python3
"""Launch one bounded read-only-checkpoint diagnostic, not a training queue."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulator.learning import write_json


def validate_repair_retry(previous,state,revision,training):
    if (state.get('state') not in ('failed','complete') or state.get('mission_success') is True
            or previous['revision']==revision or Path(state['training']).name!=training):
        raise ValueError('Retry needs a finished failed diagnostic, same training job and a checked new code revision')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training',default='1bd683a1d22b')
    parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'))
    parser.add_argument('--status',action='store_true');parser.add_argument('--retry-repaired',action='store_true')
    parser.add_argument('--model',help='Finished immutable model bundle to test instead of the previous snapshot')
    args=parser.parse_args()
    root=args.root
    if not root.parent.is_mount():raise ValueError('Connect the existing ExplorerSim disk')
    marker=root/'diagnostics/early-room-current.json'
    if args.status:
        value=json.loads(marker.read_text()) if marker.exists() else {'state':'not_launched'}
        path=Path(value['folder'])/'state.json' if 'folder' in value else None
        print(json.dumps(json.loads(path.read_text()) if path and path.exists() else value,ensure_ascii=False));return
    if not re.fullmatch('[0-9a-f]{12}',args.training):raise ValueError('Invalid training identity')
    model_source=None
    if args.model:
        from simulator.learning import read_bundle
        if not re.fullmatch('[0-9a-f]{12}',args.model):raise ValueError('Invalid model identity')
        model_source=root/'runs'/args.model
        read_bundle(model_source)
        job=json.loads((model_source/'job.json').read_text())
        if job.get('state') not in ('validated_simulation','trained_simulation_unvalidated'):
            raise ValueError('Only a finished immutable bundle can be evaluated')
        if job.get('source_training_job',args.training)!=args.training:
            raise ValueError('Model lineage differs from the diagnostic training identity')
    marker.parent.mkdir(exist_ok=True)
    with (marker.parent/'early-room-launch.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if marker.exists() and not args.retry_repaired:
            print(json.dumps(json.loads(marker.read_text()),ensure_ascii=False));return
        spec=importlib.util.spec_from_file_location('freeze_room_source',ROOT/'bin/queue-room-cleanup.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        revision,source=module.freeze_source()
        previous=None
        if args.retry_repaired:
            import psutil
            previous=json.loads(marker.read_text());state=json.loads((Path(previous['folder'])/'state.json').read_text())
            validate_repair_retry(previous,state,revision,args.training)
            if psutil.pid_exists(previous['pid']):raise ValueError('Previous diagnostic owner is still present; no duplicate')
        folder=root/'diagnostics'/('live-room-'+secrets.token_hex(6));folder.mkdir()
        env=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'src'),
            EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',EXPLORER_VISION_CPU_ONLY='1')
        with (folder/'supervisor.log').open('ab') as log:
            command=[sys.executable,'-m','simulator.live_room_diagnostic',
                '--root',str(root),'--folder',str(folder),'--training',str(root/'runs'/args.training)]
            if previous:command+=['--snapshot-copy',str(Path(previous['folder'])/'model')]
            if model_source:command+=['--model-source',str(model_source)]
            child=subprocess.Popen(command,
                cwd=source,env=env,stdout=log,stderr=log,start_new_session=True)
        value=dict(folder=str(folder),pid=child.pid,source=str(source),revision=revision,
            diagnostic_only=True,simulation_qualified=False,current_learning_interrupted=False)
        if previous:value['previous_diagnostic']=previous
        write_json(marker,value);print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
