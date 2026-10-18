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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training',default='1bd683a1d22b')
    parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'))
    parser.add_argument('--status',action='store_true');args=parser.parse_args()
    root=args.root
    if not root.parent.is_mount():raise ValueError('Connect the existing ExplorerSim disk')
    marker=root/'diagnostics/early-room-current.json'
    if args.status:
        value=json.loads(marker.read_text()) if marker.exists() else {'state':'not_launched'}
        path=Path(value['folder'])/'state.json' if 'folder' in value else None
        print(json.dumps(json.loads(path.read_text()) if path and path.exists() else value,ensure_ascii=False));return
    if not re.fullmatch('[0-9a-f]{12}',args.training):raise ValueError('Invalid training identity')
    marker.parent.mkdir(exist_ok=True)
    with (marker.parent/'early-room-launch.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if marker.exists():
            print(json.dumps(json.loads(marker.read_text()),ensure_ascii=False));return
        spec=importlib.util.spec_from_file_location('freeze_room_source',ROOT/'bin/queue-room-cleanup.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        revision,source=module.freeze_source()
        folder=root/'diagnostics'/('live-room-'+secrets.token_hex(6));folder.mkdir()
        env=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'src'),
            EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',EXPLORER_VISION_CPU_ONLY='1')
        with (folder/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-m','simulator.live_room_diagnostic',
                '--root',str(root),'--folder',str(folder),'--training',str(root/'runs'/args.training)],
                cwd=source,env=env,stdout=log,stderr=log,start_new_session=True)
        value=dict(folder=str(folder),pid=child.pid,source=str(source),revision=revision,
            diagnostic_only=True,simulation_qualified=False,current_learning_interrupted=False)
        write_json(marker,value);print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
