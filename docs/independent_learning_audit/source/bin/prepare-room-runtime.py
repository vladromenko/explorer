#!/usr/bin/env python3
"""Install an INACTIVE room candidate with rollback; never commands motors."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def plan(candidate,root):
    from simulator.room_transfer import validate_room_folder
    from simulator.learning import sha256
    checked=validate_room_folder(candidate);root=Path(root);reasons=[]
    for name,digest in checked['runtime_files'].items():
        p=root/name
        if name.startswith('config/') and p.exists() and sha256(p)!=digest:reasons.append('Accepted geometry/config differs: '+name)
    for p in (Path(candidate)/'assets/meshes').glob('*'):
        installed=root/'vendor/m3_reference/meshes'/p.name
        if installed.exists() and sha256(installed)!=sha256(p):reasons.append('Existing robot CAD differs: '+p.name)
    return dict(candidate=str(Path(candidate).resolve()),root=str(root.resolve()),blocked_by=reasons,
        runtime_files=checked['runtime_files'],model_sha256=checked['bundle']['checkpoint_sha256']['model.safetensors'],
        simulation_ready=True,physical_ready=False,automatic_execution_allowed=False)


def apply(value):
    if sys.platform!='linux':raise ValueError('This inactive installer is for Jetson Linux only')
    if value['blocked_by']:raise ValueError('; '.join(value['blocked_by']))
    root,candidate=Path(value['root']),Path(value['candidate'])
    status=json.loads((root/'data/status.json').read_text())
    if (time.time()-status['at']>.5 or status.get('stop_latched') is not True or status.get('mission')
            or max(abs(x) for x in status.get('odom_velocity',[99]*3))>.005):
        raise ValueError('Inactive installation requires fresh stationary robot status, STOP latched, no mission')
    arm=root/'data/arm-state.json'
    if arm.exists() and json.loads(arm.read_text()).get('phase') in ('command_in_progress','EXECUTING'):
        raise ValueError('Arm is still executing')
    python=root/'.venv-learning/bin/python'
    if not python.is_file():raise ValueError('Existing checked ACT/CUDA environment is missing; preserve the robot environment')
    backup=root/'data/room-runtime-backups'/str(time.time_ns());backup.mkdir(parents=True)
    # Reuse the constrained installer, never upgrade the Jetson Torch/ROS stack.
    loader=importlib.util.spec_from_file_location('observed_prepare',root/'bin/prepare-observed-runtime.py')
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    (backup/'learning-env').mkdir();(backup/'vision-env').mkdir()
    module.checked_additions(python,['mujoco==3.14.0','gymnasium==1.4.0','scipy==1.18.1'],backup/'learning-env')
    vision=root/'data/env-vision';models=root/'data/observed-models'
    if not (vision/'bin/python').is_file():subprocess.run([str(python),'-m','venv',str(vision)],check=True)
    module.checked_additions(vision/'bin/python',['torch==2.11.0','transformers==4.57.3','psutil'],backup/'vision-env')
    download='''import hashlib,json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
lock=json.loads(Path(sys.argv[1]).read_text());target=Path(sys.argv[2])/'models/grounding-dino-tiny'
snapshot_download(lock['repository'],revision=lock['revision'],local_dir=target,allow_patterns=list(lock['files']),max_workers=2)
for name,digest in lock['files'].items():
 with (target/name).open('rb') as stream:
  if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Pinned vision model mismatch: '+name)
'''
    subprocess.run([str(vision/'bin/python'),'-c',download,str(candidate/'runtime/config/grounding-model.json'),str(models)],check=True)
    replaced=[];created=[]
    try:
        for name in value['runtime_files']:
            destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,saved);replaced.append(name)
            else:created.append(name)
            temporary=destination.with_suffix(destination.suffix+'.room-new');shutil.copyfile(candidate/'runtime'/name,temporary);temporary.replace(destination)
        for p in (candidate/'assets/meshes').iterdir():
            destination=root/'vendor/m3_reference/meshes'/p.name;destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copyfile(p,destination);created.append(str(destination.relative_to(root)))
        report=root/'data/room-learning-runtime.json'
        if report.exists():shutil.copy2(report,backup/'previous-runtime-report.json')
        subprocess.run([str(python),str(root/'bin/room-cleanup-worker.py'),str(candidate),'--inspect-output',str(report)],
            cwd=root,check=True,timeout=180)
    except Exception:
        for name in replaced:shutil.copy2(backup/name,root/name)
        for name in created:(root/name).unlink(missing_ok=True)
        raise
    value.update(source_installed=True,backup=str(backup),web_restart_required=True,physical_ready=False)
    (backup/'install.json').write_text(json.dumps(value,indent=2));return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('candidate',type=Path);parser.add_argument('--robot-root',type=Path,default=ROOT)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args();value=plan(args.candidate,args.robot_root)
    if args.apply:value=apply(value)
    print(json.dumps(value,ensure_ascii=False,indent=2))
