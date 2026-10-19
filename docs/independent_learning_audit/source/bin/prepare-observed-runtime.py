#!/usr/bin/env python3
"""Prepare an INACTIVE checked skill on Jetson; never commands any motor.

Default is inspection. --apply installs checked source with a rollback copy,
adds missing dependencies without upgrading unrelated installed packages,
and fetches the exact pinned RGB model. It never clears STOP or starts a trial.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def plan(candidate,root):
    from simulator.observed_transfer import validate,sha
    candidate=Path(candidate).resolve();root=Path(root).resolve();package=candidate/'package';checked=validate(package)
    reasons=[];copies=[]
    for relative,digest in checked['qualification']['runtime_files'].items():
        source=package/'runtime'/relative;target=root/relative
        if relative.startswith('config/') and target.is_file():
            if relative in checked['qualification']['controller']['files'] and sha(target)!=digest:
                reasons.append('Accepted calibration/geometry differs: '+relative)
            continue
        copies.append(relative)
    return dict(candidate=str(candidate),root=str(root),source_files=copies,blocked_by=reasons,
        simulation_ready=True,physical_ready=False,automatic_execution_allowed=False,
        required_python=['mujoco==3.14.0','gymnasium==1.4.0','scipy==1.18.1'])


def checked_additions(python,requirements,folder):
    # Refuse replacement of ROS/OpenCV or existing numerical packages. If the
    # constraints cannot be met, preserve the environment and explain why.
    script='import importlib.metadata,json;print(json.dumps({d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}))'
    installed=json.loads(subprocess.check_output([str(python),'-c',script],text=True))
    constraints=folder/'installed-constraints.txt'
    constraints.write_text('\n'.join(name+'=='+version for name,version in installed.items())+'\n')
    subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--constraint',str(constraints),*requirements],check=True)


def apply(value):
    if sys.platform!='linux':raise ValueError('This installer is for the robot Linux environment; Mac training is separate')
    if value['blocked_by']:raise ValueError('; '.join(value['blocked_by']))
    root=Path(value['root']);candidate=Path(value['candidate']);package=candidate/'package'
    status=json.loads((root/'data/status.json').read_text())
    if (time.time()-status.get('at',0)>.5 or status.get('stop_latched') is not True or status.get('mission') or
        max(abs(float(v)) for v in status.get('odom_velocity',[99,99,99]))>.005):
        raise ValueError('Installation requires fresh stationary robot status, STOP latched, and no mission')
    arm=root/'data/arm-state.json'
    if arm.is_file() and json.loads(arm.read_text()).get('phase')=='command_in_progress':raise ValueError('Arm still executing')
    backup=root/'data/observed-runtime-backups'/str(time.time_ns());backup.mkdir(parents=True)
    python=root/'.venv/bin/python'
    if not python.is_file():raise ValueError('Existing robot web Python is missing')
    checked_additions(python,value['required_python'],backup)
    vision=root/'data/env-vision';model_root=root/'data/observed-models'
    if not (vision/'bin/python').is_file():subprocess.run([str(python),'-m','venv',str(vision)],check=True)
    subprocess.run([str(vision/'bin/python'),'-m','pip','install','--disable-pip-version-check',
        'torch==2.11.0','transformers==4.57.3','psutil'],check=True)
    script='''import hashlib,json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
lock=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);target=root/'models/grounding-dino-tiny'
snapshot_download(lock['repository'],revision=lock['revision'],local_dir=target,allow_patterns=list(lock['files']),max_workers=2)
for n,d in lock['files'].items():
 with (target/n).open('rb') as f:
  if hashlib.file_digest(f,'sha256').hexdigest()!=d:raise ValueError('Model checksum: '+n)
'''
    subprocess.run([str(vision/'bin/python'),'-c',script,str(package/'runtime/config/grounding-model.json'),str(model_root)],check=True)
    replaced=[];new=[]
    try:
        for relative in value['source_files']:
            source=package/'runtime'/relative;target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                copy=backup/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,copy);replaced.append(relative)
            else:new.append(relative)
            temporary=target.with_suffix(target.suffix+'.observed-new');shutil.copyfile(source,temporary);temporary.replace(target)
        output=root/'vendor/m3_reference/meshes';output.mkdir(parents=True,exist_ok=True)
        for path in (package/'assets/meshes').iterdir():
            destination=output/path.name
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Existing CAD differs; retained: '+path.name)
            if not destination.exists():shutil.copyfile(path,destination)
        subprocess.run([str(python),'-c','import mujoco,gymnasium,scipy;from simulator.observed_robot import ObservedRobotEnv;from src.observed_skill_execution import ObservedSkillExecution'],cwd=root,check=True)
    except Exception:
        for relative in replaced:shutil.copy2(backup/relative,root/relative)
        for relative in new:(root/relative).unlink(missing_ok=True)
        raise
    value.update(source_installed=True,backup=str(backup),web_restart_required=True,physical_ready=False,
        next_step='Restart explorer-web.service while stopped, then open /simulation-training and run preflight')
    (backup/'install.json').write_text(json.dumps(value,indent=2));return value


def main():
    p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--robot-root',type=Path,default=ROOT);p.add_argument('--apply',action='store_true');a=p.parse_args()
    value=plan(a.candidate,a.robot_root)
    if a.apply:value=apply(value)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
