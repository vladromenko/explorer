#!/usr/bin/env python3
"""Freeze a checked revision and attach a one-shot idle-only upgrade worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
SUPPORT=Path.home()/'Library/Application Support/ExplorerSimulator'


def main():
    from simulator.skill_training import identity
    from simulator.learning import write_json
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    options=parser.parse_args();identifier=identity(options.request)
    data=Path('/Volumes/ExplorerSim/ExplorerSim')
    if not data.parent.is_mount():raise ValueError('External disk missing')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Save tested code in Git first')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    marker=SUPPORT/('causal-upgrade-'+identifier+'.json')
    if marker.is_file():
        old=json.loads(marker.read_text())
        if old.get('state') in ('launched','qualified'):print(json.dumps(old));return
        if old.get('revision')!=revision:raise ValueError('Another staged revision exists; inspect before replacing')
        pid_path=marker.with_suffix('.pid.json')
        pid=json.loads(pid_path.read_text()).get('pid') if pid_path.is_file() else None
        if pid:
            import psutil
            try:
                args=psutil.Process(pid).cmdline()
                if 'simulator.repair_upgrade' in args and str(marker) in args:
                    print(json.dumps(dict(state=old.get('state'),watcher_pid=pid)));return
            except psutil.Error:pass
    source=SUPPORT/('automatic-source-'+revision[:12]);source.mkdir(exist_ok=True);files={}
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'--','simulator','src','config','bin','docs'],cwd=ROOT,text=True).splitlines()
    for name in names:
        content=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        files[name]=hashlib.sha256(content).hexdigest()
    value=dict(state='staged',revision=revision,request=identifier,root=str(data),source=str(source),
        support=str(SUPPORT),files=files,created=time.time(),stops_active_training=False)
    write_json(marker,value)
    environment=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'src'),
        EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes')
    with (SUPPORT/'causal-repair-upgrade.log').open('ab') as log:
        worker=subprocess.Popen([str(SUPPORT/'runtime/bin/python'),'-m','simulator.repair_upgrade',str(marker)],
            cwd=source,env=environment,stdout=log,stderr=log,start_new_session=True)
    # The child writes this marker too. Store PID separately to avoid racing
    # its first state change, rather than overwrite the child's newest state.
    write_json(marker.with_suffix('.pid.json'),dict(pid=worker.pid,revision=revision))
    print(json.dumps(dict(state='staged',revision=revision,watcher_pid=worker.pid,marker=str(marker))))


if __name__=='__main__':main()
