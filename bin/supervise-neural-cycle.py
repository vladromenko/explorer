#!/usr/bin/env python3
"""Keep one authorized local cycle alive, adopting surviving learner processes.

This is a program supervisor, not an assistant reminder. It never interrupts
workers, resumes an operator pause, or launches a second checkpoint writer.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulator.learning import write_json,completed_training_steps
from simulator.parallel_trials import memory_free_percent

URL='http://127.0.0.1:8766'
DATA=Path('/Volumes/ExplorerSim/ExplorerSim')
SUPPORT=Path.home()/'Library/Application Support/ExplorerSimulator'


def workers(cycle):
    identifiers={task.get('pending',{}).get('job_id') for task in cycle['tasks']}-{None}
    found=[]
    for process in psutil.process_iter(['pid','cmdline']):
        args=process.info['cmdline'] or []
        if '-m' not in args:continue
        module=args[args.index('-m')+1:args.index('-m')+2]
        if not module or module[0] not in ('simulator.train_entry','lerobot.scripts.lerobot_train',
                'simulator.feedback_audit','simulator.parallel_trials'):continue
        if any(identifier in arg for identifier in identifiers for arg in args):
            found.append(process.info['pid'])
    return found


def learner_progress(cycle):
    result=[]
    for task in cycle['tasks']:
        pending=task.get('pending',{})
        if pending.get('stage')!='train':continue
        folder=DATA/'runs'/pending['job_id']
        try:
            state=json.loads((folder/'job.json').read_text())
            with (folder/'train.log').open('rb') as stream:
                stream.seek(max(0,(folder/'train.log').stat().st_size-16000))
                content=stream.read().decode(errors='replace')
            done=completed_training_steps(content,state['steps'],state.get('resumed_step',0))
            result.append(dict(id=pending['job_id'],steps_completed=done,total_steps=state['steps']))
        except (OSError,ValueError,KeyError):continue
    return result


def snapshot():
    with urllib.request.urlopen(URL+'/api/state?compact=true',timeout=10) as response:
        return json.load(response)


def run(identifier):
    spec=importlib.util.spec_from_file_location('continue_cycle',ROOT/'bin/continue-simulator-cycle.py')
    continuation=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuation)
    marker=SUPPORT/('neural-supervisor-'+identifier+'.json')
    with marker.with_suffix('.lock').open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        def save(state,**values):
            write_json(marker,dict(cycle=identifier,pid=os.getpid(),state=state,at=time.time(),**values))
        try:
            while True:
                if not DATA.parent.is_mount():save('disk_missing');return
                cycle=json.loads((DATA/'learning-cycles'/(identifier+'.json')).read_text())
                if cycle.get('pause_kind')=='operator':save('operator_paused');return
                if cycle['state'] in ('failed','needs_diagnosis','ready_for_transfer'):
                    save(cycle['state'],reason=cycle.get('reason'));return
                pids=workers(cycle)
                if pids:
                    save('waiting_for_existing_workers',worker_pids=pids,training=learner_progress(cycle))
                    time.sleep(10);continue
                try:current=snapshot()
                except OSError:
                    # No known surviving writer. Start the server only when the
                    # memory reserve also permits its physics environment.
                    if memory_free_percent()<25:save('waiting_for_memory');time.sleep(20);continue
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'),
                        EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes')
                    with (SUPPORT/'neural-service.log').open('ab') as log:
                        child=subprocess.Popen([str(SUPPORT/'runtime/bin/python'),'-m','simulator.server','--port','8766'],
                            cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
                    save('starting_service',service_pid=child.pid)
                    for _ in range(180):
                        try:current=snapshot();break
                        except OSError:
                            if child.poll() is not None:raise RuntimeError('Service failed; see neural-service.log')
                            time.sleep(.5)
                    else:raise RuntimeError('Service startup timed out; no duplicate launched')
                if any(job.get('status')=='running' for job in current.get('jobs',[])):
                    save('running');time.sleep(10);continue
                decision=continuation.check(identifier,resume=True)
                save(decision['state'],details=decision)
                if decision['state'] in ('operator_paused','failed','needs_diagnosis','ready_for_transfer','evidence_failed'):return
                time.sleep(10)
        except Exception as error:
            save('supervisor_failed',reason=str(error));raise
        finally:
            if awake.poll() is None:awake.terminate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cycle',required=True)
    options=parser.parse_args()
    if len(options.cycle)!=12 or any(c not in '0123456789abcdef' for c in options.cycle):
        parser.error('Invalid cycle ID')
    run(options.cycle)
