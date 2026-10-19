"""One bounded observer of fresh saved weights; never a training writer.

Uses the production room executor, isolated CPU vision and a reserved diagnostic
scene. Results guide diagnosis; they cannot qualify a package or tune final tests.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .learning import write_json
from .live_room_diagnostic import numeric_checkpoint,stop_owned


def eligible_step(available,last_step,interval=10000):
    return available if available>last_step and (last_step==0 or available-last_step>=interval) else None


def paused(root,owner,folder):
    return any(p.exists() for p in [Path(folder)/'pause.request',Path(root)/'neural-recovery'/owner/'pause.request'])


def current_diagnostic(root):
    marker=Path(root)/'diagnostics/early-room-current.json'
    if not marker.exists():return None
    value=json.loads(marker.read_text());state=Path(value['folder'])/'state.json'
    return dict(marker=value,state=json.loads(state.read_text()) if state.exists() else {})


def diagnostic_busy(value):
    if not value:return False
    import psutil
    pid=value['marker'].get('pid')
    if not pid or not psutil.pid_exists(pid):return False
    return value['state'].get('state') not in ('complete','failed','paused')


def failure_scope(result):
    from .failure_repair import infrastructure_failure
    if result.get('success') is True:return None
    if infrastructure_failure(result):return 'infrastructure'
    stages={row.get('stage') for row in result.get('events',[])}
    if 'policy_pick_running' not in stages:return 'executor_before_policy'
    reason=result.get('reason','')
    if any(token in reason for token in ['Недопустимый шаг сустава','ObservedMotionViolation','Neural opening is outside']):
        return 'policy_interface_or_guard'
    return 'policy_outcome'


def run(root,training,owner,minimum_seconds=10800,interval=10000,max_trials=12):
    root=Path(root);training=root/'runs'/training
    if not root.parent.is_mount():raise ValueError('External disk is absent')
    folder=root/'diagnostics'/('checkpoint-monitor-'+training.name);folder.mkdir(exist_ok=True)
    state_path=folder/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(results=[],last_step=0,last_launched_at=0)
    state.update(pid=os.getpid(),training=str(training),owner=owner,diagnostic_only=True,simulation_qualified=False,robot_verified=False)
    stopping=False;child=None
    def stop(*args):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    def save(**values):state.update(values,at=time.time());write_json(state_path,state)
    with (root/'diagnostics/checkpoint-monitor.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while not stopping and len(state['results'])<max_trials:
                if not root.parent.is_mount():return
                if paused(root,owner,folder):raise InterruptedError('Operator pause retained')
                local=json.loads((root/'neural-recovery'/owner/'state.json').read_text())
                job=json.loads((training/'job.json').read_text())
                active=(local.get('state')=='running' and local.get('phase')=='train'
                        and local.get('training_model_id')==training.name and job.get('state')=='training')
                if not active:
                    save(state='finished',reason='Learner left this stage; main program owns subsequent evaluation')
                    return
                existing=current_diagnostic(root)
                step=None
                try:step=int(numeric_checkpoint(training).parent.name)
                except (OSError,ValueError):pass
                due=step is not None and eligible_step(step,state['last_step'],interval) is not None
                due=due and time.time()-state['last_launched_at']>=minimum_seconds
                if due and not diagnostic_busy(existing):
                    with (root/'diagnostics/early-room-launch.lock').open('a') as launch_lease:
                        fcntl.flock(launch_lease,fcntl.LOCK_EX)
                        latest=current_diagnostic(root)
                        if diagnostic_busy(latest):raise InterruptedError('Another diagnostic acquired the shared launch lease')
                        trial=folder/('live-room-'+training.name+'-'+str(step));trial.mkdir(exist_ok=False)
                        command=[sys.executable,'-m','simulator.live_room_diagnostic','--root',str(root),
                            '--training',str(training),'--folder',str(trial),'--scene-seed','1703000014']
                        environment=dict(os.environ,EXPLORER_VISION_CPU_ONLY='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                        with (trial/'supervisor.log').open('ab') as log:
                            child=subprocess.Popen(command,stdout=log,stderr=log,env=environment,start_new_session=True)
                        marker=dict(folder=str(trial),pid=child.pid,source=str(Path(__file__).resolve().parents[1]),
                            revision=json.loads((Path(__file__).resolve().parents[1]/'source-revision.json').read_text())['revision'],
                            diagnostic_only=True,simulation_qualified=False,current_learning_interrupted=False,
                            monitor=str(folder),previous_folder=existing['marker']['folder'] if existing else None)
                        write_json(root/'diagnostics/early-room-current.json',marker)
                    save(state='evaluating',last_step=step,last_launched_at=time.time(),child_pid=child.pid,trial=str(trial))
                    while child.poll() is None and not stopping:
                        if paused(root,owner,folder):raise InterruptedError('Operator pause retained')
                        local=json.loads((root/'neural-recovery'/owner/'state.json').read_text())
                        if local.get('phase')!='train' or local.get('state')!='running':
                            stop_owned(child)
                        time.sleep(2)
                    result_path=trial/'state.json'
                    result=json.loads(result_path.read_text()) if result_path.exists() else dict(state='no_result')
                    state['results'].append(dict(step=result.get('checkpoint_step',step),path=str(trial),
                        state=result.get('state'),success=result.get('mission_success'),reason=result.get('reason')))
                    save(state='waiting_for_new_checkpoint',child_pid=None)
                    outcome_path=Path(result['result']) if result.get('result') else None
                    outcome=json.loads(outcome_path.read_text()) if outcome_path and outcome_path.exists() else {}
                    scope=failure_scope(outcome) if outcome else 'infrastructure'
                    if scope in ('infrastructure','executor_before_policy','policy_interface_or_guard'):
                        save(state='needs_diagnosis',failure_scope=scope,reason=result.get('reason'));return
                    if result.get('state')=='failed':
                        save(state='needs_diagnosis',reason=result.get('reason'));return
                else:
                    save(state='waiting',reason='Existing diagnostic active' if diagnostic_busy(existing) else 'Need a newer stable checkpoint and evaluation interval')
                time.sleep(30)
            save(state='paused' if stopping else 'budget_complete')
        except InterruptedError as error:save(state='paused',reason=str(error))
        except Exception as error:
            save(state='failed',reason=str(error));raise
        finally:
            if child is not None and child.poll() is None:stop_owned(child)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training',required=True);parser.add_argument('--owner',required=True)
    args=parser.parse_args()
    import re
    if not all(re.fullmatch('[0-9a-f]{12}',x) for x in [args.training,args.owner]):raise ValueError('Invalid run identity')
    run(args.root,args.training,args.owner)


if __name__=='__main__':main()
