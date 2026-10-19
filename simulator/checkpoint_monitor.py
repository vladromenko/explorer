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


def paused(root,owner,folder,owner_kind='neural-recovery'):
    return any(p.exists() for p in [Path(folder)/'pause.request',Path(root)/owner_kind/owner/'pause.request'])


def source_revision():
    source=Path(__file__).resolve().parents[1]
    for name in ('source-revision.json','source-manifest.json'):
        if (source/name).exists():return json.loads((source/name).read_text())['revision']
    raise ValueError('Checkpoint observer requires an immutable source manifest')


def pause_auxiliary(root):
    """An operator pause covers diagnostic children, not just gradient jobs."""
    root=Path(root).resolve();folders=[]
    value=current_diagnostic(root)
    if value:
        folder=Path(value['marker']['folder']).resolve()
        if not folder.is_relative_to(root/'diagnostics'):raise ValueError('Diagnostic marker escapes storage')
        folders.append(folder)
    folders.extend(path.parent for path in (root/'diagnostics').glob('checkpoint-monitor-*/state.json'))
    for folder in folders:(folder/'pause.request').touch()
    return [str(folder) for folder in folders]


def resume_auxiliary(root):
    """Only an explicit Resume may clear saved auxiliary pauses, after drain.

    This does not launch or retry an old diagnostic. Its outcome remains saved;
    the current training worker owns any future observer on the checked source.
    """
    import psutil
    root=Path(root).resolve()
    for process in psutil.process_iter(['cmdline']):
        args=process.info['cmdline'] or []
        if '-m' in args and args[args.index('-m')+1] in (
                'simulator.checkpoint_monitor','simulator.live_room_diagnostic'):
            raise ValueError('Auxiliary evaluation is still saving; wait before explicit resume')
    pauses=list((root/'diagnostics').glob('checkpoint-monitor-*/pause.request'))
    value=current_diagnostic(root)
    if value:
        folder=Path(value['marker']['folder']).resolve()
        if not folder.is_relative_to(root/'diagnostics'):raise ValueError('Diagnostic marker escapes storage')
        pauses.append(folder/'pause.request')
    retained=[dict(path=str(path),contents=path.read_text()) for path in pauses if path.exists()]
    if retained:
        write_json(root/'maintenance'/('auxiliary-explicit-resume-'+str(time.time_ns())+'.json'),
            dict(pauses=retained,at=time.time(),automatic_retry=False))
        for row in retained:Path(row['path']).unlink()
    return retained


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


def run(root,training,owner,minimum_seconds=10800,interval=10000,max_trials=12,owner_kind='neural-recovery'):
    root=Path(root);training=root/'runs'/training
    if not root.parent.is_mount():raise ValueError('External disk is absent')
    folder=root/'diagnostics'/('checkpoint-monitor-'+training.name);folder.mkdir(exist_ok=True)
    state_path=folder/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(results=[],last_step=0,last_launched_at=0)
    state.update(pid=os.getpid(),training=str(training),owner=owner,owner_kind=owner_kind,
        code_revision=source_revision(),diagnostic_only=True,simulation_qualified=False,robot_verified=False)
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
                if paused(root,owner,folder,owner_kind):raise InterruptedError('Operator pause retained')
                local=json.loads((root/owner_kind/owner/'state.json').read_text())
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
                due=due and job.get('steps_completed',0)>=job.get('resumed_step',0)+100
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
                            revision=source_revision(),
                            diagnostic_only=True,simulation_qualified=False,current_learning_interrupted=False,
                            monitor=str(folder),previous_folder=existing['marker']['folder'] if existing else None)
                        write_json(root/'diagnostics/early-room-current.json',marker)
                    save(state='evaluating',last_step=step,last_launched_at=time.time(),child_pid=child.pid,trial=str(trial))
                    while child.poll() is None and not stopping:
                        if paused(root,owner,folder,owner_kind):raise InterruptedError('Operator pause retained')
                        local=json.loads((root/owner_kind/owner/'state.json').read_text())
                        job=json.loads((training/'job.json').read_text())
                        if local.get('phase')!='train' or local.get('state')!='running' or job.get('state')!='training':
                            stop_owned(child)
                        time.sleep(2)
                    if stopping:
                        save(state='paused',reason='Observer stopped; diagnostic evidence retained, learner untouched')
                        return
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


def start_for_training(training,owner_folder,environment,interpreter):
    """Start one read-only observer owned by this training worker."""
    import psutil
    training,owner_folder=Path(training).resolve(),Path(owner_folder).resolve()
    root=training.parent.parent;kind=owner_folder.parent.name
    if kind not in ('neural-recovery','room-learning') or owner_folder.parent.parent!=root:
        raise ValueError('Unknown checkpoint observer owner')
    for process in psutil.process_iter(['cmdline']):
        args=process.info['cmdline'] or []
        if 'simulator.checkpoint_monitor' in args:return None
    folder=root/'diagnostics'/('checkpoint-monitor-'+training.name)
    folder.mkdir(parents=True,exist_ok=True);path=folder/'state.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior.get('state')=='needs_diagnosis' and prior.get('code_revision')==source_revision():
            return None  # Same-code retry cannot repair an executor defect.
        if prior.get('code_revision')!=source_revision():
            receipt=folder/('before-source-upgrade-'+str(time.time_ns())+'.json');write_json(receipt,prior)
            prior.setdefault('source_upgrades',[]).append(dict(previous=prior.get('code_revision'),
                next=source_revision(),retained_state=str(receipt)))
            # Interrupted infrastructure attempts are retained, but do not
            # consume the next version's first genuine checkpoint comparison.
            if prior.get('failure_scope')=='infrastructure':prior.update(last_step=0,last_launched_at=0)
            write_json(path,prior)
    with (folder/'supervisor.log').open('ab') as log:
        return subprocess.Popen([interpreter,'-m','simulator.checkpoint_monitor','--root',str(root),
            '--training',training.name,'--owner',owner_folder.name,'--owner-kind',kind],
            env=environment,stdout=log,stderr=log,start_new_session=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training',required=True);parser.add_argument('--owner',required=True)
    parser.add_argument('--owner-kind',choices=('neural-recovery','room-learning'),default='neural-recovery')
    args=parser.parse_args()
    import re
    if not all(re.fullmatch('[0-9a-f]{12}',x) for x in [args.training,args.owner]):raise ValueError('Invalid run identity')
    run(args.root,args.training,args.owner,owner_kind=args.owner_kind)


if __name__=='__main__':main()
