"""One early CPU-only room diagnostic beside, never instead of, ACT training.

The model is an immutable copy of an older numeric checkpoint. This owner has
no training/robot commands and never acquires or modifies the main cycle lease.
Its private vision worker and trial are the only processes it may stop.
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

from .learning import sha256,write_json,read_bundle


class ActiveTrialBudget:
    """Only a running trial spends the finite diagnostic time allowance."""
    def __init__(self,seconds=3600,clock=time.monotonic):
        self.limit,self.clock,self.spent,self.started=seconds,clock,0.,None

    def elapsed(self):return self.spent+(self.clock()-self.started if self.started is not None else 0.)

    def check(self):
        if self.elapsed()>=self.limit:raise TimeoutError('One-hour active early diagnostic budget exhausted')

    def start(self):
        self.check()
        if self.started is not None:raise ValueError('Diagnostic time budget already running')
        self.started=self.clock()

    def pause(self):
        if self.started is not None:self.spent=self.elapsed();self.started=None


def numeric_checkpoint(training):
    rows=sorted((p for p in (Path(training)/'model/checkpoints').iterdir()
        if p.is_dir() and p.name.isdecimal()),key=lambda p:int(p.name))
    if len(rows)<2:raise ValueError('Need two numeric checkpoints; newest may still be saving')
    path=rows[-2]/'pretrained_model'
    if not all((path/n).is_file() for n in ('model.safetensors','config.json','policy_preprocessor.json','policy_postprocessor.json')):
        raise ValueError('Older numeric checkpoint incomplete')
    return path


def manifest(path):return {str(p.relative_to(path)):sha256(p) for p in Path(path).rglob('*') if p.is_file()}


def snapshot(training,destination,previous=None):
    if previous is not None:
        import shutil
        previous,destination=Path(previous),Path(destination)
        read_bundle(previous);before=manifest(previous)
        shutil.copytree(previous,destination)
        if manifest(previous)!=before or manifest(destination)!=before:
            raise ValueError('Prior frozen diagnostic snapshot changed')
        read_bundle(destination)
        receipt=destination/'early-diagnostic.json'
        if receipt.exists():
            return int(Path(json.loads(receipt.read_text())['source']).parent.name)
        job=json.loads((destination/'job.json').read_text())
        relative=job.get('source_checkpoint_relative') or job.get('checkpoint_relative','')
        step=Path(relative).parent.name
        if not step.isdecimal():raise ValueError('Frozen bundle has no numeric checkpoint provenance')
        write_json(receipt,dict(source=str(previous/relative),files=before,
            source_training_untouched=True,diagnostic_only=True,simulation_qualified=False))
        return int(step)
    from .checkpoint_experiment import stage_candidate
    training,destination=Path(training),Path(destination);source=numeric_checkpoint(training)
    before=manifest(source)
    stage_candidate(training,str(source.relative_to(training)),destination)
    bundle,copied=read_bundle(destination)
    if manifest(source)!=before or manifest(copied)!=before:
        raise ValueError('Checkpoint changed during snapshot; copy is not admissible')
    write_json(destination/'early-diagnostic.json',dict(source=str(source),files=before,
        source_training_untouched=True,diagnostic_only=True,simulation_qualified=False))
    return int(source.parent.name)


def memory():
    import psutil
    value=psutil.virtual_memory()
    return dict(free_percent=100*value.available/value.total,available_gb=value.available/2**30)


def can_start(value,after_pressure=False):
    return value['free_percent']>=(30 if after_pressure else 25) and value['available_gb']>=5.5


def private_vision_root(root,folder):
    # Darwin AF_UNIX sockets have a short byte limit. Keep logs in the long
    # diagnostic folder, but its isolated inference transport near the root.
    from .semantic_cache import endpoint
    path=Path(root)/'vprobe'/Path(folder).name.removeprefix('live-room-')
    address,_=endpoint(path)
    if len(os.fsencode(address))>=100:raise ValueError('Private vision socket path exceeds the portable byte limit')
    return path


def stop_owned(child):
    import psutil
    try:owner=psutil.Process(child.pid);owned=owner.children(recursive=True)+[owner]
    except psutil.NoSuchProcess:return
    for proc in reversed(owned):
        try:proc.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=5)
    for proc in alive:
        try:proc.kill()
        except psutil.NoSuchProcess:pass
    child.wait(timeout=10)


def choose_seed(root,model):
    from .neural_recovery import training_seeds,reserved_scene_seeds,persisted_seeds
    used=training_seeds(model)|reserved_scene_seeds(root)
    for pattern in ('room-learning/**/result.json','diagnostics/room-*/**/result.json','diagnostics/live-room-*/request.json'):
        for p in Path(root).glob(pattern):used|=persisted_seeds(json.loads(p.read_text()))
    seed=1703000004  # round 25 cm diameter, 30 cm rim; no final holdout reuse
    while seed in used:seed+=10
    return seed


def diagnostic_seed(root,model,snapshot_copy=None):
    from .neural_recovery import training_seeds
    previous=json.loads((Path(snapshot_copy).parent/'state.json').read_text()) if snapshot_copy else {}
    seed=previous.get('seed')
    if type(seed) is not int:seed=choose_seed(root,model)
    if seed in training_seeds(model):raise ValueError('Early room scene overlaps model training')
    return seed


def run(root,folder,training,snapshot_copy=None,model_source=None,scene_seed=None):
    root,folder,training=Path(root),Path(folder),Path(training)
    if not root.parent.is_mount():raise ValueError('ExplorerSim disk is not mounted')
    folder.mkdir(parents=True,exist_ok=True)
    state=dict(state='preparing',diagnostic_only=True,simulation_qualified=False,
        physical_ready=False,training=str(training),pid=os.getpid())
    def save(**values):state.update(values,at=time.time());write_json(folder/'state.json',state)
    stopping=False;child=None;vision=None
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    with (root/'diagnostics/live-room.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=ActiveTrialBudget();pressured=False
        try:
            save(state='preparing')
            # Data copy and proposal construction are read-only on live inputs.
            step=snapshot(training,folder/'model',model_source or snapshot_copy)
            from .room_policy import build_profile
            build_profile(folder/'model',root,folder/'handoff.json')
            seed=diagnostic_seed(root,folder/'model',snapshot_copy) if scene_seed is None else scene_seed
            from .neural_recovery import training_seeds
            if type(seed) is not int or seed in training_seeds(folder/'model'):
                raise ValueError('Diagnostic scene overlaps training or has invalid identity')
            vision=private_vision_root(root,folder);vision.mkdir(parents=True)
            (vision/'models').symlink_to(root/'models',target_is_directory=True)
            save(checkpoint_step=step,seed=seed,original_training_state=json.loads((training/'job.json').read_text())['state'])
            while True:
                if stopping or (folder/'pause.request').exists():raise InterruptedError('Early diagnostic paused')
                if not root.parent.is_mount():return
                budget.check()
                available=memory()
                if not can_start(available,pressured):
                    save(state='resource_wait',memory=available,active_seconds=budget.elapsed());time.sleep(2);continue
                from .semantic_cache import ensure_started
                save(state='starting_vision',memory=available,active_seconds=budget.elapsed())
                def startup_cancelled():
                    return stopping or (folder/'pause.request').exists() or not root.parent.is_mount() or memory()['free_percent']<15
                ensure_started(vision,cancelled=startup_cancelled)
                output=folder/('trial-retry' if pressured else 'trial')
                spec=dict(root=str(vision),model=str(folder/'model'),profile=str(folder/'handoff.json'),
                    seed=seed,output=str(output),stop=str(folder/'pause.request'))
                write_json(folder/'request.json',spec)
                env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])+':'+str(Path(__file__).resolve().parents[1]/'src'),
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',EXPLORER_VISION_CPU_ONLY='1',
                    HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
                with (folder/'worker.log').open('ab') as log:
                    child=subprocess.Popen([sys.executable,'-m','simulator.room_validation','trial',str(folder/'request.json')],
                        env=env,stdout=log,stderr=log,start_new_session=True)
                budget.start()
                save(state='running',trial=str(output),child_pid=child.pid,memory=available)
                exhausted=False
                while child.poll() is None:
                    available=memory()
                    if stopping or (folder/'pause.request').exists():raise InterruptedError('Early diagnostic paused')
                    budget.check()
                    if available['free_percent']<15:
                        stop_owned(child);budget.pause();exhausted=True;save(state='resource_wait',memory=available,active_seconds=budget.elapsed(),
                            reason='Only diagnostic processes stopped to preserve the active learner');break
                    time.sleep(.5)
                budget.pause()
                if exhausted:
                    if pressured:raise InterruptedError('Two resource-limited diagnostic attempts retained; learner untouched')
                    pressured=True;continue
                if child.returncode:raise RuntimeError('Early diagnostic execution error; see worker.log')
                result=json.loads((output/'result.json').read_text())
                save(state='complete',result=str(output/'result.json'),mission_success=result['success'],
                    reason=result.get('reason'),memory=memory())
                return
        except Exception as error:
            budget.pause()
            save(state='paused' if isinstance(error,InterruptedError) else 'failed',reason=str(error),active_seconds=budget.elapsed());raise
        finally:
            from .semantic_cache import stop_servers
            if vision is not None:stop_servers(vision)
            if child is not None:
                # A private vision child can outlive a normally finished trial.
                if child.poll() is None:stop_owned(child)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--snapshot-copy',type=Path)
    parser.add_argument('--model-source',type=Path);parser.add_argument('--scene-seed',type=int)
    args=parser.parse_args();run(args.root,args.folder,args.training,args.snapshot_copy,args.model_source,args.scene_seed)
