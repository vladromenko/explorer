"""Finite full-room diagnostics, separate from current local ACT training."""
import argparse
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from .learning import read_bundle,write_json,sha256
from .domain_randomization import sample_sock_scene


def room_scene(seed):
    scene=sample_sock_scene(seed);rng=np.random.default_rng(seed+712)
    def position():
        for _ in range(200):
            value=rng.uniform([-1.4,-1.35],[1.4,1.35])
            if np.linalg.norm(value)>.7 and abs(value[0]-.5)>.35:return value
        raise ValueError('No separated room scene')
    source=position();destination=position()
    for _ in range(200):
        if np.linalg.norm(source-destination)>=1.:break
        destination=position()
    else:raise ValueError('Room targets not separated')
    scene['sock_position']=[*source,.015];scene['bin_position']=[*destination,0.]
    return scene


def frozen_source():
    from .room_policy import RUNTIME_FILES
    source=Path(__file__).resolve().parents[1];path=source/'source-revision.json'
    if not path.is_file():return None
    manifest=json.loads(path.read_text())
    if (not re.fullmatch('[0-9a-f]{40}',manifest.get('revision','')) or set(manifest.get('files',{}))!=set(RUNTIME_FILES)
            or any(sha256(source/name)!=digest for name,digest in manifest['files'].items())):
        raise ValueError('Frozen room qualification source changed')
    return manifest['revision']


class CollisionAudit:
    """Evaluator-only contact recorder; never queried by the mission actor."""
    def __init__(self,env):
        self.env=env;self.contacts=[]
        self.walls={i for i in range(env.model.ngeom) if env.model.geom(i).name.startswith(('cleanup_room_','bin_'))}
        self.robot=set();self.arm=set();self.floor=env.model.geom('floor').id
        for i in range(env.model.ngeom):
            body=int(env.model.geom_bodyid[i])
            while body and env.model.body(body).name!='base_link':
                if env.model.body(body).name=='arm1':self.arm.add(i)
                body=int(env.model.body_parentid[body])
            if body:self.robot.add(i)

    def sample(self):
        for contact in self.env.data.contact:
            a,b=map(int,contact.geom)
            if contact.dist<-.001 and ((a in self.walls and b in self.robot) or (b in self.walls and a in self.robot)
                    or (a==self.floor and b in self.arm) or (b==self.floor and a in self.arm)):
                if len(self.contacts)<100:self.contacts.append(dict(time=float(self.env.data.time),geoms=[a,b],distance=float(contact.dist)))


def trial(spec):
    from .environment import ExplorerEnv
    from .room_cleanup_port import RoomCleanupPort
    from src.room_cleanup import RoomCleanup
    import torch
    torch.set_num_threads(1)
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
    revision=frozen_source();scene=room_scene(spec['seed']);env=ExplorerEnv(config={'room_layout':True})
    port=None;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    latest={};preview_at=-1.
    def record(value):latest.update(value);write_json(output/'progress.json',latest)
    def motion(value):
        nonlocal preview_at
        latest['motion']=value;write_json(output/'progress.json',latest)
        from .worker_views import capture_overview
        stamp=capture_overview(env,output,preview_at)
        if stamp!=preview_at:
            from PIL import Image
            preview=output/'wrist-preview.tmp';Image.fromarray(env.render_rgb('wrist')).save(preview,format='JPEG',quality=75)
            preview.replace(output/'wrist-preview.jpg')
        preview_at=stamp
    try:
        env.reset(seed=spec['seed'],options=dict(task_id='sock_basket',scene_profile='sock_variants_v1',generated_scene=scene))
        audit=CollisionAudit(env);env._outcome_observer=audit.sample
        profile=json.loads(Path(spec['profile']).read_text())
        cancelled=lambda:stopping or ('stop' in spec and Path(spec['stop']).exists())
        port=RoomCleanupPort(env,output/'observed',spec['model'],profile,spec['root'],cancelled=cancelled,progress=motion)
        result=RoomCleanup(port,record,cancelled=cancelled).run()
        # Only AFTER action termination is truth used to score the run.
        truth=env.state()['task']
        if frozen_source()!=revision:raise ValueError('Qualification source changed during room execution')
        result.update(seed=spec['seed'],physical_outcome=truth,domain_variation=scene,
            success=result['state']=='succeeded' and truth['success'] is True and not audit.contacts,
            collision_free=not audit.contacts,collision_evidence=audit.contacts,
            room_layout=True,collision_sampling='each_physics_substep',
            immutable_source_verified=revision is not None,source_revision=revision,
            actor_truth_inputs=False,teacher_fallback=False,simulation_only=True,physical_ready=False,
            mission_contract_sha256=port.contract,model_sha256=port.actor.sha)
        write_json(output/'result.json',result);return result
    finally:
        if port is not None:port.close()
        env.close()


def wait_and_run(root,identifier):
    """One authorized background dependency; no Codex reminders or robot I/O."""
    import fcntl
    import psutil
    from .neural_recovery import current,training_seeds,reserved_scene_seeds,fresh_seeds
    from .parallel_trials import memory_free_percent
    from .room_policy import build_profile
    if not re.fullmatch('[0-9a-f]{12}',identifier):raise ValueError('Invalid room cycle identifier')
    root=Path(root)
    if not root.parent.is_mount():return
    folder=root/'room-learning'/identifier
    folder.mkdir(parents=True,exist_ok=True);path=folder/'state.json'
    value=json.loads(path.read_text()) if path.exists() else dict(id=identifier,created=time.time(),state='queued')
    def save(**kwargs):value.update(kwargs,at=time.time(),pid=os.getpid());write_json(path,value)
    with (folder/'owner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if value['state'] in ('ready_for_observed_transfer','needs_room_diagnosis','needs_room_algorithm_repair','operator_paused'):return
        while True:
            # Never recreate an external data directory on an unmounted disk.
            if not root.parent.is_mount():return
            if (folder/'pause.request').exists():save(state='operator_paused');return
            recovery=current(root)
            busy=bool(recovery and recovery['alive'])
            # Adopt no process and create no second model writer. Current ACT
            # recovery keeps all resources until its own finite cycle ends.
            for process in psutil.process_iter(['cmdline']):
                try:args=process.info['cmdline'] or []
                except (psutil.NoSuchProcess,psutil.AccessDenied):continue
                if '-m' in args and args.index('-m')+1<len(args) and args[args.index('-m')+1] in (
                        'simulator.train_entry','lerobot.scripts.lerobot_train','simulator.neural_recovery'):
                    busy=True
            if busy or memory_free_percent()<30:
                save(state='waiting_for_local_learning',phase='waiting',local_cycle=(recovery or {}).get('id'),
                    local_cycle_id=(recovery or {}).get('cycle_id'),
                    message='Полный сценарий ждёт завершения текущего обучения; оно не прерывается')
                time.sleep(15);continue
            if recovery is None:save(state='blocked',reason='No existing neural recovery checkpoint');return
            if recovery['state']=='paused':save(state='operator_paused');return
            break
        (root/'skill-training').mkdir(exist_ok=True)
        with (root/'skill-training/owner.lock').open('a') as physics_lease:
            while True:
                try:fcntl.flock(physics_lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if (folder/'pause.request').exists():save(state='operator_paused');return
                    if not root.parent.is_mount():return
                    save(state='waiting_for_local_learning',phase='waiting',message='Ждём освобождения единственного владельца вычислений')
                    time.sleep(15)
            from .room_control import publish_controller
            try:save(controller=publish_controller(root))
            except Exception as error:
                save(state='failed',reason='Idle controller update: '+str(error));raise
            from .room_learning import RoomLearning
            from .neural_recovery import persisted_seeds
            model=Path(recovery['model']);read_bundle(model)
            excluded=training_seeds(model)|reserved_scene_seeds(root)
            for previous in root.glob('room-learning/*/state.json'):
                if previous!=path:excluded|=persisted_seeds(json.loads(previous.read_text()))
            for previous in root.glob('room-learning/*/**/result.json'):
                if not previous.is_relative_to(folder):excluded|=persisted_seeds(json.loads(previous.read_text()))
            for previous in root.glob('diagnostics/room-development-*/trial/result.json'):
                excluded|=persisted_seeds(json.loads(previous.read_text()))
            if value.get('active_pids'):
                for pid in value['active_pids']:
                    try:process=psutil.Process(pid)
                    except psutil.NoSuchProcess:continue
                    if any(str(folder) in part for part in process.cmdline()):
                        raise RuntimeError('A remaining child owns this cycle; no duplicate writer started')
            initial=folder/'initial-model.json'
            if not initial.exists():write_json(initial,dict(model=str(model)))
            model=Path(json.loads(initial.read_text())['model'])
            engine=RoomLearning(root,folder,value,save,excluded)
            awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
            try:
                while True:
                    try:qualified=engine.run_learning(model);break
                    except InterruptedError as error:
                        engine.shutdown();engine.children={}
                        if (folder/'pause.request').exists():raise
                        if not root.parent.is_mount():return
                        if memory_free_percent()<30:
                            value['parallel_ceiling']=max(1,value.get('workers',1)//2)
                        save(state='resource_wait',phase='waiting',active_pids=[],reason=str(error),
                            message='Данные сохранены; программа ждёт ресурсов для продолжения того же цикла')
                        import shutil
                        while memory_free_percent()<30 or shutil.disk_usage(root).free<16*1024**3:
                            if (folder/'pause.request').exists():raise InterruptedError('Пауза пользователя')
                            if not root.parent.is_mount():return
                            time.sleep(15)
                if qualified is None:return
                from .room_transfer import export_room_package
                archive=export_room_package(qualified['model'],qualified['profile'],qualified['groups'],
                    root/'exports'/('room-'+identifier+'.tar.gz'),excluded_seeds=qualified['excluded'])
                from .room_archive import archive_cycle
                save(phase='archive',archive=archive)
                evidence=archive_cycle(folder,archive,root/'exports'/('room-cycle-'+identifier+'.tar.gz'),
                    cancelled=lambda:(folder/'pause.request').exists())
                save(state='ready_for_observed_transfer',phase='ready',archive=archive,evidence_archive=evidence,physical_ready=False)
            except Exception as error:
                if not root.parent.is_mount():return
                save(state='operator_paused' if (folder/'pause.request').exists() else 'failed',reason=str(error))
                if not (folder/'pause.request').exists():raise
            finally:
                engine.shutdown()
                if root.parent.is_mount():save(active_pids=[])
                if awake.poll() is None:awake.terminate()


if __name__=='__main__':
    if sys.argv[1:2]==['trial']:trial(json.loads(Path(sys.argv[2]).read_text()))
    elif sys.argv[1:2]==['worker']:
        request=Path(sys.argv[2]);spec=json.loads(request.read_text())
        if spec['kind']=='train':
            from .neural_recovery import worker
            worker(request)
        elif spec['kind']=='room_correct':
            from .room_learning import demonstrate
            demonstrate(spec)
        elif spec['kind']=='room_evaluate':trial(spec)
        else:raise ValueError('Unknown room learning worker')
    else:
        parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--id',required=True)
        args=parser.parse_args();wait_and_run(args.root,args.id)
