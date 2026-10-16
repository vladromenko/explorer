"""Durable, finite black-box policy learning from observed skill rollouts.

The learner changes bounded control parameters. It never changes the model,
physics, collision margins or evidence gates. Actors see sensors and their own
kinematics; reward reads outcome booleans only AFTER their actions stop.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import time

import numpy as np

from .resource_budget import memory_budget
from .learning import write_json
from .skill_parameters import PARAMETERS,parameters
from .skill_workshop import execution_fingerprint


def identity(value):
    if not isinstance(value,str) or len(value)!=12 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid skill training identity')
    return value


def list_trainings(root):
    output=[]
    for path in (Path(root)/'skill-training').glob('*/state.json'):
        try:
            row=json.loads(path.read_text());identity(row['id']);output.append(row)
        except (OSError,ValueError,KeyError,TypeError):continue
    return sorted(output,key=lambda row:row.get('created',0),reverse=True)



def recover_trainings(root):
    """Expose interrupted work for explicit resume; never restart it on boot."""
    folder=Path(root)/'skill-training';folder.mkdir(exist_ok=True)
    with (folder/'owner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        for state in list_trainings(root):
            if state['state'] not in ('running','validating','packaging','waiting_for_memory'):continue
            trainer=SkillTraining(root,state['id']);trainer.pause.touch(exist_ok=True)
            state.update(state='interrupted',active_workers=[],reason='Сервис перезапущен; продолжите сохранённое обучение кнопкой')
            trainer.save(state)


def outcome_score(report):
    """Search reward is NOT a success/qualification assertion."""
    execution=report.get('execution',{});truth=report.get('independent_physics_outcome',{})
    if report.get('success') is True and execution.get('placed_verified') is True and truth.get('success') is True:
        return 100.
    passed={e['stage'] for e in execution.get('events',[]) if e['result'].get('outcome')=='success'}
    if not (execution.get('held_verified') is True and truth.get('lifted') is True):
        return float(sum(stage in passed for stage in ('find','approach','reobserve')))
    if truth.get('success') is True:return 20. # Physical placement alone is not accepted success.
    if truth.get('grasp_contact') is not True:return 10. if 'release' in passed else 3.
    return 10.+sum(stage in passed for stage in ('carry','support'))


def failure_group(report):
    stage=report.get('execution',{}).get('stage',report.get('phase'))
    if stage=='find':return 'view'
    if stage=='carry':return 'pick' if 'Транспортная поза' in str(report.get('reason')) else 'view'
    if stage in ('support','release','withdraw','verify_place'):return 'place'
    return 'pick'


def propose(state):
    """Cross-entropy proposals only for the empirically blocked skill group."""
    incumbent=parameters(state['incumbent']);group=state['phase_group']
    rng=np.random.default_rng(state['seed']+state['generation']*1009)
    population=[incumbent]
    for index in range(3):
        choice=dict(incumbent)
        for name,(lo,hi,_,phase) in PARAMETERS.items():
            if phase!=group:continue
            # Regular exploration across the full permitted range prevents
            # a collapsed distribution from repeating identical failed moves.
            value=(rng.uniform(lo,hi) if index==2 or state['stagnation']>=3 else
                   rng.normal(state['mean'][name],state['std'][name]))
            choice[name]=float(np.clip(value,lo,hi))
        population.append(parameters(choice))
    return population


def update_search(state,reports):
    grouped={i:[] for i in range(4)}
    for row in reports:grouped[row['candidate']].append(row)
    if any(len(rows)!=2 for rows in grouped.values()):raise ValueError('Paired search batch incomplete')
    def rank(index):
        scores=[row['score'] for row in grouped[index]]
        return min(scores),sum(scores)/len(scores)
    baseline=rank(0)
    eligible=[i for i in grouped if rank(i)[0]>=baseline[0]]
    chosen=max(eligible,key=lambda i:(rank(i)[1],rank(i)[0],-i))
    improved=rank(chosen)[1]>baseline[1]
    if improved:
        state['incumbent']=dict(state['batch']['population'][chosen]);state['stagnation']=0
    else:state['stagnation']+=1
    elite=sorted(eligible,key=rank,reverse=True)[:2]
    for name,(lo,hi,_,phase) in PARAMETERS.items():
        if phase!=state['phase_group']:continue
        values=[state['batch']['population'][i][name] for i in elite]
        state['mean'][name]=float(np.mean(values))
        state['std'][name]=float(np.clip(np.std(values),.07*(hi-lo),.4*(hi-lo)))
        if state['stagnation']>=3:state['std'][name]=.35*(hi-lo)
    chosen_rows=grouped[chosen]
    groups=Counter(row['failure_group'] for row in chosen_rows if row['success'] is not True)
    if groups:state['phase_group']=groups.most_common(1)[0][0]
    state['last_selection']=dict(candidate=chosen,improved=improved,
        baseline_scores=[r['score'] for r in grouped[0]],selected_scores=[r['score'] for r in chosen_rows])
    return all(row['success'] is True for row in chosen_rows)


def episode(spec):
    """Isolated process owns one simulator. No SSH/ROS or scene truth input."""
    from PIL import Image
    from .environment import ExplorerEnv
    from .skill_workshop import trial
    folder=Path(spec['folder']);pause=Path(spec['pause_file']);storage=Path(spec['storage'])
    if execution_fingerprint()['sha256']!=spec['controller_sha256']:
        raise RuntimeError('Controller changed before worker start')
    folder.mkdir(parents=True,exist_ok=False);(folder/'frames').mkdir()
    env=ExplorerEnv();last=-1.;captures=0
    def cancelled():
        return pause.exists() or not storage.is_dir() or os.getppid()!=spec['owner_pid']
    def progress(value):
        nonlocal last,captures
        if cancelled():raise InterruptedError('Пауза обучения навыков')
        now=time.monotonic()
        if now-last<.5:return
        last=now;captures+=1
        for camera,path in (('overview',folder/'overview-preview.jpg'),
                            ('wrist',folder/'frames'/f'{captures:06d}.jpg')):
            tmp=path.with_suffix('.tmp')
            Image.fromarray(env.render_rgb(camera)).save(tmp,format='JPEG',quality=80);tmp.replace(path)
        write_json(folder/'progress.json',dict(phase=env.phase,simulation_time=float(env.data.time),
            seed=spec['scene_seed'],candidate=spec['candidate']))
    try:
        report=trial(env,folder,'attempt',spec.get('task_id','sock_basket'),spec['scene_seed'],
            spec.get('scene_profile','sock_variants_v1'),cancelled,progress,parameters=spec['parameters'],
            scene_recipe=spec.get('scene_recipe'),semantic_root=spec.get('semantic_root'),
            validation=spec.get('split','train')=='validation')
        # Outcome and scene provenance become feedback only after trial stops.
        result=dict(episode_id=spec['episode_id'],candidate=spec['candidate'],seed=spec['scene_seed'],
            split=spec.get('split','train'),
            status='paused' if report['state']=='paused' else 'complete',success=report.get('success') is True,
            score=outcome_score(report),failure_group=failure_group(report),stage=report.get('phase'),
            reason=report.get('reason'),report=str(folder/'skill-workshops/attempt/state.json'),
            domain_variation=env.domain_variation)
        write_json(folder/'result.json',result);return result
    finally:env.close()


class SkillTraining:
    def __init__(self,root,identifier):
        self.root=Path(root);self.id=identity(identifier);self.folder=self.root/'skill-training'/self.id
        self.path=self.folder/'state.json';self.pause=self.folder/'pause.request'

    def save(self,state):
        state['updated']=time.time();write_json(self.path,state)

    def create(self,*,attempts=100,hours=24.,workers=4,seed=906160000,task_id='sock_basket',scene_recipe=None):
        if self.path.exists():
            saved=json.loads(self.path.read_text())
            if saved.get('task_id')!=task_id or saved.get('scene_recipe')!=scene_recipe:
                raise ValueError('Saved training belongs to a different task or scene recipe')
            return saved
        if type(attempts) is not int or not 8<=attempts<=1000 or not 0.<hours<=24. or workers not in (1,4):
            raise ValueError('Invalid bounded skill learning budget')
        from .curriculum import task_spec
        task_spec(task_id)
        if scene_recipe is not None:
            from .scene_recipes import validate_recipe
            scene_recipe=validate_recipe(scene_recipe)
            if scene_recipe['task_id']!=task_id:raise ValueError('Training recipe/task mismatch')
        self.folder.mkdir(parents=True,exist_ok=False)
        values=parameters()
        state=dict(schema=1,id=self.id,created=time.time(),state='queued',task_id=task_id,scene_recipe=scene_recipe,
            scene_profile='sock_variants_v1' if task_id in ('sock_basket','cloth_basket') else 'baseline',
            algorithm='phase_local_cross_entropy_policy_search_v1',seed=seed,generation=0,
            options=dict(attempts=attempts,hours=hours,workers=workers),phase_group='place',stagnation=0,
            incumbent=values,mean=dict(values),std={k:(s[1]-s[0])*.30 for k,s in PARAMETERS.items()},
            episodes=[],active_workers=[],completed_attempts=0,simulation_ready=False,physical_ready=False,
            automatic_hardware_execution=False,controller=execution_fingerprint(),
            validation_seeds=[list(range(1300160000,1300160010)),list(range(1400160000,1400160010))])
        if scene_recipe is not None:
            critical=('src/task_executor.py','simulator/skill_port.py','simulator/skill_parameters.py',
                'simulator/grasp_planning.py','simulator/manipulation_planning.py','simulator/inspection_planning.py',
                'simulator/explorer_model.py','simulator/model_config.json','config/explorer.urdf',
                'config/handeye-accepted.json','config/arm-motion.json')
            for old in list_trainings(self.root):
                if (old.get('id')==self.id or old.get('task_id')!=task_id or old.get('generation',0)<1 or
                    old.get('algorithm')!=state['algorithm']):continue
                source=old.get('controller',{}).get('files',{})
                if not all(source.get(name)==state['controller']['files'].get(name) for name in critical):continue
                try:values=parameters(old['incumbent'])
                except (ValueError,KeyError):continue
                state['incumbent']=values;state['mean']=dict(values)
                state['initialization']=dict(kind='bounded_parameter_warm_start',from_training_id=old['id'],
                    source_controller_sha256=old['controller']['sha256'],unverified=True,
                    old_episodes_counted_for_qualification=False)
                break
        if scene_recipe is not None and len(scene_recipe['request'])>1:
            state['validation_seeds'].append(list(range(1500160000,1500160010)))
        self.save(state);return state

    def _batch(self,state,specs,cancelled,progress):
        workers=state['options']['workers'];pending=list(specs);results=[];active={}
        previous_state=state['state']
        reservation=int((3.2 if state.get('scene_recipe') else 1.7)*2**30)
        while True:
            memory=memory_budget()
            # Reserve each worker's measured-order 1.7 GiB footprint BEFORE
            # spawn; workers still starting have not allocated that RAM yet.
            capacity=min(workers,max(0,int((memory['available_bytes']-memory['total_bytes']*.15)/reservation)))
            memory['reserved_bytes_per_worker']=reservation
            state['resource_capacity']=capacity
            state['resource_measurement']=memory
            if capacity:
                state.update(state=previous_state,reason=None);break
            # No worker exists yet: wait within the same finite time budget,
            # rather than allocate unsafely or require a repeated manual start.
            state.update(state='waiting_for_memory',active_workers=[],
                reason=f'Ожидаю свободную память: минимум 15% запаса плюс {reservation/2**30:.1f} ГиБ на исполнителя')
            self.save(state);progress(state)
            if cancelled() or self.pause.exists():raise InterruptedError('Ожидание памяти остановлено')
            time.sleep(2.)
        with ProcessPoolExecutor(max_workers=capacity,mp_context=multiprocessing.get_context('spawn')) as pool:
            try:
                while pending or active:
                    if cancelled():self.pause.touch(exist_ok=True)
                    memory=memory_budget()
                    if memory['available_bytes']/memory['total_bytes']<.15:
                        state['pause_kind']='memory';self.pause.touch(exist_ok=True)
                    while pending and len(active)<capacity and not self.pause.exists():
                        spec=pending.pop(0);slot=next(i for i in range(capacity) if i not in [x['slot'] for x in active.values()])
                        spec['slot']=slot;spec['owner_pid']=os.getpid();active[pool.submit(episode,spec)]=spec
                    state['active_workers']=[dict(slot=s['slot'],episode_id=s['episode_id'],seed=s['scene_seed'],
                        candidate=s['candidate']) for s in active.values()]
                    self.save(state);progress(state)
                    if not active:
                        if pending and not self.pause.exists():
                            state['pause_kind']='memory';self.pause.touch(exist_ok=True)
                        break
                    done,_=wait(active,timeout=.5,return_when=FIRST_COMPLETED)
                    for future in done:
                        spec=active.pop(future)
                        try:result=future.result()
                        except Exception:
                            self.pause.touch(exist_ok=True)
                            # Cooperatively stop siblings before reporting the error.
                            for other in active:
                                try:other.result()
                                except Exception:pass
                            raise
                        results.append(result);state['episodes'].append(result)
                        state['completed_attempts']=sum(r['status']=='complete' and r.get('split','train')=='train'
                            for r in state['episodes'])
                        self.save(state)
                state['active_workers']=[];self.save(state)
            except BaseException:
                # Set the cooperative flag before the pool context waits for
                # workers, including when the parent receives KeyboardInterrupt.
                self.pause.touch(exist_ok=True)
                raise
        if pending or any(r['status']!='complete' for r in results):raise InterruptedError('Обучение сохранено на паузе')
        return results

    def _spec(self,state,params,candidate,scene_seed,*,split='train'):
        token=secrets.token_hex(6)
        return dict(episode_id=token,folder=str(self.folder/'episodes'/token),storage=str(self.root),
            pause_file=str(self.pause),parameters=parameters(params),candidate=candidate,scene_seed=scene_seed,
            controller_sha256=state['controller']['sha256'],split=split,owner_pid=os.getpid(),
            task_id=state['task_id'],scene_recipe=state.get('scene_recipe'),semantic_root=str(self.root),
            scene_profile=state.get('scene_profile','sock_variants_v1'))

    def run(self,*,cancelled=lambda:False,progress=lambda state:None,resume=False):
        state=json.loads(self.path.read_text())
        lock=(self.root/'skill-training'/'owner.lock').open('a')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();raise RuntimeError('Another skill learner owns the simulator workers')
        try:
            if state['state'] in ('simulation_qualified','needs_diagnosis','validation_failed'):return state
            if state['controller']['sha256']!=execution_fingerprint()['sha256']:
                raise RuntimeError('Resume requires the exact saved controller revision')
            if state['state'] in ('operator_paused','budget_paused','memory_paused','interrupted') and not resume:
                return state
            if resume:self.pause.unlink(missing_ok=True)
            deadline=time.monotonic()+state['options']['hours']*3600
            def stopped():
                if cancelled():state['pause_kind']='operator';return True
                if time.monotonic()>=deadline:state['pause_kind']='budget';return True
                return self.pause.exists()
            state.update(state='running',reason=None,pause_kind=None);self.save(state)
            if state.get('validation_pending'):
                self._qualify(state,stopped,progress);return state
            while state['completed_attempts']+8<=state['options']['attempts']:
                if stopped():raise InterruptedError('Обучение сохранено на паузе')
                batch=state.get('batch')
                if batch is None:
                    population=propose(state)
                    # Paired candidates use identical scenes. User basket and
                    # varied socks are followed by the unchanged wider profile.
                    offset=(state['generation']%3)*10
                    seeds=([state['seed']+4+offset,state['seed']+14+offset]
                        if state['generation']<3 else [state['seed']+100+state['generation']*2+i for i in range(2)])
                    specs=[self._spec(state,params,i,seed) for i,params in enumerate(population) for seed in seeds]
                    batch=dict(population=population,specs=specs,results=[]);state['batch']=batch;self.save(state)
                done={r['episode_id'] for r in state['episodes'] if r['status']=='complete'}
                finished=[r for r in state['episodes'] if r['episode_id'] in {s['episode_id'] for s in batch['specs']} and r['status']=='complete']
                specs=[]
                for spec in batch['specs']:
                    if spec['episode_id'] in done:continue
                    if Path(spec['folder']).exists():
                        # Preserve a partial recording; retry in a NEW folder.
                        old=spec['episode_id'];fresh=self._spec(state,spec['parameters'],spec['candidate'],spec['scene_seed'],
                            split=spec.get('split','train'))
                        state.setdefault('partial_attempts',[]).append(dict(episode_id=old,folder=spec['folder']))
                        spec.update(fresh)
                    specs.append(spec)
                self.save(state)
                results=finished+self._batch(state,specs,stopped,progress)
                all_success=update_search(state,results)
                state.setdefault('generations',[]).append(dict(index=state['generation'],selection=state['last_selection'],
                    episode_ids=[r['episode_id'] for r in results],phase_group=state['phase_group']))
                state['generation']+=1;state.pop('batch',None)
                checkpoint=self.folder/'checkpoints'/f'generation-{state["generation"]:04d}.json'
                checkpoint.parent.mkdir(exist_ok=True);write_json(checkpoint,state)
                write_json(self.folder/'policy.json',dict(schema=1,algorithm=state['algorithm'],task_id=state['task_id'],
                    scene_recipe=state.get('scene_recipe'),observation_contract='registered_RGBD_own_FK_odometry_lidar',
                    parameters=state['incumbent'],controller_sha256=state['controller']['sha256'],
                    simulation_ready=False,physical_ready=False))
                self.save(state);progress(state)
                if all_success:
                    self._qualify(state,stopped,progress);return state
                if state['stagnation']>=8:
                    state.update(state='needs_diagnosis',reason='Восемь разных поисковых раундов не улучшили результат; бесконечный повтор остановлен')
                    self.save(state);return state
            state.update(state='budget_paused',reason='Бюджет попыток завершён; все результаты и лучшая политика сохранены')
        except InterruptedError:
            kind=state.get('pause_kind') or 'operator'
            state.update(state=kind+'_paused',reason='Работники остановлены; прогресс сохранён')
        except KeyboardInterrupt:
            self.pause.touch(exist_ok=True);state.update(state='interrupted',reason='Вычисления прерваны; результаты сохранены')
            raise
        except Exception as error:
            self.pause.touch(exist_ok=True);state.update(state='failed',reason=str(error));self.save(state);raise
        finally:
            state['active_workers']=[];self.save(state);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        return state

    def _qualify(self,state,cancelled,progress):
        from .curriculum import report_passes
        state['state']='validating';state['validation_pending']=True
        state.setdefault('validation',[]);state.setdefault('validation_batches',[]);self.save(state)
        training={r['seed'] for r in state['episodes'] if r.get('split','train')=='train'}
        reserved=[seed for group in state['validation_seeds'] for seed in group]
        if len(set(reserved))!=10*len(state['validation_seeds']) or training.intersection(reserved):raise RuntimeError('Validation scene overlap')
        for index,seeds in enumerate(state['validation_seeds']):
            if len(state['validation'])>index:continue
            if len(state['validation_batches'])<=index:
                state['validation_batches'].append([self._spec(state,state['incumbent'],0,seed,
                    split='validation' if index<2 else 'user_validation') for seed in seeds]);self.save(state)
            specs=state['validation_batches'][index]
            ids={s['episode_id'] for s in specs}
            rows=[r for r in state['episodes'] if r['episode_id'] in ids and r['status']=='complete']
            done={r['episode_id'] for r in rows};pending=[]
            for spec in specs:
                if spec['episode_id'] in done:continue
                if Path(spec['folder']).exists():
                    state.setdefault('partial_attempts',[]).append(dict(episode_id=spec['episode_id'],folder=spec['folder']))
                    spec.update(self._spec(state,spec['parameters'],0,spec['scene_seed'],split='validation' if index<2 else 'user_validation'))
                pending.append(spec)
            self.save(state)
            rows+=self._batch(state,pending,cancelled,progress)
            report=dict(heldout_independent=True,episodes=10,successes=sum(r['success'] for r in rows),
                scene_profile=state.get('scene_profile','sock_variants_v1') if index<2 else 'user_constraints',results=rows)
            state['validation'].append(report);self.save(state)
        passed=all(report_passes(r) for r in state['validation'])
        state.update(state='packaging' if passed else 'validation_failed',simulation_ready=False,validation_pending=passed,
            reason='Два независимых набора по десять сцен пройдены' if passed else 'Независимые сцены не пройдены; готовность не подтверждена',
            physical_ready=False,automatic_hardware_execution=False)
        if passed:
            policy=json.loads((self.folder/'policy.json').read_text());policy['simulation_ready']=True
            write_json(self.folder/'policy.json',policy)
            state['policy_sha256']=hashlib.sha256((self.folder/'policy.json').read_bytes()).hexdigest()
            state['archive']=self._archive(state,cancelled)
            state.update(state='simulation_qualified',simulation_ready=True,validation_pending=False)
        self.save(state)

    def _archive(self,state,cancelled=lambda:False):
        """Archive and re-read every frozen file; no transfer/hardware claim."""
        import shutil
        import tarfile
        source=Path(__file__).resolve().parents[1]
        def check():
            if cancelled():raise InterruptedError('Подготовка архива сохранена на паузе')
        frozen=self.folder/'accepted-source'
        for name,digest in state['controller']['files'].items():
            check()
            path=source/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Controller changed before packaging')
            target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(self.folder/'accepted-state.json',state)
        files={}
        for p in self.folder.rglob('*'):
            check()
            if p.is_file() and p.name not in ('state.json','pause.request','archive-manifest.json'):
                with p.open('rb') as handle:files[str(p.relative_to(self.folder))]=hashlib.file_digest(handle,'sha256').hexdigest()
        manifest=self.folder/'archive-manifest.json';write_json(manifest,dict(schema=1,files=files,
            controller_sha256=state['controller']['sha256'],physical_ready=False))
        files['archive-manifest.json']=hashlib.sha256(manifest.read_bytes()).hexdigest()
        output=self.root/'exports'/f'observed-skill-{self.id}.tar.gz';output.parent.mkdir(exist_ok=True)
        temporary=output.with_suffix('.tmp')
        with tarfile.open(temporary,'w:gz') as archive:
            for name in sorted(files):check();archive.add(self.folder/name,arcname=name,recursive=False)
        seen=set()
        with tarfile.open(temporary,'r:gz') as archive:
            for member in archive:
                check()
                if not member.isfile() or member.name not in files or member.name in seen:
                    raise RuntimeError('Unexpected archive member')
                handle=archive.extractfile(member)
                digest=hashlib.file_digest(handle,'sha256').hexdigest()
                if digest!=files[member.name]:raise RuntimeError('Archive content checksum mismatch')
                seen.add(member.name)
        if seen!=set(files):raise RuntimeError('Archive incomplete')
        temporary.replace(output)
        with output.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        return dict(path=str(output),sha256=digest,files=len(files),verified=True,kind='simulation_skill_policy')
