"""Durable, finite black-box policy learning from observed skill rollouts.

The learner changes bounded control parameters. It never changes the model,
physics, collision margins or evidence gates. Actors see sensors and their own
kinematics; reward reads outcome booleans only AFTER their actions stop.
"""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

import numpy as np

from .resource_budget import memory_budget
from .learning import write_json
from .skill_parameters import PARAMETERS,parameters
from .skill_workshop import execution_fingerprint
from .worker_runtime import WorkerStalled,stalled,terminate_pool,EpisodePool

ADAPTIVE={'observed_curriculum_cross_entropy_v3','observed_causal_program_repair_v4'}
CURRICULUM=ADAPTIVE|{'observed_curriculum_cross_entropy_v2'}
REPAIR='observed_causal_program_repair_v4'


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
    if stage=='carry':
        reason=str(report.get('reason','')).lower()
        return 'view' if any(s in reason for s in ('маршрут','лидар','проход','положения корзины')) else 'pick'
    if stage in ('support','release','withdraw','verify_place'):return 'place'
    return 'pick'


def phase_credit(report):
    """Diagnostic learning feedback AFTER stopping, not acceptance evidence.

    A previously placed object on one easy scene must not outweigh a better
    pickup on the hard scene. Command completion provides small shaping
    credits; verified physical pickup has a separate, protected tier.
    """
    execution=report.get('execution',{});truth=report.get('independent_physics_outcome',{})
    passed={e['stage'] for e in execution.get('events',[]) if e['result'].get('outcome')=='success'}
    pick=sum(s in passed for s in ('find','approach','reobserve','pregrasp','grasp','lift'))
    if truth.get('lifted') is True:pick=8
    if execution.get('held_verified') is True and truth.get('lifted') is True:pick=9
    place=sum(s in passed for s in ('support','release','withdraw'))
    if truth.get('success') is True:place=4
    if report.get('success') is True and execution.get('placed_verified') is True and truth.get('success') is True:place=5
    return dict(pick=float(pick),view=float(sum(s in passed for s in ('find','carry'))),place=float(place))


def search_mode(state):
    """A stalled controller changes its search subspace, not its physics."""
    group=state['phase_group'];rounds=state.get('phase_stagnation',{}).get(group,0)
    if group=='pick':
        modes=(('geometry',('pick_fraction','pick_lateral_fraction','pick_standoff_m','pick_approach_angle_deg')),
               ('contact',('pick_fraction','pick_lateral_fraction','pick_surface_offset_m','closure_duration_s')),
               ('hold_view',('initial_lift_height_m','lift_clearance_m','pick_fraction')))
    elif group=='place':
        modes=(('inspection',('reorientation_distance_m','reorientation_camera_height_m','destination_standoff_m',
                              'verification_distance_m','verification_aim_fraction')),
               ('release',('release_fraction','release_margin_m','release_roll_deg')),
               ('withdraw',('withdraw_height_m','verification_distance_m','verification_aim_fraction')))
    else:modes=(('view',('view_baseline_m','inspection_reach_m','inspection_height_m')),)
    return modes[(rounds//2)%len(modes)]


def propose(state):
    """Cross-entropy proposals only for the empirically blocked skill group."""
    incumbent=parameters(state['incumbent']);group=state['phase_group']
    rng=np.random.default_rng(state['seed']+state['generation']*1009)
    population=[incumbent]
    adaptive=state.get('algorithm') in ADAPTIVE
    mode,names=search_mode(state) if adaptive else ('legacy',tuple(k for k,s in PARAMETERS.items() if s[3]==group))
    if adaptive:state['search_mode']=mode
    for index in range(3):
        choice=dict(incumbent)
        for name,(lo,hi,_,phase) in PARAMETERS.items():
            if phase!=group or name not in names:continue
            # Regular exploration across the full permitted range prevents
            # a collapsed distribution from repeating identical failed moves.
            value=(rng.uniform(lo,hi) if index==2 or state['stagnation']>=3 else
                   rng.normal(state['mean'][name],state['std'][name]))
            choice[name]=float(np.clip(value,lo,hi))
        population.append(parameters(choice))
    return population


def update_search(state,reports,*,preview=False):
    searched_group=state['phase_group']
    grouped={i:[] for i in range(4)}
    for row in reports:grouped[row['candidate']].append(row)
    if any(len(rows)!=2 for rows in grouped.values()):raise ValueError('Paired search batch incomplete')
    causal=state.get('algorithm')==REPAIR and state['batch'].get('repair_programs')
    if causal:
        from .failure_repair import program
        populations=state['batch']['population']
        if any(p!=populations[0] for p in populations):raise RuntimeError('Repair attribution requires identical numeric policy')
        seeds={r['seed'] for r in grouped[0]}
        if len(seeds)!=2 or any({r['seed'] for r in rows}!=seeds for rows in grouped.values()):
            raise RuntimeError('Repair attribution requires identical paired scene seeds')
        if causal[0]!=program(state.get('controller_program')):raise RuntimeError('Causal control differs from incumbent program')
    adaptive=state.get('algorithm') in ADAPTIVE
    def rank(index):
        scores=[row['score'] for row in grouped[index]]
        if adaptive:
            credits=sorted(row.get('phase_credit',{}).get(searched_group,row['score']) for row in grouped[index])
            return (*credits,min(scores),sum(scores)/len(scores))
        return min(scores),sum(scores)/len(scores)
    baseline=rank(0)
    def preserves_learned_pick(index):
        # Preserve a proven pickup on EACH paired scene; downstream scores
        # can vary while refining the blocked phase. No seed enters the actor.
        if not adaptive:return True
        for base,row in zip(sorted(grouped[0],key=lambda r:r.get('seed',0)),sorted(grouped[index],key=lambda r:r.get('seed',0))):
            if base.get('phase_credit',{}).get('pick',0)>=9 and row.get('phase_credit',{}).get('pick',0)<9:return False
        return True
    rejected=set(state['batch'].get('rejected_candidates',[]))
    if state['batch'].get('repair_programs'):
        programs=state['batch']['repair_programs']
        for i in range(1,len(programs)):
            if programs[i] in programs[:i]:rejected.add(i)
    eligible=[i for i in grouped if i not in rejected and rank(i)[0]>=baseline[0] and preserves_learned_pick(i)]
    chosen=max(eligible,key=lambda i:(rank(i),-i) if adaptive else (rank(i)[1],rank(i)[0],-i))
    improved=rank(chosen)>baseline if adaptive else rank(chosen)[1]>baseline[1]
    if causal and improved and causal[chosen]!=causal[0] and not preview:
        confirmation=state['batch'].get('confirmations',{}).get(str(chosen),{})
        if confirmation.get('status')!='accepted':raise RuntimeError('Repair promotion requires new-scene confirmation')
    if improved:
        state['incumbent']=dict(state['batch']['population'][chosen]);state['stagnation']=0
    else:state['stagnation']+=1
    # Lack of progress in grasp, navigation and placement are different
    # diagnoses. The former shared counter stopped after eight TOTAL rounds,
    # even if a newly reached stage had barely been explored.
    counters=state.setdefault('phase_stagnation',{})
    counters[searched_group]=0 if improved else counters.get(searched_group,0)+1
    elite=sorted(eligible,key=rank,reverse=True)[:2]
    for name,(lo,hi,_,phase) in PARAMETERS.items():
        if phase!=state['phase_group']:continue
        values=[state['batch']['population'][i][name] for i in elite]
        state['mean'][name]=float(np.mean(values))
        state['std'][name]=float(np.clip(np.std(values),.07*(hi-lo),.4*(hi-lo)))
        if state['stagnation']>=3:state['std'][name]=.35*(hi-lo)
    chosen_rows=grouped[chosen]
    groups=Counter(row['failure_group'] for row in chosen_rows if row['success'] is not True)
    if groups:
        state['phase_group']=(next(g for g in ('pick','view','place') if g in groups) if adaptive else groups.most_common(1)[0][0])
    state['stagnation']=counters.get(state['phase_group'],0)
    state['last_selection']=dict(candidate=chosen,improved=improved,
        baseline_scores=[r['score'] for r in grouped[0]],selected_scores=[r['score'] for r in chosen_rows])
    if adaptive:
        state['last_selection'].update(objective='lexicographic_worst_phase_first',phase=searched_group,
            baseline_credit=list(baseline),selected_credit=list(rank(chosen)),search_mode=state.get('search_mode'))
        # Keep a bounded archive of empirically useful policies and the
        # concrete failure that caused automatic exploration/restart.
        if improved:
            archive=state.setdefault('skill_archive',{}).setdefault(searched_group,[])
            archive.append(dict(generation=state['generation'],parameters=dict(state['incumbent']),
                episode_ids=[r.get('episode_id') for r in chosen_rows],credit=list(rank(chosen))))
            del archive[:-16]
        if not improved and counters[searched_group]%2==0:
            action=dict(generation=state['generation'],phase=searched_group,
                from_mode=state.get('search_mode'),next_mode=search_mode(state)[0],
                reasons=dict(Counter(str(r.get('reason','unknown')) for r in chosen_rows if not r['success'])),
                action='change_search_subspace_and_expand',evidence_thresholds_changed=False)
            state.setdefault('recovery_history',[]).append(action)
            for name,(lo,hi,_,phase) in PARAMETERS.items():
                if phase==searched_group:state['std'][name]=.4*(hi-lo)
    if state.get('algorithm')==REPAIR and state['batch'].get('repair_programs'):
        from .failure_repair import record_trial
        record_trial(state,state['batch']['repair_programs'],chosen,improved)
        state['last_selection'].update(causal_comparison='same_scenes_same_numeric_policy',
            selected_program=state.get('controller_program'))
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
    write_json(folder/'worker.json',dict(pid=os.getpid(),owner_pid=spec['owner_pid'],started_at=time.time()))
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
        experience=None
        if spec.get('experience_path'):
            from .experience_memory import checked_memory,actor_prior
            memory=checked_memory(spec['experience_path'])
            if memory['sha256']!=spec['experience_sha256']:raise RuntimeError('Experience changed before worker start')
            experience=actor_prior(memory)
        report=trial(env,folder,'attempt',spec.get('task_id','sock_basket'),spec['scene_seed'],
            spec.get('scene_profile','sock_variants_v1'),cancelled,progress,parameters=spec['parameters'],
            scene_recipe=spec.get('scene_recipe'),semantic_root=spec.get('semantic_root'),
            validation=spec.get('split','train')=='validation',repair_program=spec.get('controller_program'),experience=experience)
        # Outcome and scene provenance become feedback only after trial stops.
        result=dict(episode_id=spec['episode_id'],candidate=spec['candidate'],seed=spec['scene_seed'],
            split=spec.get('split','train'),
            status='paused' if report['state']=='paused' else 'complete',success=report.get('success') is True,
            score=outcome_score(report),phase_credit=phase_credit(report),failure_group=failure_group(report),stage=report.get('phase'),
            reason=report.get('reason'),report=str(folder/'skill-workshops/attempt/state.json'),
            controller_program=report.get('controller_program'),
            domain_variation=env.domain_variation)
        import resource
        import sys
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_rss_bytes']=peak if sys.platform=='darwin' else peak*1024
        commands=folder/'skill-workshops/attempt/execution/commands.jsonl'
        result['trajectory_sha256']=hashlib.sha256(commands.read_bytes()).hexdigest()
        from .failure_repair import infrastructure_failure
        runtime_failure=infrastructure_failure(report)
        if runtime_failure and result['status']=='complete':
            result.update(status='infrastructure_failed',infrastructure_failure=runtime_failure)
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
            algorithm=REPAIR,seed=seed,generation=0,
            options=dict(attempts=attempts,hours=hours,workers=workers),phase_group='pick',stagnation=0,
            phase_stagnation={},curriculum_pairs_passed=0,
            incumbent=values,mean=dict(values),std={k:(s[1]-s[0])*.30 for k,s in PARAMETERS.items()},
            episodes=[],active_workers=[],completed_attempts=0,simulation_ready=False,physical_ready=False,
            automatic_hardware_execution=False,controller=execution_fingerprint(),
            validation_seeds=[list(range(1300160000,1300160010)),list(range(1400160000,1400160010))])
        from .failure_repair import program
        state.update(controller_program=program(),initial_attempt_budget=attempts,repair_trials=[],historical_diagnostics=[],historical_repair_trials=[])
        from .observed_transfer import runtime_snapshot
        state['runtime_files']=runtime_snapshot()
        state['minimum_curriculum_pairs']=16 if scene_recipe is not None else 3
        state['resource_objective']='more_diverse_full_fidelity_training_material'
        memory=self.root/'maintenance'/('historical-joint-memory-'+task_id+'.json')
        if memory.is_file():
            from .experience_memory import checked_memory
            experience=checked_memory(memory)
            write_json(self.folder/'experience.json',experience)
            state['experience']={key:experience[key] for key in ('sha256','ik_seeds','admitted','scanned')}
            state['experience']['source_seeds']=sorted({s['seed'] for s in experience['sources'] if type(s.get('seed')) is int})
            state['historical_cause_priors']=dict(Counter(s['cause_prior'] for s in experience.get('failure_summaries',[])))
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
                state['controller_program']=program(old.get('controller_program'))
                break
        if scene_recipe is not None and len(scene_recipe['request'])>1:
            state['validation_seeds'].append(list(range(1500160000,1500160010)))
        # Reuse previous post-stop failure evidence and numeric best policies
        # as UNVERIFIED starting hints. Old episodes never count as new tests.
        seeds=[]
        for old in list_trainings(self.root):
            if old.get('id')==self.id or old.get('task_id')!=task_id:continue
            if (old.get('scene_recipe') or {}).get('sha256')!=(scene_recipe or {}).get('sha256'):continue
            source=old.get('controller',{}).get('files',{})
            own_files=state['controller']['files']
            if any(source.get(n)!=own_files.get(n) for n in ('simulator/environment.py','simulator/explorer_model.py',
                    'simulator/model_config.json','config/explorer.urdf','config/handeye-accepted.json')):continue
            if not state.get('initialization') and old.get('algorithm') in ADAPTIVE|{REPAIR}:
                try:values=parameters(old['incumbent'])
                except (KeyError,ValueError):continue
                state.update(incumbent=values,mean=dict(values),initialization=dict(kind='unverified_historical_policy_hint',
                    from_training_id=old['id'],old_episodes_counted_for_qualification=False))
                state['controller_program']=program(old.get('controller_program'))
            for previous in old.get('repair_trials',[]):
                if len(state['historical_repair_trials'])>=64:break
                state['historical_repair_trials'].append(dict(program=program(previous.get('program')),
                    source_controller_sha256=old.get('controller',{}).get('sha256'),not_new_acceptance_evidence=True))
            for row in reversed(old.get('episodes',[])):
                if row.get('status')!='complete' or row.get('split','train')!='train' or row.get('success') is True:continue
                if len(state['historical_diagnostics'])<128:state['historical_diagnostics'].append(row)
                if type(row.get('seed')) is int and row['seed']<1_000_000_000 and row['seed'] not in seeds:seeds.append(row['seed'])
        state['historical_challenge_pairs']=[seeds[i:i+2] for i in range(0,min(6,len(seeds))-1,2)]
        self.save(state);return state

    def _batch(self,state,specs,cancelled,progress):
        # Recreate a hung native renderer without counting it as a failed
        # grasp or changing the scene, numeric policy or controller program.
        pending=list(specs);recovered=[]
        for restart in range(3):
            try:
                rows=recovered+self._batch_once(state,pending,cancelled,progress)
                if not recovered and not state.get('infrastructure_events'):
                    from .learning_capacity import calibrate
                    calibrate(self,state,specs,cancelled,progress)
                return rows
            except WorkerStalled as error:
                if cancelled() or self.pause.exists():raise InterruptedError('Пауза сохранена') from error
                done={r['episode_id']:r for r in state['episodes'] if r['status']=='complete'}
                retry=[]
                for spec in pending:
                    if spec['episode_id'] in done:
                        recovered.append(done[spec['episode_id']]);continue
                    old=dict(spec);folder=Path(old['folder']);folder.mkdir(parents=True,exist_ok=True)
                    report=folder/'runtime-failure.json'
                    write_json(report,dict(state='infrastructure_failed',reason='Renderer worker stalled: '+str(error),
                        actions_stopped=True,independent_physics_outcome={},physics_success_evaluated=False))
                    result=dict(episode_id=old['episode_id'],candidate=old['candidate'],seed=old['scene_seed'],
                        split=old.get('split','train'),status='infrastructure_failed',success=False,
                        reason='Renderer worker stalled',report=str(report),infrastructure_failure='renderer_stall')
                    state['episodes'].append(result);state.setdefault('infrastructure_events',[]).append(result)
                    fresh=self._spec(state,old['parameters'],old['candidate'],old['scene_seed'],
                        split=old.get('split','train'),controller_program=old.get('controller_program'))
                    spec.update(fresh);retry.append(spec)
                    state.setdefault('runtime_recoveries',[]).append(dict(cause='renderer_stall',old_episode=old['episode_id'],
                        replacement=spec['episode_id'],action='recreate_renderer_retry_identical_scene',
                        excluded_from_learning_reward=True))
                state.update(active_workers=[],effective_workers=1,stable_resource_batches=0,
                    reason='Восстанавливаю зависший процесс симуляции; записи сохранены')
                self.save(state);progress(state)
                if restart==2:raise RuntimeError('Рендерер завис повторно после двух восстановлений; записи сохранены') from error
                pending=retry
        raise AssertionError('Unreachable')

    def _pause_for_update(self,state):
        if not (self.folder/'update.request').exists() or self.pause.exists():return False
        state.update(state='maintenance_paused',active_workers=[],
            reason='Сравнение завершено и сохранено; подключается исправление программы')
        self.save(state);return True

    def _batch_once(self,state,specs,cancelled,progress):
        workers=(state['options']['workers'] if state.get('capacity_replay') else
            state.get('calibrated_workers',state['options']['workers']))
        pending=list(specs);results=[];active={}
        previous_state=state['state']
        reservation=state.get('measured_worker_reservation_bytes',int((3.2 if state.get('scene_recipe') else 1.7)*2**30))
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
        # A reused macOS GL process can retain a Metal shader-cache file lock
        # while idle and block another worker's first frame indefinitely.
        # Exit after each episode so all driver locks and contexts are released.
        with EpisodePool(max_workers=capacity) as pool:
            launch_limit=min(capacity,state.get('effective_workers',capacity))
            runtime_events_before=len(state.get('infrastructure_events',[]))
            cancelling_since=None
            try:
                while pending or active:
                    if cancelled():self.pause.touch(exist_ok=True)
                    memory=memory_budget()
                    state['minimum_memory_fraction']=min(state.get('minimum_memory_fraction',1.),memory['available_bytes']/memory['total_bytes'])
                    if memory['available_bytes']/memory['total_bytes']<.15:
                        state['pause_kind']='memory';self.pause.touch(exist_ok=True)
                    while pending and len(active)<launch_limit and not self.pause.exists():
                        spec=pending.pop(0);slot=next(i for i in range(capacity) if i not in [x['slot'] for x in active.values()])
                        spec['slot']=slot;spec['owner_pid']=os.getpid();spec['started_at']=time.time()
                        active[pool.submit(spec)]=spec
                    state['actual_peak_workers']=max(state.get('actual_peak_workers',0),len(active))
                    state['active_workers']=[dict(slot=s['slot'],episode_id=s['episode_id'],seed=s['scene_seed'],
                        candidate=s['candidate'],started_at=s['started_at'],
                        frame_folder=str(Path(s['folder']).relative_to(self.root/'skill-training'/self.id))) for s in active.values()]
                    self.save(state);progress(state)
                    if self.pause.exists():
                        cancelling_since=cancelling_since or time.monotonic()
                        if time.monotonic()-cancelling_since>10.:raise InterruptedError('Остановка зависшего исполнителя; записи сохранены')
                    if any(stalled(spec) for spec in active.values()):raise WorkerStalled('Нет кадров или прогресса более 240 секунд')
                    if not active:
                        if pending and not self.pause.exists():
                            state['pause_kind']='memory';self.pause.touch(exist_ok=True)
                        break
                    done=[future for future in active if future.done()]
                    if not done:time.sleep(.1)
                    for future in done:
                        spec=active.pop(future)
                        try:result=future.result()
                        except Exception:
                            self.pause.touch(exist_ok=True)
                            raise
                        state['episodes'].append(result)
                        if result.get('status')=='infrastructure_failed':
                            state.setdefault('infrastructure_events',[]).append(result)
                            retries=spec.get('infrastructure_retries',0)
                            if retries>=2:raise RuntimeError('Сбой вычислений повторился после восстановления: '+str(result.get('reason')))
                            old=dict(spec)
                            fresh=self._spec(state,spec['parameters'],spec['candidate'],spec['scene_seed'],
                                split=spec.get('split','train'),controller_program=spec.get('controller_program'))
                            fresh['infrastructure_retries']=retries+1;spec.update(fresh);pending.append(spec)
                            launch_limit=max(1,launch_limit//2);state['effective_workers']=launch_limit
                            state.setdefault('runtime_recoveries',[]).append(dict(cause=result['infrastructure_failure'],
                                old_episode=old['episode_id'],replacement=spec['episode_id'],
                                action='retry_same_scene_same_policy_reduce_parallelism',workers=launch_limit,
                                excluded_from_learning_reward=True))
                            self.save(state);continue
                        results.append(result)
                        state['completed_attempts']=sum(r['status']=='complete' and r.get('split','train')=='train'
                            for r in state['episodes'])
                        self.save(state)
                state['active_workers']=[];self.save(state)
            except WorkerStalled:
                terminate_pool(pool)
                raise
            except BaseException:
                # Set the cooperative flag before the pool context waits for
                # workers, including when the parent receives KeyboardInterrupt.
                self.pause.touch(exist_ok=True)
                terminate_pool(pool)
                raise
        if pending or any(r['status']!='complete' for r in results):raise InterruptedError('Обучение сохранено на паузе')
        if len(state.get('infrastructure_events',[]))==runtime_events_before:
            state['stable_resource_batches']=state.get('stable_resource_batches',0)+1
            if state['stable_resource_batches']>=3 and launch_limit<capacity:
                state['effective_workers']=min(capacity,launch_limit+1);state['stable_resource_batches']=0
        else:state['stable_resource_batches']=0
        return results

    def _spec(self,state,params,candidate,scene_seed,*,split='train',controller_program=None):
        token=secrets.token_hex(6)
        from .failure_repair import program
        return dict(episode_id=token,folder=str(self.folder/'episodes'/token),storage=str(self.root),
            pause_file=str(self.pause),parameters=parameters(params),candidate=candidate,scene_seed=scene_seed,
            controller_sha256=state['controller']['sha256'],split=split,owner_pid=os.getpid(),
            task_id=state['task_id'],scene_recipe=state.get('scene_recipe'),semantic_root=str(self.root),
            scene_profile=state.get('scene_profile','sock_variants_v1'),
            controller_program=program(controller_program if controller_program is not None else state.get('controller_program')),
            experience_path=str(self.folder/'experience.json') if state.get('experience') else None,
            experience_sha256=state.get('experience',{}).get('sha256'))

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
            if state['state'] in ('operator_paused','budget_paused','memory_paused','interrupted','maintenance_paused') and not resume:
                return state
            if resume:self.pause.unlink(missing_ok=True)
            deadline=time.monotonic()+state['options']['hours']*3600
            def stopped():
                if cancelled():state['pause_kind']='operator';return True
                if time.monotonic()>=deadline:state['pause_kind']='budget';return True
                return self.pause.exists()
            state.update(state='running',reason=None,pause_kind=None);self.save(state)
            if state.get('validation_pending'):
                self._qualify(state,stopped,progress)
                if not self._continue_after_validation(state):return state
            while True:
                if stopped():raise InterruptedError('Обучение сохранено на паузе')
                if state.get('batch') is None and state['completed_attempts']+8>state['options']['attempts']:
                    if not self._extend_for_repair(state):break
                batch=state.get('batch')
                if batch is None:
                    population=propose(state)
                    repair_programs=None
                    if state['algorithm']==REPAIR:
                        from .failure_repair import diagnosis,repair_proposals,repair_target,CAUSES
                        rows=state['episodes'][-64:] or state.get('historical_diagnostics',[])
                        analysis=diagnosis(rows,self.root);state['diagnosis']=analysis
                        known={c['code'] for c in analysis['causes']}
                        for code,count in state.get('historical_cause_priors',{}).items():
                            if code not in known and code in CAUSES:
                                analysis['causes'].append(dict(code=code,title=CAUSES[code][0],count=count,
                                    operators=list(CAUSES[code][1]),certainty='historical_hypothesis_requires_new_A_B_test'))
                        write_json(self.folder/'diagnosis.json',analysis)
                        if state.get('force_repair') or state.get('stagnation',0)>=2 or state['generation']==0:
                            repair_programs=repair_proposals(state,analysis)
                            if repair_programs:
                                population=[dict(state['incumbent']) for _ in range(4)]
                                state['search_mode']='causal_repair'
                                state['repair_target']=repair_target(repair_programs[0],repair_programs[1],analysis)
                        state.pop('force_repair',None)
                    # Paired candidates use identical scenes. User basket and
                    # varied socks are followed by the unchanged wider profile.
                    if state['algorithm'] in CURRICULUM:
                        # Keep a paired training challenge until it is solved.
                        # Otherwise every round changes BOTH the policy and
                        # its failure scene, destroying a meaningful curriculum.
                        offset=state.get('curriculum_pairs_passed',0)*100+state.get('validation_round',0)*1000
                        pairs=state.get('historical_challenge_pairs',[])
                        index=state.get('curriculum_pairs_passed',0)
                        seeds=(pairs[index] if state.get('validation_round',0)==0 and index<len(pairs)
                            else [state['seed']+4+offset,state['seed']+14+offset])
                    else:
                        offset=(state['generation']%3)*10
                        seeds=([state['seed']+4+offset,state['seed']+14+offset]
                            if state['generation']<3 else [state['seed']+100+state['generation']*2+i for i in range(2)])
                    specs=[self._spec(state,params,i,seed,controller_program=repair_programs[i] if repair_programs else None)
                        for i,params in enumerate(population) for seed in seeds]
                    batch=dict(population=population,specs=specs,results=[],repair_programs=repair_programs)
                    state['batch']=batch;self.save(state)
                done={r['episode_id'] for r in state['episodes'] if r['status']=='complete'}
                finished=[r for r in state['episodes'] if r['episode_id'] in {s['episode_id'] for s in batch['specs']} and r['status']=='complete']
                specs=[]
                for spec in batch['specs']:
                    if spec['episode_id'] in done:continue
                    if Path(spec['folder']).exists():
                        # Preserve a partial recording; retry in a NEW folder.
                        old=spec['episode_id'];fresh=self._spec(state,spec['parameters'],spec['candidate'],spec['scene_seed'],
                            split=spec.get('split','train'),controller_program=spec.get('controller_program'))
                        state.setdefault('partial_attempts',[]).append(dict(episode_id=old,folder=spec['folder']))
                        spec.update(fresh)
                    specs.append(spec)
                self.save(state)
                results=finished+self._batch(state,specs,stopped,progress)
                if self._pause_for_update(state):return state
                if batch.get('repair_programs'):
                    import copy
                    while True:
                        preview=copy.deepcopy(state);update_search(preview,results,preview=True)
                        winner=preview['last_selection']['candidate']
                        if not preview['last_selection']['improved'] or winner==0:break
                        if self._confirm_program(state,winner,stopped,progress):break
                        batch.setdefault('rejected_candidates',[]).append(winner);self.save(state)
                if self._pause_for_update(state):return state
                all_success=update_search(state,results)
                state.setdefault('generations',[]).append(dict(index=state['generation'],selection=state['last_selection'],
                    episode_ids=[r['episode_id'] for r in results],phase_group=state['phase_group']))
                state['generation']+=1;state.pop('batch',None)
                checkpoint=self.folder/'checkpoints'/f'generation-{state["generation"]:04d}.json'
                checkpoint.parent.mkdir(exist_ok=True);write_json(checkpoint,state)
                write_json(self.folder/'policy.json',dict(schema=1,algorithm=state['algorithm'],task_id=state['task_id'],
                    scene_recipe=state.get('scene_recipe'),observation_contract='registered_RGBD_own_FK_odometry_lidar',
                    parameters=state['incumbent'],controller_sha256=state['controller']['sha256'],
                    controller_program=state.get('controller_program'),experience_sha256=state.get('experience',{}).get('sha256'),
                    simulation_ready=False,physical_ready=False))
                self.save(state);progress(state)
                if all_success:
                    state['curriculum_pairs_passed']=state.get('curriculum_pairs_passed',0)+1
                    if state['algorithm'] not in CURRICULUM or state['curriculum_pairs_passed']>=state.get('minimum_curriculum_pairs',3):
                        self._qualify(state,stopped,progress)
                        if not self._continue_after_validation(state):return state
                        continue
                    state.update(phase_stagnation={},stagnation=0,phase_group='pick')
                    self.save(state)
                if state['stagnation']>=8 and state['algorithm'] not in ADAPTIVE:
                    state.update(state='needs_diagnosis',reason='Восемь поисковых раундов одного этапа не улучшили результат: '+state['phase_group']+'; требуется исправление причины')
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
            state['active_workers']=[]
            try:
                self.save(state)
                if state.get('algorithm')==REPAIR:
                    from .semantic_cache import stop_servers
                    remaining=stop_servers(self.root)
                    if remaining:state['model_cleanup_pending']=remaining;self.save(state)
            finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        return state

    def _confirm_program(self,state,candidate,cancelled,progress):
        """Replicate a proposed repair on two NEW training scenes before use."""
        from .failure_repair import digest,repair_target
        from .repair_acceptance import confirms_nonregression
        batch=state['batch'];choices=batch['repair_programs']
        ledger=batch.setdefault('confirmations',{})
        key=str(candidate)
        if key not in ledger:
            if state['completed_attempts']+4>state['options']['attempts']:
                extra=state['completed_attempts']+4-state['options']['attempts']
                if state['options']['attempts']+extra>state.get('quality_attempt_budget',state.get('initial_attempt_budget',1000))+256:return False
                state['options']['attempts']+=extra
            used={r['seed'] for r in state['episodes'] if type(r.get('seed')) is int}
            seed=200_000_000+int(state['id'],16)%650_000_000+state['generation']*20+candidate*2
            seeds=[]
            while len(seeds)<2:
                if seed not in used:seeds.append(seed)
                seed+=1
            specs=[self._spec(state,state['incumbent'],i,seed,controller_program=choices[index])
                for i,index in enumerate((0,candidate)) for seed in seeds]
            ledger[key]=dict(specs=specs,program_sha256=digest(choices[candidate]),
                split='train',status='pending',final_holdout=False)
            self.save(state)
        entry=ledger[key]
        if entry['status'] in ('accepted','rejected'):return entry['status']=='accepted'
        ids={s['episode_id'] for s in entry['specs']}
        rows=[r for r in state['episodes'] if r['episode_id'] in ids and r['status']=='complete']
        done={r['episode_id'] for r in rows};pending=[]
        for spec in entry['specs']:
            if spec['episode_id'] in done:continue
            if Path(spec['folder']).exists():
                state.setdefault('partial_attempts',[]).append(dict(episode_id=spec['episode_id'],folder=spec['folder']))
                spec.update(self._spec(state,spec['parameters'],spec['candidate'],spec['scene_seed'],
                    controller_program=spec['controller_program']))
            pending.append(spec)
        self.save(state);state['search_mode']='repair_confirmation'
        state['repair_target']=repair_target(choices[0],choices[candidate],state.get('diagnosis',{}))
        rows+=self._batch(state,pending,cancelled,progress)
        accepted,reason=confirms_nonregression(rows,state['phase_group'])
        entry.update(status='accepted' if accepted else 'rejected',reason=reason,
            episode_ids=[r['episode_id'] for r in rows],no_evidence_gates_changed=True)
        folder=self.folder/'repair-confirmations';folder.mkdir(exist_ok=True)
        write_json(folder/f'{state["generation"]:04d}-{candidate}.json',entry)
        self.save(state);return accepted

    def _extend_for_repair(self,state):
        """The initial search cap is not a terminal diagnosis for an untested fix.

        Allocate only finite, evidence-led A/B tests; never increase the wall
        time allowance, repeat an exhausted program set or reset user pauses.
        """
        if state.get('algorithm')!=REPAIR:return False
        if state.get('pause_kind')=='operator' or state.get('state')=='operator_paused' or self.pause.exists():return False
        from .failure_repair import diagnosis,repair_proposals
        if state['options']['attempts']>=state.get('quality_attempt_budget',state.get('initial_attempt_budget',1000))+256:return False
        analysis=diagnosis(state['episodes'][-64:],self.root)
        if repair_proposals(state,analysis) is None:return False
        state['options']['attempts']+=8;state['force_repair']=True
        state.setdefault('recovery_history',[]).append(dict(generation=state['generation'],
            action='allocate_causal_repair_trial',new_attempt_budget=state['options']['attempts'],
            wall_time_extended=False,evidence_thresholds_changed=False))
        self.save(state);return True

    def _continue_after_validation(self,state):
        """Failed tests trigger NEW training challenges, never test leakage.

        Quarantine used holdouts permanently. Do not fit to their poses or
        replay them as demonstrations. The next final check uses entirely new
        independent scenes and the SAME unchanged acceptance criteria.
        """
        if (state.get('algorithm') not in ADAPTIVE or
            state['state']!='validation_failed' or state['completed_attempts']+8>state['options']['attempts']):return False
        used=[r for group in state.get('validation',[]) for r in group.get('results',[])]
        failures=Counter(r['failure_group'] for r in used if not r.get('success') and r.get('failure_group'))
        state.setdefault('validation_history',[]).append(dict(round=state.get('validation_round',0),
            validation=state['validation'],seeds=state['validation_seeds'],parameters=dict(state['incumbent']),
            training_use_forbidden=True))
        state['validation_round']=state.get('validation_round',0)+1
        shift=state['validation_round']*1_000_000
        state['validation_seeds']=[list(range(base+shift,base+shift+10)) for base in
            (1300160000,1400160000,1500160000)[:len(state['validation_seeds'])]]
        phase=next((g for g in ('pick','view','place') if g in failures),'pick')
        state.update(state='running',validation=[],validation_batches=[],validation_pending=False,
            curriculum_pairs_passed=0,phase_group=phase,phase_stagnation={},stagnation=0,
            reason='Проверка выявила слабый этап; продолжаю на новых учебных сценах: '+phase)
        state.setdefault('recovery_history',[]).append(dict(generation=state['generation'],phase=phase,
            action='fresh_curriculum_after_failed_holdout',validation_round=state['validation_round'],
            old_holdouts_quarantined=True,evidence_thresholds_changed=False))
        self.save(state);return True

    def _qualify(self,state,cancelled,progress):
        from .curriculum import report_passes
        state['state']='validating';state['validation_pending']=True
        state.setdefault('validation',[]);state.setdefault('validation_batches',[]);self.save(state)
        training={r['seed'] for r in state['episodes'] if r.get('split','train')=='train'}
        training.update(state.get('experience',{}).get('source_seeds',[]))
        reserved=[seed for group in state['validation_seeds'] for seed in group]
        if len(set(reserved))!=10*len(state['validation_seeds']) or training.intersection(reserved):raise RuntimeError('Validation scene overlap')
        quarantined={seed for old in state.get('validation_history',[]) for group in old['seeds'] for seed in group}
        if quarantined.intersection(reserved):raise RuntimeError('Previously examined holdout reused for final qualification')
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
            if self._pause_for_update(state):return
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
        for name,digest in dict(state.get('runtime_files',{}),**state['controller']['files']).items():
            check()
            path=source/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Controller changed before packaging')
            target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(self.folder/'accepted-state.json',state)
        files={}
        for p in self.folder.rglob('*'):
            check()
            if p.is_file() and p.relative_to(self.folder).as_posix() not in ('state.json','pause.request','archive-manifest.json'):
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
