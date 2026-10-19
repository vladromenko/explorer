"""Durable, cause-directed data aggregation and ACT fine-tuning on Mac only.

The expert is explicitly a privileged SIMULATION data generator. It is never
substituted for the actor during evaluation or included in a Jetson policy.
Student-induced deviations are recorded, but are not copied as expert targets.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

from .learning import read_bundle, write_json, sha256
from .checkpoint_experiment import stage_candidate, training_seeds, persisted_seeds, shortlist
from .parallel_trials import memory_free_percent


PHASES=('approach','transport','place','release')
TERMINAL=('ready_for_transfer','failed','needs_algorithm_repair','paused')


def diagnose(folder):
    """Post-rollout evidence only; never feeds object truth to an actor."""
    findings=[]
    for path in sorted(Path(folder).glob('**/episode.json')):
        episode=json.loads(path.read_text())
        if episode.get('generator')!='policy' or not episode.get('attempt_completed'):continue
        rows=[json.loads(line) for line in (path.parent/'samples.jsonl').read_text().splitlines()]
        if episode.get('error'):raise ValueError('Execution error requires a code fix: '+episode['error'])
        outcome=episode.get('outcome',{})
        if episode['success']:phase=None;reason='delivered'
        elif not outcome.get('lifted'):phase='approach';reason='no_verified_lift'
        else:
            last=rows[-1]['simulator_diagnostics'];placement=last.get('placement',{})
            import math
            distance=math.dist(last['sock_center_m'],last['jaw_center_m'])
            if placement.get('inside_fraction',0)>=.5:
                phase='release';reason='material_at_rim_without_completed_release'
            elif distance>.12:
                phase='transport';reason='lifted_then_material_left_gripper'
            else:phase='place';reason='held_without_delivery'
        findings.append(dict(seed=episode['seed'],success=episode['success'],phase=phase,
            evidence=reason,episode=str(path),error=episode.get('error')))
    if not findings:raise ValueError('No completed policy evidence')
    return dict(results=findings,failures={p:sum(r['phase']==p for r in findings) for p in PHASES},
        interpretation='observed_failure_location_not_proven_root_cause',privileged_diagnostics_only=True)


def curriculum(diagnosis,excluded,start,count=24):
    """Half failed scene identities; half fresh scenes, including the user rim."""
    failures=[r for r in diagnosis['results'] if r['phase'] and r['seed'] not in excluded]
    distinct={}
    for row in failures:distinct.setdefault(row['seed'],row)
    selected=[dict(seed=r['seed'],phase=r['phase'],from_failure=True) for r in list(distinct.values())[:count//2]]
    priorities=sorted(PHASES,key=lambda p:-diagnosis['failures'][p])
    used=set(excluded)|{r['seed'] for r in selected}
    # Reserve a quarter of fresh scenes for the exact user basket, while the
    # others still cover every height. No scene geometry is edited here.
    seed=start;fresh_index=0
    while len(selected)<count:
        if fresh_index%4==0:
            candidate=start-start%10+4+(fresh_index//4)*10
            while candidate in used:candidate+=10
        else:
            while seed in used:seed+=1
            candidate=seed;seed+=1
        selected.append(dict(seed=candidate,phase=priorities[len(selected)%len(priorities)],from_failure=False));used.add(candidate)
        fresh_index+=1
    return selected


def fresh_seeds(excluded,start,count):
    # Consecutive blocks of ten cover all heights and the user's round basket.
    seed=start-start%10
    while set(range(seed,seed+count)) & excluded:seed+=count
    return list(range(seed,seed+count))


def reserved_scene_seeds(root):
    root=Path(root);found=set()
    for pattern in ('learning-cycles/*.json','night-batches/*.json','checkpoint-experiments/*/state.json',
                    'feedback-audits/*/result.json','diagnostics/*/comparison.json',
                    'diagnostics/live-room-*/request.json','diagnostics/live-room-*/trial*/result.json',
                    'diagnostics/checkpoint-monitor-*/live-room-*/request.json',
                    'room-learning/**/result.json',
                    'neural-recovery/*/round-*/selection/*/result.json',
                    'neural-recovery/*/round-*/qualification-*/*/result.json'):
        for path in root.glob(pattern):found.update(persisted_seeds(json.loads(path.read_text())))
    for path in root.glob('room-learning/*/round-*/acceptance*-seeds.json'):
        seeds=json.loads(path.read_text())
        if not isinstance(seeds,list) or any(type(seed) is not int for seed in seeds):
            raise ValueError('Invalid reserved final scene ledger: '+str(path))
        found.update(seeds)
    return found


def owner_alive(pid,identifier):
    import psutil
    try:
        args=psutil.Process(pid).cmdline()
        return ('-m' in args and args[args.index('-m')+1]=='simulator.neural_recovery'
            and '--id' in args and args[args.index('--id')+1]==identifier)
    except (psutil.Error,IndexError,TypeError):return False


def current(root):
    try:
        marker=json.loads((Path(root)/'neural-recovery/current.json').read_text())
        identifier=marker['id']
        if len(identifier)!=12 or any(c not in '0123456789abcdef' for c in identifier):return None
        state=json.loads((Path(root)/'neural-recovery'/identifier/'state.json').read_text())
        state['alive']=owner_alive(state.get('pid'),identifier)
        return state
    except (OSError,ValueError,KeyError):return None


def worker(request):
    spec=json.loads(Path(request).read_text());folder=Path(spec['output'])
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    cancelled=lambda:stopping or Path(spec['stop']).exists()
    def progress(value):
        write_json(folder/'progress.json',value)
        if cancelled():raise InterruptedError('Recovery computation paused')
    if spec['kind']=='train':
        from .learning import train_job
        saved=folder/'job.json'
        previous=json.loads(saved.read_text()) if saved.exists() else None
        if previous and previous['state'] in ('trained_simulation_unvalidated','validated_simulation'):
            read_bundle(folder);result=previous
        else:
            resume=bool(previous and (folder/'model/checkpoints/last').exists())
            from .learning import prepared_training_available
            reuse_prepared=bool(previous and not resume and prepared_training_available(folder))
            if previous and not (resume or reuse_prepared):
                raise ValueError('Interrupted export retained; assign a new training identity')
            result=train_job(folder,[Path(p) for p in spec['episodes']],steps=1000,batch_size=32,
                seed=spec['seed'],skill_id=spec['task_id'],task=spec['title'],minimum_epochs=8,
                warm_start=spec['model'],resume=resume,reuse_prepared=reuse_prepared,correction_replays=4,
                maximum_steps=spec.get('maximum_steps',200000),
                corrective_training_seeds=spec.get('corrective_training_seeds',[]),progress=progress,cancelled=cancelled)
            if result['state'] not in ('trained_simulation_unvalidated','validated_simulation'):
                raise RuntimeError(result.get('error','Training failed'))
    else:
        import torch
        torch.set_num_threads(2)
        from .environment import ExplorerEnv
        from .learning import load_policy
        env=ExplorerEnv();actor=load_policy(spec['model'],device='cpu')
        try:
            bundle,_=read_bundle(spec['model'])
            if spec['kind']=='correct':
                episode_folder=folder/('recovery-'+folder.parent.parent.name+'-'+folder.parent.name+'-'+folder.name)
                result=env.make_episode_trial(episode_folder,seed=spec['seed'],task_id=bundle['skill_id'],
                    scene_profile=bundle['scene_profile'],teacher_profile='planned_release_v4',
                    recovery_policy=actor,recovery_phase=spec['phase'],recovery_steps=spec['intervention_steps'],progress=progress)
                result['source_model_sha256']=bundle['checkpoint_sha256']
                write_json(episode_folder/'episode.json',result)
                result['episode_folder']=str(episode_folder)
                if result.get('error') and not result.get('attempt_completed'):
                    raise InterruptedError(result['error'])
            elif spec['kind']=='evaluate':
                rows=[]
                for index,seed in enumerate(spec['seeds']):
                    if cancelled():raise InterruptedError('Evaluation paused')
                    episode=env.make_episode_trial(folder/f'scene-{index:02d}',seed=seed,policy=actor,
                        task_id=bundle['skill_id'],scene_profile=bundle['scene_profile'],progress=progress)
                    if episode.get('error'):raise RuntimeError(episode['error'])
                    rows.append({key:episode.get(key) for key in ('seed','success','outcome','domain_variation')})
                    progress(dict(completed=len(rows),total=len(spec['seeds'])))
                result=dict(model_id=Path(spec['model']).name,results=rows,seeds=spec['seeds'],
                    episodes=len(rows),successes=sum(r['success'] is True for r in rows),
                    task_id=bundle['skill_id'],scene_profile=bundle['scene_profile'],
                    validation_kind='closed_loop_policy',model_weights_sha256=bundle['checkpoint_sha256']['model.safetensors'],
                    model_checkpoint_sha256=bundle['checkpoint_sha256'],heldout_independent=True,
                    physics_validated=False,real_hardware_accepted=False,teacher_fallback=False)
            else:raise ValueError('Unknown recovery worker')
        finally:env.close()
    write_json(folder/'result.json',result)


class Recovery:
    def __init__(self,root,identifier):
        self.root=Path(root);self.folder=self.root/'neural-recovery'/identifier
        self.state=json.loads((self.folder/'state.json').read_text())
        self.stop=self.folder/'pause.request'
        self.children={}

    def save(self,**values):
        self.state.update(values,at=time.time(),pid=os.getpid())
        write_json(self.folder/'state.json',self.state)

    def check(self):
        if self.stop.exists():raise InterruptedError('Пауза пользователя; данные сохранены')
        if not self.root.parent.is_mount():raise InterruptedError('Внешний диск отключён')
        if shutil.disk_usage(self.root).free<16*1024**3:raise InterruptedError('Недостаточно свободного места; данные сохранены')

    def jobs(self,specs,parallel=None):
        pending=[];results={}
        for spec in specs:
            output=Path(spec['output']);result=output/'result.json'
            if result.exists():
                expected=dict(spec,stop=str(self.stop))
                if json.loads((output/'request.json').read_text())!=expected:
                    raise ValueError('Completed stage contract changed; retain evidence and use a new identity')
                results[str(output)]=json.loads(result.read_text());continue
            if output.exists() and spec['kind']!='train':
                output.rename(output.with_name(output.name+'-interrupted-'+str(time.time_ns())))
            output.mkdir(parents=True,exist_ok=True)
            spec=dict(spec,stop=str(self.stop));write_json(output/'request.json',spec);pending.append(spec)
        if not pending:return [results[spec['output']] for spec in specs]
        if parallel is None:
            capacity=self.root/'parallel-capacity.json'
            measured=json.loads(capacity.read_text()).get('assessment',{}) if capacity.exists() else {}
            parallel=4 if measured.get('fits') and measured.get('workers',0)>=4 and memory_free_percent()>=40 else 2
        self.save(workers=parallel)
        started=time.monotonic()
        while pending or self.children:
            self.check()
            free=memory_free_percent()
            if free<10:raise InterruptedError('Недостаточно памяти; сохранена пауза')
            while pending and len(self.children)<parallel and free>=25:
                spec=pending.pop(0);output=Path(spec['output']);log=(output/'worker.log').open('ab')
                child=subprocess.Popen([sys.executable,'-m',getattr(self,'worker_module','simulator.neural_recovery'),'worker',str(output/'request.json')],
                    stdout=log,stderr=log,start_new_session=True,
                    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
                self.children[str(output)]=dict(child=child,log=log,spec=spec,started=time.monotonic())
                free=memory_free_percent()
            details={}
            for key,item in list(self.children.items()):
                child=item['child'];path=Path(key)
                if child.poll() is not None:
                    if child.returncode:raise RuntimeError('Исполнитель завершился с ошибкой: '+str(path/'worker.log'))
                    results[key]=json.loads((path/'result.json').read_text());item['log'].close();del self.children[key]
                else:
                    progress=path/'progress.json'
                    if progress.exists():details=json.loads(progress.read_text())
                    limit=getattr(self,'training_timeout_s',32*3600) if item['spec']['kind']=='train' else 1800
                    if time.monotonic()-item['started']>limit:raise TimeoutError('Worker exceeded its finite budget: '+key)
            views=[]
            for item in self.children.values():
                output=Path(item['spec']['output'])
                folders=sorted(output.glob('recovery-*')) if item['spec']['kind']=='correct' else sorted(output.glob('scene-*'))
                if item['spec']['kind'] in ('room_correct','room_evaluate'):folders=[output]
                if folders:views.append(dict(folder=str(folders[-1]),live=True,label='Корректирующий показ' if item['spec']['kind'] in ('correct','room_correct') else 'Самостоятельная проверка'))
            self.save(active_pids=[i['child'].pid for i in self.children.values()],views=views,
                completed=len(results),total=len(specs),details=details,
                waiting_for_memory=bool(pending and not self.children))
            if time.monotonic()-started>getattr(self,'stage_timeout_s',36*3600):raise TimeoutError('Stage budget exhausted; evidence retained')
            time.sleep(1)
        return [results[spec['output']] for spec in specs]

    def shutdown(self):
        # Train workers handle SIGTERM and stop their separately grouped learner
        # before returning. No process outside this owner's children is killed.
        for item in self.children.values():
            if item['child'].poll() is None:item['child'].terminate()
        for item in self.children.values():
            try:item['child'].wait(timeout=15)
            except subprocess.TimeoutExpired:
                raise RuntimeError('Owned worker has not exited; do not resume or eject disk')
            finally:item['log'].close()

    def evaluate(self,models,seeds,stage):
        for model in models:
            if training_seeds(model)&set(seeds):raise ValueError('Evaluation overlaps train or recorded validation')
        specs=[dict(kind='evaluate',model=str(m),output=str(stage/m.name),seeds=seeds) for m in models]
        return self.jobs(specs)

    def run(self):
        if self.state['state']=='ready_for_transfer':return
        with (self.root/'neural-recovery/owner.lock').open('a') as lease, (self.root/'skill-training/owner.lock').open('a') as skills:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(skills,fcntl.LOCK_EX|fcntl.LOCK_NB)
            prior=self.state.get('active_pids',[])
            for pid in prior:
                try:os.kill(pid,0)
                except ProcessLookupError:continue
                raise RuntimeError('Surviving child found; no duplicate recovery started')
            self.save(state='running',active_pids=[],reason=None)
            def pause(*_):self.stop.touch()
            signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
            awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
            try:
                self._run()
            except InterruptedError as error:self.save(state='pausing',reason=str(error))
            except Exception as error:self.save(state='failed',reason=str(error));raise
            finally:
                self.shutdown();self.save(active_pids=[],state='paused' if self.state['state']=='pausing' else self.state['state'])
                if awake.poll() is None:awake.terminate()

    def _run(self):
        from .curriculum import task_spec,report_passes
        from .transfer import export_package,export_curriculum,import_archive
        import tempfile
        spec=task_spec(self.state['task_id'])
        model=Path(self.state['model'])
        for iteration in range(self.state.get('iteration',1),4):
            self.check();round_folder=self.folder/f'round-{iteration:02d}';round_folder.mkdir(exist_ok=True)
            round_input=round_folder/'round-input.json'
            if not round_input.exists():write_json(round_input,dict(model=str(model),evidence=self.state['evidence']))
            frozen=json.loads(round_input.read_text());model=Path(frozen['model'])
            self.save(iteration=iteration,phase='diagnosis',message='Разбираем физические срывы и формируем корректирующие показы')
            evidence=Path(frozen['evidence'])
            diagnosis=diagnose(evidence);write_json(round_folder/'diagnosis.json',diagnosis)
            heldout=json.loads((model/'validation/explorer-provenance.json').read_text())
            excluded={r['episode']['seed'] for r in heldout['episodes']}
            data_plan=curriculum(diagnosis,excluded,self.state['seed_base']+iteration*10000+4000)
            write_json(round_folder/'curriculum.json',data_plan)
            self.save(phase='trials',message='Эксперт исправляет отклонения, созданные текущей нейросетью; её ошибочные команды исключаются из целей обучения')
            previous=[]
            if iteration>1:
                for path in (self.folder/f'round-{iteration-1:02d}').glob('correction-*/result.json'):
                    previous.append(json.loads(path.read_text()))
            strengths={}
            for phase in PHASES:
                rows=[r for r in previous if r.get('learner_intervention',{}).get('phase')==phase]
                rate=sum(r.get('success') is True for r in rows)/len(rows) if rows else None
                # If the expert cannot recover the deviation, collect nearer
                # the recoverable boundary; if it can, expand visited states.
                strengths[phase]=10 if rate is not None and rate<.5 else 40 if rate is not None and rate>=.8 else 20
            write_json(round_folder/'intervention-strengths.json',dict(steps=strengths,
                rule='expand_successful_recovery_regions_shorten_unrecoverable_handoffs',previous_attempts=len(previous)))
            corrected=self.jobs([dict(kind='correct',output=str(round_folder/f'correction-{i:02d}'),
                model=str(model),seed=row['seed'],phase=row['phase'],intervention_steps=strengths[row['phase']])
                for i,row in enumerate(data_plan)])
            successful=[Path(row['episode_folder']) for row in corrected
                if row.get('success') is True and row.get('learner_intervention')]
            if len(successful)<8:
                raise RuntimeError(f'Only {len(successful)} successful recovery demonstrations; expert needs diagnosis, not blind neural training')
            # All compatible historical successes are replayed, not discarded.
            episodes=[]
            for path in (self.root/'demonstrations').glob('*/episode.json'):
                value=json.loads(path.read_text())
                if (value.get('task_id','sock_basket')==spec['id'] and value.get('scene_profile','baseline')==self.state['scene_profile']
                        and value.get('success') is True and value.get('attempt_completed',True)):
                    episodes.append(path.parent)
            for previous in (self.root/'neural-recovery').glob('*/round-*/correction-*/recovery-*/episode.json'):
                value=json.loads(previous.read_text())
                if (value.get('success') is True and value.get('learner_intervention') and value.get('task_id')==spec['id']
                        and value.get('scene_profile')==self.state['scene_profile']):episodes.append(previous.parent)
            data_selection=round_folder/'data-selection.json'
            if data_selection.exists():
                episodes=[Path(p) for p in json.loads(data_selection.read_text())['episodes']]
            else:
                episodes=sorted(set(episodes));write_json(data_selection,dict(
                    episodes=[str(p) for p in episodes],new_recovery_successes=len(successful),
                    correction_replays=4,minimum_full_dataset_passes=8,failures_preserved=True))
            training_path=round_folder/'training-id.json'
            if not training_path.exists():write_json(training_path,dict(id=secrets.token_hex(6)))
            training=self.root/'runs'/json.loads(training_path.read_text())['id']
            from .learning import prepared_training_available
            if ((training/'job.json').exists() and not (training/'model/checkpoints/last').exists()
                    and not (training/'bundle.json').exists() and not prepared_training_available(training)):
                retained=round_folder/'incomplete-exports.json'
                previous=json.loads(retained.read_text()) if retained.exists() else []
                previous.append(dict(id=training.name,reason='Export interrupted before optimizer checkpoint; original files retained'))
                write_json(retained,previous);write_json(training_path,dict(id=secrets.token_hex(6)))
                training=self.root/'runs'/json.loads(training_path.read_text())['id']
            self.save(phase='train',message='Дообучаем ACT на прежних успехах и новых исправлениях; корректирующие примеры имеют вес 4',training_model_id=training.name)
            self.jobs([dict(kind='train',output=str(training),model=str(model),episodes=[str(p) for p in episodes],
                seed=self.state['seed_base']+iteration,task_id=spec['id'],title=spec['title'],
                corrective_training_seeds=[r['seed'] for r in data_plan if r['from_failure']])],parallel=1)
            # Recorded MAE only shortlists diverse checkpoints. Actual task
            # completion, including the unchanged reference, chooses the model.
            self.save(phase='compare',message='Выбираем веса по самостоятельным доставкам, а не по ошибке повторения команд')
            selection=json.loads((training/'checkpoint-selection.json').read_text())
            choices=[model]
            for i,relative in enumerate(shortlist(selection)):
                target=round_folder/'candidates'/f'{i:02d}'
                if not target.exists():stage_candidate(training,relative,target)
                choices.append(target)
            excluded=set().union(*(training_seeds(p) for p in choices))
            excluded.update(reserved_scene_seeds(self.root))
            for p in self.folder.glob('**/result.json'):excluded.update(persisted_seeds(json.loads(p.read_text())))
            seeds_path=round_folder/'selection-seeds.json'
            if not seeds_path.exists():write_json(seeds_path,fresh_seeds(excluded,self.state['seed_base']+iteration*10000,10))
            seeds=json.loads(seeds_path.read_text())
            selection_results=self.evaluate(choices,seeds,round_folder/'selection')
            baseline=selection_results[0]
            best=max(zip(choices,selection_results),key=lambda pair:(pair[1]['successes'],pair[0]==model))
            chosen,result=best
            write_json(round_folder/'selection.json',dict(selected=str(chosen),results=selection_results,
                improved=result['successes']>baseline['successes'],diagnostic_only=True))
            self.state.setdefault('history',[]).append(dict(iteration=iteration,reference=baseline['successes'],
                candidate=result['successes'],episodes=10,selected=str(chosen),train_id=training.name))
            if chosen!=model:
                promoted=self.root/'runs'/secrets.token_hex(6)
                state=json.loads((chosen/'job.json').read_text())
                stage_candidate(training,state['source_checkpoint_relative'],promoted)
                model=promoted
            self.save(model=str(model),evidence=str(round_folder/'selection'/chosen.name),history=self.state['history'])
            # Two fresh, disjoint groups must pass on the exact frozen weights.
            if result['successes']>=8:
                self.save(phase='validate',message='Две независимые группы по десять сцен; без подмены нейросети экспертом')
                final_seeds=fresh_seeds(excluded|set(seeds),self.state['seed_base']+iteration*10000+2000,20)
                reports=[]
                for group in range(2):
                    report=self.evaluate([model],final_seeds[group*10:(group+1)*10],round_folder/f'qualification-{group}')[0]
                    reports.append(report)
                    if not report_passes(report):break
                if len(reports)==2 and all(report_passes(r) for r in reports):
                    if set(reports[0]['seeds'])&set(reports[1]['seeds']):raise ValueError('Final groups overlap')
                    bundle,_=read_bundle(model)
                    if any(r['model_checkpoint_sha256']!=bundle['checkpoint_sha256'] for r in reports):raise ValueError('Qualification checkpoint changed')
                    write_json(model/'physics-validation.json',reports[-1])
                    write_json(model/'qualification-evidence.json',dict(groups=reports,code_revision=self.state['code_revision'],
                        source_model_sha256=bundle['checkpoint_sha256'],independent_groups=True))
                    package=export_package(model,self.root/'exports'/f'recovery-{self.folder.name}.tar.gz')
                    archive=export_curriculum(dict(id=self.state['cycle_id'],tasks=[dict(task_id=spec['id'],state='complete',
                        stages=dict(export=package,validate=reports[-1]))]),self.root/'exports'/f'cycle-{self.state["cycle_id"]}-recovery.tar.gz')
                    with tempfile.TemporaryDirectory(prefix='verify-recovery-',dir=self.folder) as temporary:
                        imported=import_archive(archive['path'],Path(temporary)/'inactive',register=False)
                        if len(imported['tasks'])!=1 or not imported['tasks'][0]['job']['simulation_qualification']['simulation_qualified']:
                            raise ValueError('Imported package did not preserve simulation qualification')
                        write_json(self.folder/'archive-verification.json',dict(archive=archive,import_result=imported))
                    cycle_path=self.root/'learning-cycles'/(self.state['cycle_id']+'.json')
                    cycle=json.loads(cycle_path.read_text())
                    write_json(self.folder/'source-cycle-before-qualification.json',cycle)
                    cycle.update(state='ready_for_transfer',phase='ready',simulation_qualified=True,physical_ready=False,
                        archive=archive,reason=None,recovery_id=self.folder.name)
                    task=next(t for t in cycle['tasks'] if t['task_id']==spec['id'])
                    task.update(state='ready_for_transfer',model_id=model.name,archive=package,validation=reports[-1],simulation_qualified=True)
                    write_json(cycle_path,cycle)
                    self.save(state='ready_for_transfer',phase='ready',archive=archive,model=str(model),simulation_ready=True,physical_ready=False)
                    return
                # Failed qualification scenes become diagnostic material only.
                self.save(evidence=str(round_folder/f'qualification-{len(reports)-1}'/model.name),message='Независимая проверка выявила срывы; добавляем их в следующий учебный раунд')
            self.save(iteration=iteration+1)
        self.save(state='needs_algorithm_repair',phase='diagnosis',reason='Три корректирующих раунда исчерпаны; повторы без нового исправления запрещены')


def create(root,experiment,cycle_id,revision):
    root=Path(root);experiment=Path(experiment);value=json.loads((experiment/'state.json').read_text())
    if not root.parent.is_mount():raise ValueError('External volume missing')
    existing=current(root)
    if existing and existing.get('alive') and existing['state'] not in TERMINAL:raise ValueError('Recovery owner already active')
    if value.get('state')!='complete' or not value.get('diagnostic_only'):raise ValueError('Wait for the checkpoint experiment')
    selected=value['selected_model_id']
    source=root/'runs'/value['model_id']
    if selected==value['reference_id']:model=root/'runs'/selected
    else:
        candidate=experiment/'candidates'/selected
        state=json.loads((candidate/'job.json').read_text());model=root/'runs'/secrets.token_hex(6)
        stage_candidate(source,state['source_checkpoint_relative'],model)
    identifier=secrets.token_hex(6);folder=root/'neural-recovery'/identifier;folder.mkdir(parents=True)
    state=dict(id=identifier,cycle_id=cycle_id,task_id='sock_basket',scene_profile='sock_variants_v1',
        state='queued',phase='diagnosis',created=time.time(),source_experiment=str(experiment),
        model=str(model),evidence=str(experiment/f'candidate-{int(selected)+1:02d}') if selected!=value['reference_id'] else str(experiment/'candidate-00'),
        seed_base=1400000000+secrets.randbelow(20000000)//100000*100000,
        code_revision=revision,physical_ready=False,simulation_ready=False,history=[],active_pids=[])
    write_json(folder/'state.json',state);write_json(root/'neural-recovery/current.json',dict(id=identifier))
    return identifier


if __name__=='__main__':
    if sys.argv[1:2]==['worker']:worker(sys.argv[2])
    else:
        parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
        parser.add_argument('--id',required=True);options=parser.parse_args()
        Recovery(options.root,options.id).run()
