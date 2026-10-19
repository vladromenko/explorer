"""Automatic room-specific demonstration generation and causal correction.

The geometric demonstrator exists ONLY in this simulation data worker. It is
never imported by the hardware actor, used as a validation fallback, or counted
as a learned success. Training inputs remain RGB, commanded state and clock.
"""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .environment import PERIOD
from .learning import write_json,read_bundle,sha256
from .room_policy import issued_command,place_clock,mission_limits
from .neural_recovery import Recovery,training_seeds,fresh_seeds,persisted_seeds


def diagnose(rows):
    findings=[]
    for row in rows:
        if row.get('success') is True:continue
        stages=[event['stage'] for event in row.get('events',[])]
        reason=str(row.get('reason','Unconfirmed physical result'))
        last=stages[-1] if stages else None
        if row.get('collision_free') is False and last in ('policy_pick_running','policy_place_running'):cause='neural_collision'
        elif row.get('collision_free') is False:cause='collision'
        elif last in ('policy_place_running','policy_place','verify_place'):cause='placement'
        elif last in ('policy_pick_running','policy_pick','verify_hold'):cause='pickup_or_transport'
        elif any(s.startswith('approach_') for s in stages):cause='alignment_or_reacquisition'
        else:cause='search_or_perception'
        findings.append(dict(seed=row['seed'],cause=cause,reason=reason,
            last_stage=stages[-1] if stages else None,
            evidence='completed_rollout_and_independent_outcome',
            root_cause_proven=False))
        if row.get('policy_command_refusal'):
            findings[-1].update(command_refusal=row['policy_command_refusal'],
                mechanism='neural_command_rejected_before_actuation',
                unsafe_prediction_is_training_target=False)
    causes=sorted({r['cause'] for r in findings})
    # A gradient update cannot repair a blocked route, detector service or a
    # collision guard. Only observed manipulation failures request ACT data.
    trainable=bool(findings) and all(r['cause'] in ('pickup_or_transport','placement','neural_collision') for r in findings)
    return dict(findings=findings,causes=causes,neural_correction_supported=trainable,
        action='collect_observed_expert_corrections' if trainable else 'diagnose_executor_without_retraining')


def correction_plan(diagnosis,excluded,start,count=24):
    if not diagnosis['neural_correction_supported']:raise ValueError('This failure is not addressed by neural retraining')
    from .neural_recovery import fresh_seeds
    failed=list(dict.fromkeys(r['seed'] for r in diagnosis['findings'] if r['seed'] not in excluded))[:count//2]
    fresh=fresh_seeds(set(excluded)|set(failed),start,count)
    seeds=failed+fresh[:count-len(failed)]
    return [dict(seed=seed,from_failure=seed in failed) for seed in seeds]


class PhaseRecorder:
    """Pre-action RGB/command paired with the endpoint actually applied."""
    def __init__(self,env,folder,profile,seed):
        self.env,self.folder,self.profile,self.seed=env,Path(folder),profile,seed
        self.original=env.step;self.active=None;self.phases=[];self.residual=np.zeros(6)
        env.step=self.step

    def begin(self,phase,target):
        if self.active is not None:raise ValueError('Overlapping teacher phase recording')
        import hashlib
        identity=hashlib.sha256(str(self.folder.resolve()).encode()).hexdigest()[:12]
        folder=self.folder/('room-expert-'+identity+'-'+phase)
        folder.mkdir(parents=True,exist_ok=False);(folder/'frames').mkdir()
        clock=self.profile['pick_elapsed_s'] if phase=='pick' else place_clock(self.profile,target,self.env.command)
        self.active=dict(folder=folder,phase=phase,clock=clock,index=0,expert_targets=True,handle=(folder/'samples.jsonl').open('w'))
        self.residual[:]=0

    def targets(self,expert):
        if type(expert) is not bool or self.active is None:raise ValueError('No active recording target mode')
        self.active['expert_targets']=expert

    def end(self):
        if self.active is not None:
            self.active['handle'].close();self.phases.append(self.active);self.active=None

    def step(self,action,*args,**kwargs):
        if self.active is None:return self.original(action,*args,**kwargs)
        from src.mobile_policy_contract import CONTRACT_SHA256
        item=self.active;env=self.env;index=item['index'];image=f'frames/{index:06d}.jpg'
        row=dict(at=time.time(),simulation_time=float(env.data.time),observation=env.command.tolist(),
            wrist_rgb=image,phase='approach' if item['phase']=='pick' else 'place',
            actor_elapsed_s=item['clock']+index*PERIOD,training_target_valid=item['expert_targets'],
            sample_contract_sha256=CONTRACT_SHA256,action_alignment='same_row_pre_action')
        Image.fromarray(env.render_rgb('wrist')).save(item['folder']/image,quality=92)
        command=issued_command(env.command,np.asarray(action,float),self.residual)
        result=self.original(command,applied=True)
        row['action']=env.command.tolist();item['handle'].write(json.dumps(row)+'\n');item['index']+=1
        return result

    def finish(self,success,outcome):
        from src.mobile_policy_contract import CONTRACT_SHA256
        self.end();self.env.step=self.original
        paths=[]
        for item in self.phases:
            manifest=dict(id=item['folder'].name,source='simulation',state='complete',
                task='Носок → корзина',task_id='sock_basket',scene_profile='sock_variants_v1',
                seed=self.seed,sample_period_s=PERIOD,fps=round(1/PERIOD),
                sample_contract_sha256=CONTRACT_SHA256,action_alignment='same_row_pre_action',
                generator='room_observed_geometric_teacher',teacher_profile='room_observed_v1',
                corrective_demonstration=True,teacher_fallback=False,attempt_completed=True,
                success=success,physics_verified_success=success,outcome=outcome,
                domain_variation=self.env.domain_variation,phase=item['phase'],
                teacher_inputs='RGBD_own_FK_and_observed_targets',privileged_actor_inputs=False,
                room_scene=True,full_mission_success=success,learner_intervention=item.get('intervention'))
            write_json(item['folder']/'episode.json',manifest)
            if success and item['index']>=3:
                rows=[json.loads(line) for line in (item['folder']/'samples.jsonl').read_text().splitlines()]
                if sum(row['training_target_valid'] for row in rows)>=3:paths.append(str(item['folder']))
        return paths


def demonstrate(spec):
    """Automatically generate a complete observed mission, not a replay."""
    import signal
    import torch
    from .environment import ExplorerEnv
    from .room_cleanup_port import RoomCleanupPort
    from .room_validation import room_scene,CollisionAudit,independent_outcome
    from src.room_cleanup import RoomCleanup
    torch.set_num_threads(1)
    folder=Path(spec['output']);folder.mkdir(parents=True,exist_ok=True)
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping)
    cancelled=lambda:stop or Path(spec['stop']).exists()
    env=ExplorerEnv(config={'room_layout':True});port=None;recorder=None
    preview_at=-1.
    def progress(value):
        nonlocal preview_at
        write_json(folder/'progress.json',value)
        from .worker_views import capture_overview
        stamp=capture_overview(env,folder,preview_at)
        if stamp!=preview_at:
            preview=folder/'wrist-preview.tmp';Image.fromarray(env.render_rgb('wrist')).save(preview,format='JPEG',quality=75)
            preview.replace(folder/'wrist-preview.jpg')
        preview_at=stamp
    try:
        env.reset(seed=spec['seed'],options=dict(task_id='sock_basket',scene_profile='sock_variants_v1',generated_scene=room_scene(spec['seed'])))
        audit=CollisionAudit(env);env._outcome_observer=audit.sample
        profile=json.loads(Path(spec['profile']).read_text())
        class TrainingPort(RoomCleanupPort):
            def load_actor(self,model,profile,*,device):
                if spec.get('student_steps',0):return super().load_actor(model,profile,device=device)
                from types import SimpleNamespace
                bundle,_=read_bundle(model)
                return SimpleNamespace(sha=bundle['checkpoint_sha256']['model.safetensors'])
        port=TrainingPort(env,folder/'observed',spec['model'],profile,spec['root'],cancelled=cancelled,
            progress=progress)
        recorder=PhaseRecorder(env,folder,profile,spec['seed'])
        geometry=port.geometry
        from .failure_repair import program
        geometry.controller_program=program(dict(schema=1,operators=[
            'cartesian_feedback','surface_axis_grasp','kinematic_hold_probe',
            'extended_hold_probe','visibility_planned_lift','reacquire_surface']))
        # These checked generators are data sources only. Qualification still
        # invokes RoomCleanupPort.learned_manipulation with the actual ACT.
        def expert(phase,target,check):
            ctx=port._ctx(check);recorder.begin(phase,target)
            def require(result):
                if result.get('outcome')!='success':raise ValueError(result.get('reason','Expert action unconfirmed'))
            try:
                if phase=='pick':
                    from .grasp_planning import observed_pick_patch
                    if spec.get('student_steps',0):
                        from .room_intervention import student_prefix,reacquire_pick
                        student_prefix(port,recorder,target,spec['student_steps'],check)
                        if not port.held:target=reacquire_pick(port,target,check)
                        recorder.targets(True)
                    geometry.target=target
                    if not port.held:
                        geometry.grasp_xyz=observed_pick_patch(geometry.depth_records[-1],target,env.base_pose(),fraction=.25,lateral_fraction=.5)
                        require(geometry.pregrasp(ctx));require(geometry.grasp(ctx));require(geometry.lift(ctx))
                    require(geometry.verify_hold(ctx));port.held=True
                    geometry.pick_lift_height=.42
                    require(geometry.raise_verified_hold(ctx))
                    if not port._transport_ready():raise ValueError('Observed teacher did not reach the required transport envelope')
                    require(geometry.verify_stationary_hold(ctx))
                else:
                    geometry.destination=target
                    require(geometry.support(ctx));require(geometry.release(ctx));require(geometry.withdraw(ctx))
                    require(geometry.verify_place(ctx));port.held=False
            finally:recorder.end()
        class TrainingMission(RoomCleanup):
            def manipulate(self,phase,target):
                self.event('expert_'+phase+'_running');expert(phase,target,self.check)
                self.event('expert_'+phase,dict(training_only=True))
        result=TrainingMission(port,lambda v:write_json(folder/'mission-progress.json',v),cancelled=cancelled,
            limits=mission_limits(profile)).run()
        truth=independent_outcome(env)  # evaluator only, after the complete mission stopped
        success=result['state']=='succeeded' and truth['success'] is True and not audit.contacts
        episodes=recorder.finish(success,truth)
        result.update(success=success,physical_outcome=truth,seed=spec['seed'],
            collision_free=not audit.contacts,collision_evidence=audit.contacts,
            episodes=episodes,generator='room_observed_geometric_teacher',
            student_interventions=[item['intervention'] for item in recorder.phases if item.get('intervention')],
            neural_qualification=False,physical_ready=False)
        from .policy_trace import encode_trace
        import subprocess
        result['training_videos']=[]
        for trace in sorted((folder/'observed').glob('policy-training-pick-prefix-*')):
            try:result['training_videos'].append(dict(encode_trace(trace),training_only=True))
            except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as error:
                result['training_videos'].append(dict(trace=str(trace),encoding_error=str(error),raw_frames_retained=True))
        write_json(folder/'result.json',result);return result
    finally:
        if recorder is not None and env.step!=recorder.original:recorder.finish(False,{})
        if port is not None:port.close()
        env.close()


def historical_episodes(model,root):
    """Replay exactly compatible old data, including the inherited validation."""
    root,model=Path(root),Path(model);read_bundle(model)
    from .episode_lineage import recording_index,trained_recording
    lookup=recording_index(root);digests={}
    selected=[]
    for partition in ('train','validation'):
        provenance=json.loads((model/partition/'explorer-provenance.json').read_text())
        for row in provenance['episodes']:
            path=trained_recording(row,lookup,digests)
            selected.append(path)
    return sorted(set(selected))


class RoomLearning(Recovery):
    """Durable finite correction → dataset → ACT → actual room qualification.

    Parent owns the same compute lease as local recovery. Its worker scheduler
    admits four data actors only against the measured capacity and free memory;
    exactly one gradient learner writes each new checkpoint.
    """
    worker_module='simulator.room_validation'
    training_timeout_s=14*24*3600
    stage_timeout_s=15*24*3600

    def __init__(self,root,folder,state,save,excluded):
        self.root,self.folder,self.state=Path(root),Path(folder),state
        self.stop=self.folder/'pause.request';self.children={};self.save=save
        self.excluded=set(excluded)
        from .executor_recovery import program
        saved=self.folder/'executor-program.json'
        self.executor_program=program(json.loads(saved.read_text())['operators']) if saved.exists() else []
        for p in self.folder.glob('round-*/acceptance*-seeds.json'):
            self.excluded.update(json.loads(p.read_text()))

    def evaluate_room(self,model,profile,seeds,stage):
        if set(seeds)&training_seeds(model):raise ValueError('Room evaluation overlaps model data')
        rows=self.jobs([dict(kind='room_evaluate',root=str(self.root),model=str(model),profile=str(profile),
            seed=seed,output=str(stage/f'scene-{index:02d}')) for index,seed in enumerate(seeds)],parallel=1)
        return dict(results=rows,episodes=len(rows),successes=sum(r.get('success') is True for r in rows))

    def jobs(self,specs,parallel=None):
        if parallel is None:
            from .parallel_trials import memory_free_percent
            capacity=self.root/'parallel-capacity.json'
            measured=json.loads(capacity.read_text()).get('assessment',{}) if capacity.exists() else {}
            allowed=4 if measured.get('fits') and measured.get('workers',0)>=4 and memory_free_percent()>=40 else 2
            parallel=min(allowed,self.state.get('parallel_ceiling',4))
        return super().jobs(specs,parallel=parallel)

    def profile(self,model,path):
        from .room_policy import build_profile
        if not path.exists():
            value=build_profile(model,self.root,path);value['executor_recovery']=self.executor_program
            write_json(path,value)
        return path

    def repair_executor(self,model,profile,seeds,baseline,stage):
        """A measured repair beats the control on the SAME ten full scenes."""
        from .executor_recovery import propose,comparison,OPERATORS
        if not profile.exists():return None
        promoted=None
        for attempt in range(len(OPERATORS)):
            value=json.loads(profile.read_text());diagnosis=diagnose(baseline['results'])
            proposal=propose(value.get('executor_recovery',[]),diagnosis)
            if proposal is None:break
            folder=stage/'executor-repair'/f'attempt-{attempt:02d}';folder.mkdir(parents=True,exist_ok=True);path=folder/'handoff.json'
            candidate=dict(value,executor_recovery=proposal)
            if path.exists() and json.loads(path.read_text())!=candidate:
                raise ValueError('Executor repair contract changed; preserve evidence and use a new repair identity')
            write_json(path,candidate);write_json(folder/'diagnosis.json',diagnosis)
            self.save(phase='executor_repair',message='Проверяем исправление поиска/подхода против прежнего исполнителя на тех же десяти сценах')
            repaired=self.evaluate_room(model,path,seeds,folder/'paired')
            checked=comparison(baseline,repaired);write_json(folder/'comparison.json',checked)
            if not checked['accepted']:break
            self.executor_program=proposal
            write_json(self.folder/'executor-program.json',dict(operators=proposal,paired_evidence=str(folder/'comparison.json'),
                simulation_qualified=False,physical_ready=False))
            baseline,profile=repaired,path;promoted=(baseline,profile)
        return promoted

    def used_seeds(self):
        used=set(self.excluded)
        for p in self.folder.glob('**/result.json'):used|=persisted_seeds(json.loads(p.read_text()))
        return used

    def run_learning(self,initial_model):
        import secrets
        from .checkpoint_experiment import shortlist,stage_candidate
        model=Path(initial_model)
        for iteration in range(4):
            self.check();stage=self.folder/f'round-{iteration:02d}';stage.mkdir(exist_ok=True)
            plan_path=stage/'plan.json'
            if not plan_path.exists():
                seeds=fresh_seeds(self.used_seeds()|training_seeds(model),1601000000+iteration*100000,10)
                write_json(plan_path,dict(model=str(model),seeds=seeds))
            plan=json.loads(plan_path.read_text());model=Path(plan['model'])
            profile=self.profile(model,stage/'handoff.json')
            self.save(state='running',phase='room_diagnostics',iteration=iteration+1,model=str(model),reason=None,
                message='Проверяем полный поиск и уборку фактической ACT без экспертной подмены')
            baseline=self.evaluate_room(model,profile,plan['seeds'],stage/'baseline')
            write_json(stage/'baseline.json',baseline)
            if baseline['successes']<10:
                repaired=self.repair_executor(model,profile,plan['seeds'],baseline,stage)
                if repaired is not None:baseline,profile=repaired
            if baseline['successes']==10:
                candidate=model;candidate_profile=profile
            else:
                diagnosis=diagnose(baseline['results']);write_json(stage/'diagnosis.json',diagnosis)
                if not diagnosis['neural_correction_supported']:
                    self.save(state='needs_room_algorithm_repair',phase='diagnosis',diagnosis=diagnosis,
                        reason='Нужна правка указанного механизма поиска/подъезда/контроля; повторное обучение весов не скроет эту причину')
                    return None
                if iteration==3:
                    self.save(state='needs_room_algorithm_repair',phase='diagnosis',diagnosis=diagnosis,
                        reason='Три исправляющих раунда завершены; физические неудачи сохранены, готовность не объявлена')
                    return None
                curriculum_path=stage/'curriculum.json'
                if not curriculum_path.exists():
                    write_json(curriculum_path,correction_plan(diagnosis,self.excluded|training_seeds(model),
                        1601004000+iteration*100000))
                curriculum=json.loads(curriculum_path.read_text())
                from .room_intervention import prefix_budget
                self.save(phase='room_corrections',message='Программа создаёт корректирующие показы в полных сценах; неудачи сохраняются отдельно')
                corrected=self.jobs([dict(kind='room_correct',root=str(self.root),model=str(model),profile=str(profile),
                    student_steps=prefix_budget(diagnosis,index),
                    seed=row['seed'],output=str(stage/'corrections'/f'scene-{index:02d}')) for index,row in enumerate(curriculum)])
                successful=[r for r in corrected if r.get('success') is True]
                recovered=[r for r in successful if any(i.get('applied_steps',0)>0 for i in r.get('student_interventions',[]))]
                write_json(stage/'correction-coverage.json',dict(full_successes=len(successful),
                    successful_student_state_corrections=len(recovered),minimum_full_successes=8,
                    minimum_student_state_corrections=4,qualification=False))
                if len(successful)<8 or len(recovered)<4:
                    teacher_diagnosis=diagnose(corrected);write_json(stage/'teacher-diagnosis.json',teacher_diagnosis)
                    self.save(state='needs_room_algorithm_repair',phase='teacher_diagnosis',diagnosis=teacher_diagnosis,
                        reason=f'Полных показов {len(successful)}/8, исправлений состояний ACT {len(recovered)}/4; недостаточно подтверждённых корректирующих целей')
                    return None
                selection_path=stage/'data-selection.json'
                if not selection_path.exists():
                    episodes=historical_episodes(model,self.root)+[Path(p) for r in successful for p in r['episodes']]
                    write_json(selection_path,dict(episodes=[str(p) for p in sorted(set(episodes))],
                        previous_data_preserved=True,new_full_mission_successes=len(successful),
                        correction_replays=4,minimum_epochs=8))
                selection=json.loads(selection_path.read_text())
                identity=stage/'training-id.json'
                if not identity.exists():write_json(identity,dict(id=secrets.token_hex(6)))
                training=self.root/'runs'/json.loads(identity.read_text())['id']
                if (training/'job.json').exists() and not (training/'model/checkpoints/last').exists() and not (training/'bundle.json').exists():
                    preserved=stage/'incomplete-exports.json'
                    previous=json.loads(preserved.read_text()) if preserved.exists() else []
                    previous.append(dict(id=training.name,reason='Export interrupted before optimizer checkpoint; original files retained'))
                    write_json(preserved,previous);write_json(identity,dict(id=secrets.token_hex(6)))
                    training=self.root/'runs'/json.loads(identity.read_text())['id']
                self.save(phase='train',training_model_id=training.name,
                    message='Одна ACT учится минимум восемь проходов по прежним данным и новым корректирующим показам')
                self.jobs([dict(kind='train',output=str(training),model=str(model),episodes=selection['episodes'],
                    seed=1601000000+iteration,task_id='sock_basket',title='Носок → корзина',
                    maximum_steps=1000000,
                    corrective_training_seeds=[r['seed'] for r in curriculum])],parallel=1)
                choices=[model]
                checkpoints=json.loads((training/'checkpoint-selection.json').read_text())
                for index,relative in enumerate(shortlist(checkpoints)):
                    path=stage/'candidates'/str(index)
                    if not path.exists():stage_candidate(training,relative,path)
                    choices.append(path)
                compare_path=stage/'comparison-seeds.json'
                if not compare_path.exists():write_json(compare_path,fresh_seeds(self.used_seeds()|
                    set().union(*(training_seeds(p) for p in choices)),1601006000+iteration*100000,10))
                seeds=json.loads(compare_path.read_text());scores=[]
                self.save(phase='room_checkpoint_comparison',message='Сравниваем checkpoint и прежнюю модель по завершению полной уборки на одинаковых новых сценах')
                for index,choice in enumerate(choices):
                    proposal=self.profile(choice,stage/f'candidate-{index}-handoff.json')
                    report=self.evaluate_room(choice,proposal,seeds,stage/'comparison'/str(index))
                    scores.append((choice,proposal,report))
                # Ties retain the unchanged reference; no offline-MAE promotion.
                candidate,candidate_profile,report=max(scores,key=lambda r:(r[2]['successes'],r[0]==model))
                write_json(stage/'comparison.json',dict(selected=str(candidate),results=[
                    dict(model=str(p),report=r) for p,_,r in scores],diagnostic_only=True))
                if report['successes']!=10:
                    # Next round diagnoses this winner on NEW independent scenes.
                    model=candidate;continue
            self.save(phase='room_validation',model=str(candidate),qualification_required=100,
                message='Итоговая проверка: две отдельные группы по 50 независимых полных уборок')
            protocol=stage/'acceptance100-protocol.json'
            expected=dict(schema='explorer_room_acceptance100_v1',groups=2,episodes_per_group=50,
                required_successes=100,allowed_collisions=0,task_timeout_s=1200,
                tuning_on_final_scenes=False,teacher_fallback=False,physical_ready=False)
            if protocol.exists() and json.loads(protocol.read_text())!=expected:
                raise ValueError('Final evaluation protocol changed; retain prior experiment')
            if not protocol.exists():write_json(protocol,expected)
            acceptance=stage/'acceptance100-seeds.json'
            if not acceptance.exists():write_json(acceptance,fresh_seeds(self.used_seeds()|training_seeds(candidate),
                1601008000+iteration*100000,100))
            seeds=json.loads(acceptance.read_text());groups=[]
            if len(seeds)!=100 or len(set(seeds))!=100:raise ValueError('Final evaluation needs 100 distinct scenes')
            for index in (0,1):
                report_path=stage/f'qualification100-{index+1}.json'
                report=self.evaluate_room(candidate,candidate_profile,seeds[index*50:(index+1)*50],stage/f'qualification100-{index+1}')
                write_json(report_path,report);groups.append(report_path)
            if all(json.loads(p.read_text())['successes']==50 for p in groups):
                return dict(model=candidate,profile=candidate_profile,groups=groups,
                    excluded=self.used_seeds()-set(seeds))
            # These failed acceptance scenes stay excluded from future training.
            self.excluded.update(seeds);model=candidate
        self.save(state='needs_room_algorithm_repair',phase='diagnosis',reason='Независимая проверка полного сценария не пройдена; данные и веса сохранены')
        return None
