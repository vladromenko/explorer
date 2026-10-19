"""Explicit ACT handoff proposal for room navigation, not a local qualification.

Source examples propose a placement clock/pose after a long external drive.
This changes execution semantics and MUST pass separate room mission tests.
No trajectory from the dataset is replayed as manipulation commands.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .learning import read_bundle,write_json,sha256

# Stable repository-relative paths bind the composition, sensors, actuator
# semantics, geometry and live owner as well as the neural phase handoff.
RUNTIME_FILES=('src/room_cleanup.py','src/room_cleanup_rpc.py','src/room_cleanup_execution.py',
    'src/web.py','src/index.html','src/room-cleanup-ui.js','bin/prepare-room-runtime.py','bin/prepare-observed-runtime.py',
    'src/observed_skill_execution.py','src/mobile_policy_contract.py','src/simulation_task_contract.py','src/delivery_robot.py',
    'src/delivery_vision.py','src/factory_delivery.py','src/arm_commissioning.py','src/learning_environment.py',
    'src/task_executor.py','src/missions.py','src/map_tools.py','bin/room-cleanup-worker.py',
    'simulator/room_policy.py','simulator/room_cleanup_port.py','simulator/observed_robot.py',
    'simulator/grounded_skill_port.py','simulator/skill_port.py','simulator/inspection_planning.py',
    'simulator/skill_parameters.py','simulator/failure_repair.py','simulator/observed_free_space.py','simulator/resource_budget.py',
    'simulator/scan_navigation.py','simulator/observed_navigation.py','simulator/semantic_vision.py',
    'simulator/room_navigation.py',
    'simulator/semantic_cache.py','simulator/vision_backend.py','bin/simulator-grounding-service.py',
    'simulator/rgbd_observation.py','simulator/multiview_container.py','simulator/placement_evidence.py',
    'simulator/manipulation_planning.py','simulator/grasp_planning.py','simulator/repair_motion.py',
    'simulator/explorer_model.py','simulator/model_config.json','simulator/domain_randomization.py','simulator/curriculum.py','simulator/environment.py',
    'simulator/learning.py','simulator/transfer.py','simulator/room_validation.py','simulator/room_transfer.py',
    'simulator/room_archive.py','simulator/room_learning.py','simulator/room_control.py','simulator/executor_recovery.py','simulator/neural_recovery.py','simulator/episode_lineage.py',
    'simulator/server.py','simulator/training_activity.py','simulator/learning_status.py','simulator/static/app.js','simulator/worker_views.py',
    'simulator/checkpoint_experiment.py','simulator/parallel_trials.py',
    'src/frontiers.py','config/arm_model.json','config/arm-motion.json',
    'config/explorer.urdf','config/explorer.srdf','config/grounding-model.json')


def runtime_files():
    root=Path(__file__).resolve().parents[1]
    return [root/name for name in RUNTIME_FILES]


def build_profile(model,root,destination):
    model,root=Path(model),Path(root);bundle,_=read_bundle(model)
    provenance=json.loads((model/'train/explorer-provenance.json').read_text())
    from .episode_lineage import recording_index,trained_recording
    paths=recording_index(root);digests={}
    proposals=[];initial=[];pick_clocks=[]
    for source in provenance['episodes']:
        episode=source['episode'];folder=trained_recording(source,paths,digests)
        rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        pickup=next(((i,r) for i,r in enumerate(rows) if r.get('phase')=='approach' and r.get('training_target_valid',True)),None)
        if pickup:
            index,row=pickup;initial.append(row['observation'])
            pick_clocks.append(row.get('actor_elapsed_s',index*episode['sample_period_s']))
        first=next(((i,r) for i,r in enumerate(rows) if r.get('phase')=='place' and r.get('training_target_valid',True)),None)
        if first is None:continue
        index,row=first;bin_spec=episode['domain_variation']['parameters']['bin']
        proposals.append(dict(seed=episode['seed'],height_m=bin_spec['height_m'],shape=bin_spec['shape'],
            half_size_m=bin_spec['inner_half_size_m'],elapsed_s=row.get('actor_elapsed_s',index*episode['sample_period_s']),
            observed_command=row['observation'],source_sha256=source['input_sha256']['samples.jsonl']))
    if not proposals or not initial:raise ValueError('No verified phase-labelled training demonstrations')
    value=dict(schema='explorer_room_act_handoff_v1',model_checkpoint_sha256=bundle['checkpoint_sha256'],
        initial_command=np.median(initial,axis=0).tolist(),placement_proposals=proposals,
        pick_elapsed_s=float(np.median(pick_clocks)),source_standoff_m=.20,
        navigation_owner='observed_lidar_and_map_planner',actor_execution='arm_only_after_observed_alignment',
        actuator_quantization='integer_degree_error_diffusion_2deg_step',
        transport_height_m=.40,transport_maximum_reach_m=.28,
        executor_recovery=[],
        actor_inputs_unchanged=True,teacher_fallback=False,
        execution_changed=True,requires_new_room_qualification=True,simulation_qualified=False,
        place_handoff='nearest_training_geometry_and_command_then_fresh_ACT_inference',
        scope='sock_and_observed_basket_up_to_30cm_rim',physical_ready=False)
    write_json(destination,value);return value


def place_clock(profile,basket,command):
    lo,hi=np.asarray(basket['bounds_min']),np.asarray(basket['bounds_max'])
    half=(hi[:2]-lo[:2])/2;rim=float(hi[2])
    if not .06<=rim<=.31 or not .085<=half.min()<=half.max()<=.24:
        raise ValueError('Observed basket outside the trained handoff envelope')
    def score(row):
        geometry=abs(row['height_m']-rim)/.02+float(np.linalg.norm(np.asarray(row['half_size_m'])-half))/.03
        posture=float(np.linalg.norm((np.asarray(row['observed_command'])[:5]-np.asarray(command)[:5])/30.))
        return geometry+posture
    nearest=sorted(profile['placement_proposals'],key=score)[:5]
    return float(np.median([r['elapsed_s'] for r in nearest]))


def validate_profile(profile):
    from .executor_recovery import program
    program(profile.get('executor_recovery',[]))
    if (profile.get('schema')!='explorer_room_act_handoff_v1'
            or profile.get('navigation_owner')!='observed_lidar_and_map_planner'
            or profile.get('actor_execution')!='arm_only_after_observed_alignment'
            or profile.get('actuator_quantization')!='integer_degree_error_diffusion_2deg_step'
            or profile.get('transport_height_m')!=.40 or profile.get('transport_maximum_reach_m')!=.28
            or profile.get('source_standoff_m')!=.20 or profile.get('teacher_fallback') is not False
            or profile.get('actor_inputs_unchanged') is not True or profile.get('execution_changed') is not True
            or profile.get('requires_new_room_qualification') is not True or profile.get('physical_ready') is not False):
        raise ValueError('Unsupported or unbound room ACT composition')
    initial=np.asarray(profile.get('initial_command'),float)
    if initial.shape!=(9,) or not np.isfinite(initial).all() or abs(initial[5]-30)>.5 or np.any(abs(initial[6:])>.001):
        raise ValueError('Invalid recorded preparation pose')
    from src.mobile_policy_contract import bound_action
    bound_action(initial,initial)
    if not 0<=profile.get('pick_elapsed_s',-1)<=160:raise ValueError('Invalid pick clock')
    rows=profile.get('placement_proposals')
    if not rows:raise ValueError('No training-derived handoff proposals')
    for row in rows:
        values=np.r_[row['observed_command'],row['half_size_m'],row['height_m'],row['elapsed_s']]
        if len(row['observed_command'])!=9 or len(row['half_size_m'])!=2 or not np.isfinite(values).all() or not 0<=row['elapsed_s']<=160:
            raise ValueError('Invalid placement training proposal')


def mission_limits(profile):
    from src.room_cleanup import CleanupLimits
    from .executor_recovery import program
    operators=program(profile.get('executor_recovery',[]))
    return CleanupLimits(search_stops=36 if 'expanded_coverage' in operators else 24,
        realignments=8 if 'expanded_approaches' in operators else 4)


def issued_command(reference,prediction,residual=None):
    """Exactly the same issued limits and sequential arm/base mode in both ports."""
    from src.mobile_policy_contract import bound_action
    bounded=bound_action(reference,prediction)
    result=np.r_[bounded['issued_joint_goal_deg'],bounded['issued_body_velocity']]
    if residual is not None:
        from src.mobile_policy_contract import JOINT_LIMITS
        accumulated=np.asarray(reference)[:6]+np.clip(np.asarray(prediction)[:6]-np.asarray(reference)[:6],-2.,2.)+residual
        for i,(low,high) in enumerate(JOINT_LIMITS):
            if i==5:high=170
            result[i]=np.clip(round(float(accumulated[i])),max(low,np.ceil(reference[i]-2)),min(high,np.floor(reference[i]+2)))
        residual[:]=np.clip(accumulated-result[:6],-.5,.5)
    # Whole-room base motion belongs to the observed planner. The local
    # actor's old transport velocities cannot drive through an unseen room.
    result[6:]=0.
    return result


def admit_release(target,jaw):
    """ACT may request opening only over the freshly observed inner extent."""
    lo,hi=np.asarray(target['bounds_min']),np.asarray(target['bounds_max']);jaw=np.asarray(jaw)
    margin=float(target.get('uncertainty_m',1.))+.012
    if (lo.shape!=(3,) or hi.shape!=(3,) or jaw.shape!=(3,) or not np.isfinite(np.r_[lo,hi,jaw,margin]).all()
            or target.get('evidence',{}).get('shape')!='raised_walls_with_visible_opening'
            or margin>.032 or np.any(jaw[:2]<=lo[:2]+margin) or np.any(jaw[:2]>=hi[:2]-margin)
            or jaw[2]<hi[2]+.03):
        raise ValueError('Neural opening is outside the observed basket release envelope')


class RoomActor:
    def __init__(self,model,profile,*,device='cpu'):
        from .learning import load_policy
        validate_profile(profile)
        bundle,_=read_bundle(model)
        if profile['model_checkpoint_sha256']!=bundle.get('validation_checkpoint_sha256',bundle['checkpoint_sha256']):
            raise ValueError('Handoff belongs to a different neural checkpoint')
        self.actor=load_policy(model,device=device);self.profile=profile
        self.sha=bundle['checkpoint_sha256']['model.safetensors']

    def begin(self,phase,target,command):
        self.actor.reset()
        self.actor.elapsed_s=self.profile['pick_elapsed_s'] if phase=='pick' else place_clock(self.profile,target,command)
        self.replans=0;self.command_evidence=None

    def command(self,state,rgb,residual):
        """Never issue a refused command or weaken the original 20/2 deg gates."""
        from .environment import LOW,HIGH
        import hashlib
        self.command_evidence=dict(reference=np.asarray(state).tolist(),
            image_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),replanned=False)
        try:prediction=np.asarray(self.actor(state,rgb),float)
        except Exception as error:
            self.command_evidence['inference_refusal']=str(error);raise
        values=lambda array:[float(v) if np.isfinite(v) else str(v) for v in array]
        self.command_evidence.update(prediction=values(prediction),
            elapsed_s=self.actor.last_action_elapsed_s,queued=self.actor.last_action_was_queued)
        try:return issued_command(state,np.clip(prediction,LOW,HIGH),residual)
        except ValueError as error:
            self.command_evidence['refusal']=str(error)
            # Only a finite cached joint target can be stale relative
            # to a host-limited trajectory. Invalid new predictions still stop.
            bounded=np.clip(prediction,LOW,HIGH)
            if (not self.actor.last_action_was_queued or self.replans>=3 or
                    not np.isfinite(prediction).all() or not np.any(abs(bounded[:6]-np.asarray(state)[:6])>20)):
                raise
            fresh=np.asarray(self.actor.replan(state,rgb),float);self.replans+=1
            self.command_evidence.update(replanned=True,fresh_prediction=values(fresh),replans=self.replans)
            return issued_command(state,np.clip(fresh,LOW,HIGH),residual)

    def __call__(self,state,rgb):return self.actor(state,rgb)


def contract_sha(profile,code_files):
    payload=dict(profile=profile,code={'/'.join(Path(p).parts[-2:]):sha256(p) for p in code_files})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
