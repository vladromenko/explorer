"""Reuse old successful simulation records as IK priors, never as commands.

Streaming import deliberately whitelists phase and robot servo commands. It
does not import simulator_diagnostics, environmental vectors or object truth.
Old physical/visual success labels do NOT grant new-controller qualification.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .learning import write_json

PHASES={'inspect_floor','approach','lower','close','lift','place','release_retreat'}


def kinematic_signature(metadata):
    model=metadata.get('model_provenance',metadata)
    required=('geometry_source','arm_joints','handeye')
    if any(not model.get(key) for key in required):return None
    return hashlib.sha256(json.dumps({k:model[k] for k in required},sort_keys=True).encode()).hexdigest()


def valid_angles(value):
    try:angles=np.asarray(value,dtype=float)
    except (TypeError,ValueError):return None
    if angles.shape!=(9,) or not np.isfinite(angles).all():return None
    if np.any(angles[:5]<0) or np.any(angles[:5]>[180,180,180,180,270]):return None
    if not 30<=angles[5]<=170:return None
    return angles[:5]


def build_memory(root,destination,*,task_id='sock_basket',cancelled=lambda:False,progress=lambda v:None):
    from .explorer_model import model_metadata
    root=Path(root).resolve();destination=Path(destination);signature=kinematic_signature(model_metadata())
    if signature is None:raise ValueError('Current robot kinematics lack provenance')
    excluded=Counter();sources=[];failures=[];poses=defaultdict(list);seen=set();scanned=0
    protected=set(range(1300160000,1300160010))|set(range(1400160000,1400160010))|set(range(1500160000,1500160010))
    for state_path in (root/'skill-training').glob('*/state.json'):
        try:state=json.loads(state_path.read_text())
        except (OSError,ValueError):continue
        for group in state.get('validation_seeds',[]):protected.update(group)
        for entry in state.get('validation_history',[]):
            for group in entry.get('seeds',[]):protected.update(group)
    # Stable ordering makes the bounded diverse posture memory reproducible.
    for path in sorted((root/'demonstrations').glob('*/episode.json')):
        if cancelled():raise InterruptedError('Historical import cancelled; originals retained')
        scanned+=1
        if not path.resolve().is_relative_to(root):excluded['external_metadata']+=1;continue
        try:meta_bytes=path.read_bytes();meta=json.loads(meta_bytes)
        except (OSError,ValueError):excluded['unreadable_metadata']+=1;continue
        if meta.get('source')!='simulation' or meta.get('task_id')!=task_id:
            excluded['different_task_or_source']+=1;continue
        if meta.get('seed') in protected:excluded['reserved_validation_scene']+=1;continue
        if meta.get('success') is not True or meta.get('attempt_completed') is not True or meta.get('outcome',{}).get('success') is not True:
            if (meta.get('success') is False and meta.get('attempt_completed') is True and
                kinematic_signature(meta)==signature and type(meta.get('outcome',{}).get('lifted')) is bool):
                outcome=meta['outcome']
                failures.append(dict(source=str(path),metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),
                    seed=meta.get('seed'),phase=outcome.get('phase'),lifted=outcome.get('lifted'),
                    cause_prior='missed_material' if outcome.get('lifted') is False else 'hold_lost_or_occluded',
                    certainty='hypothesis_from_old_controller',actor_input_forbidden=True))
            excluded['not_completed_physical_success']+=1;continue
        if meta.get('action_alignment') not in ('same_row_pre_action','pre_action_observation'):
            excluded['unknown_action_alignment']+=1;continue
        contract=meta.get('sample_contract',{})
        if (contract.get('arm_action')!='applied_servo_endpoint' or
            contract.get('action_order',[])[:5]!=['base','shoulder','elbow','wrist_pitch','wrist_roll'] or
            contract.get('units',[])[:5]!=['deg']*5):
            excluded['unknown_command_units_or_order']+=1;continue
        if kinematic_signature(meta)!=signature:excluded['different_kinematics']+=1;continue
        samples=path.parent/'samples.jsonl'
        if not samples.resolve().is_relative_to(root):excluded['external_source']+=1;continue
        before=samples.stat() if samples.exists() else None
        if before is None:excluded['missing_samples']+=1;continue
        hasher=hashlib.sha256();last={};invalid=False;row_count=0
        try:
            with samples.open('rb') as stream:
                for line in stream:
                    if cancelled():raise InterruptedError('Historical import cancelled')
                    hasher.update(line);row=json.loads(line);row_count+=1
                    if row.get('action_alignment',meta['action_alignment']) not in ('same_row_pre_action','pre_action_observation'):
                        invalid=True;break
                    phase=row.get('phase')
                    if phase not in PHASES:continue
                    command=row.get('applied_action',row.get('action'))
                    angles=valid_angles(command)
                    if angles is None:invalid=True;break
                    last[phase]=angles
        except (OSError,ValueError,TypeError):invalid=True
        after=samples.stat()
        if invalid or row_count!=meta.get('frames') or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            excluded['invalid_or_changed_recording']+=1;continue
        source_hash=hasher.hexdigest()
        if source_hash in seen:excluded['duplicate_recording']+=1;continue
        seen.add(source_hash)
        for phase,angles in last.items():
            if len(poses[phase])<8 and all(np.linalg.norm(angles-np.asarray(old))>=10. for old in poses[phase]):
                poses[phase].append(angles.tolist())
        sources.append(dict(folder=str(path.parent),metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),
            samples_sha256=source_hash,seed=meta.get('seed'),old_success_not_new_qualification=True))
        progress(dict(scanned=scanned,admitted=len(sources),excluded=dict(excluded)))
    seeds=[]
    for phase in ('lower','close','approach','lift','place'):
        for pose in poses[phase]:
            if all(np.linalg.norm(np.asarray(pose)-old)>=10. for old in seeds):seeds.append(pose)
    result=dict(schema=1,importer_version=2,task_id=task_id,kind='historical_joint_posture_prior',kinematic_sha256=signature,
        scanned=scanned,admitted=len(sources),excluded=dict(excluded),sources=sources,
        phase_postures=dict(poses),ik_seeds=seeds[:24],
        failure_summaries=failures,
        failure_summaries_diagnostic_only=True,
        executed_as_commands=False,truth_features_imported=False,qualification_credit=0)
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()
    write_json(destination,result);return result


def checked_memory(path):
    value=json.loads(Path(path).read_text());expected=value.pop('sha256')
    actual=hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if actual!=expected or value.get('importer_version')!=2 or value.get('truth_features_imported') is not False or value.get('executed_as_commands') is not False:
        raise ValueError('Historical memory checksum/contract mismatch')
    seeds=value.get('ik_seeds',[])
    if len(seeds)>24 or any(valid_angles([*s,30.,0.,0.,0.]) is None for s in seeds):raise ValueError('Invalid historical joint prior')
    value['sha256']=expected;return value


def actor_prior(value):
    """No historical scene IDs, outcome labels or diagnostics enter the actor."""
    return dict(sha256=value['sha256'],ik_seeds=value['ik_seeds'])
