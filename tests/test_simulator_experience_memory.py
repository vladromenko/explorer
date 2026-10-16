import hashlib
import json

import pytest

from simulator.experience_memory import build_memory,checked_memory


MODEL=dict(geometry_source='known CAD',arm_joints=['own calibrated joints'],handeye={'calibration':'known'})


def record(root,name,**changes):
    folder=root/'demonstrations'/name;folder.mkdir(parents=True)
    meta=dict(source='simulation',task_id='sock_basket',success=True,attempt_completed=True,
        outcome={'success':True},seed=101,frames=2,model_provenance=MODEL,action_alignment='same_row_pre_action')
    meta['sample_contract']=dict(arm_action='applied_servo_endpoint',
        action_order=['base','shoulder','elbow','wrist_pitch','wrist_roll'],units=['deg']*5)
    meta.update(changes)
    (folder/'episode.json').write_text(json.dumps(meta))
    rows=[dict(phase='lower',action=[90.,40.,120.,90.,90.,30.,0.,0.,0.],
        simulator_diagnostics={'sock_center_m':['NEVER_IMPORT_SECRET']},
        environment_state=['NEVER_IMPORT_SECRET']) for _ in range(2)]
    (folder/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return folder


def test_historical_reuse_is_audited_kinematic_prior_not_commands_truth_or_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr('simulator.explorer_model.model_metadata',lambda:MODEL)
    source=record(tmp_path,'old');before=hashlib.sha256((source/'samples.jsonl').read_bytes()).hexdigest()
    output=tmp_path/'memory.json';v=build_memory(tmp_path,output)
    assert v['admitted']==1 and v['ik_seeds']==[[90.,40.,120.,90.,90.]]
    assert v['qualification_credit']==0 and not v['executed_as_commands'] and not v['truth_features_imported']
    assert 'NEVER_IMPORT_SECRET' not in output.read_text()
    assert hashlib.sha256((source/'samples.jsonl').read_bytes()).hexdigest()==before
    assert checked_memory(output)['sha256']==v['sha256']
    v['ik_seeds'][0][0]=45.;output.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='checksum'):checked_memory(output)


def test_import_excludes_incompatible_failed_truncated_duplicate_and_reserved_episodes(tmp_path,monkeypatch):
    monkeypatch.setattr('simulator.explorer_model.model_metadata',lambda:MODEL)
    record(tmp_path,'good');record(tmp_path,'duplicate')
    record(tmp_path,'failed',success=False)
    record(tmp_path,'unknown_alignment',action_alignment='unknown')
    record(tmp_path,'truncated',frames=3)
    record(tmp_path,'different',model_provenance=dict(MODEL,arm_joints=['other geometry']))
    record(tmp_path,'reserved',seed=1300160000)
    v=build_memory(tmp_path,tmp_path/'memory.json')
    assert v['admitted']==1
    assert v['excluded']==dict(duplicate_recording=1,not_completed_physical_success=1,unknown_action_alignment=1,
        invalid_or_changed_recording=1,different_kinematics=1,reserved_validation_scene=1)


def test_actor_projection_cannot_receive_historical_scene_ids_or_outcomes(tmp_path,monkeypatch):
    from simulator.experience_memory import actor_prior
    monkeypatch.setattr('simulator.explorer_model.model_metadata',lambda:MODEL)
    record(tmp_path,'old');v=build_memory(tmp_path,tmp_path/'memory.json')
    prior=actor_prior(v)
    assert set(prior)=={'sha256','ik_seeds'}
    assert 'sources' not in prior and 'failure_summaries' not in prior
