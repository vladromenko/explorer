import json
from simulator.training_activity import activity,measured_rate


def fixture(tmp_path,stage='training',step=0,text=''):
    folder=tmp_path/'runs'/'aaaaaaaaaaaa'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'job.json').write_text(json.dumps(dict(state=stage,steps_completed=step,steps=1000,at=999)))
    (folder/'train.log').write_text(text)
    return folder


def test_fresh_owner_heartbeat_is_not_weight_progress(tmp_path):
    folder=fixture(tmp_path,text='Creating dataset')
    value=activity(tmp_path,folder.name)
    assert value['stage']=='loading_data' and not value['counter_change_observed']
    fixture(tmp_path,text='Start offline training')
    assert activity(tmp_path,folder.name)['stage']=='waiting_for_first_batch'
    fixture(tmp_path,step=10,text='INFO step:10 ')
    value=activity(tmp_path,folder.name)
    assert value['stage']=='updating_weights' and value['counter_change_observed']
    assert not activity(tmp_path,folder.name)['counter_change_observed']


def test_rate_uses_recent_successful_updates_not_startup_or_heartbeat():
    text='INFO 2026-10-19 12:00:00 step:10 \nINFO 2026-10-19 12:00:10 step:20 \nINFO 2026-10-19 12:00:20 step:30 '
    assert measured_rate(text)==1.
    assert measured_rate('Creating dataset') is None


def test_stale_log_and_unfinished_latest_checkpoint_are_explicit(tmp_path):
    folder=fixture(tmp_path,step=20,text='INFO step:20 ')
    (folder/'model/checkpoints/001000').mkdir(parents=True)
    (folder/'model/checkpoints/002000').mkdir()
    now=(folder/'train.log').stat().st_mtime+121
    value=activity(tmp_path,folder.name,now=now)
    assert value['stage']=='progress_not_confirmed'
    assert value['stable_checkpoint'].endswith('001000')
    assert not value['simulation_qualified'] and not value['robot_verified']


def test_rate_retains_exact_updates_when_info_counter_is_abbreviated():
    text='INFO 2026-10-19 12:00:00 step:3K \rTraining | 3010/145799 [11:10<30:00, 2step/s]\rTraining | 3030/145799 [11:20<30:00, 2step/s]\rTraining | 3050/145799 [11:30<30:00, 2step/s]'
    assert measured_rate(text)==2.


def test_early_room_result_cannot_claim_final_qualification(tmp_path):
    from simulator.training_activity import room_evidence
    folder=tmp_path/'diagnostics/live-room-example';folder.mkdir(parents=True)
    (tmp_path/'diagnostics/early-room-current.json').write_text(json.dumps(dict(folder=str(folder))))
    (folder/'state.json').write_text(json.dumps(dict(state='complete',checkpoint_step=29000,mission_success=True,seed=1703000014)))
    value=room_evidence(tmp_path)
    assert value['success'] and not value['simulation_qualified'] and not value['robot_verified']


def test_local_model_selection_remains_separate_from_full_mission(tmp_path):
    from simulator.training_activity import local_selection_evidence
    folder=tmp_path/'neural-recovery'/'aaaaaaaaaaaa'/'round-01';folder.mkdir(parents=True)
    (folder/'selection.json').write_text(json.dumps(dict(selected='/candidates/00',results=[
        dict(model_id='00',successes=5,episodes=10,model_weights_sha256='a'*64)])))
    value=local_selection_evidence(tmp_path,'aaaaaaaaaaaa')
    assert value['successes']==5 and value['scope']=='previous_local_checkpoint_selection'
    assert not value['simulation_qualified']


def test_resume_never_calls_restored_counter_fresh_updates(tmp_path):
    import json
    from simulator.training_activity import activity
    folder=tmp_path/'runs/resumed';folder.mkdir(parents=True)
    old=b'Training: 6000/145799 [01:00:00<20:00:00]\n'
    (folder/'train.log').write_bytes(old+b'INFO Creating dataset\n')
    (folder/'job.json').write_text(json.dumps(dict(state='training',resumed_step=6000,
        steps_completed=6000,steps=145799,training_log_start_bytes=len(old),learner_started_at=1)))
    state=activity(tmp_path,'resumed')
    assert state['stage']=='loading_data' and state['steps_completed']==6000
    with (folder/'train.log').open('a') as stream:stream.write('Training: 20/139799 [00:10<20:00:00]\n')
    state=activity(tmp_path,'resumed')
    assert state['stage']=='updating_weights' and state['steps_completed']==6020
