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
