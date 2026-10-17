import copy
import json

import pytest

from simulator.repair_upgrade import decision,source_verified


def snapshot(state):
    return dict(queue_pause=None,requests=[dict(id='a'*12,state=state)],trainings=[])


@pytest.mark.parametrize('state',['running','training','validating','queued','pausing','waiting_for_memory'])
def test_upgrade_never_interrupts_current_or_queued_work(state):
    assert decision(snapshot(state),'a'*12)=='active'


def test_upgrade_respects_operator_pauses_orphans_and_success():
    s=snapshot('budget_paused');assert decision(s,'a'*12)=='ready'
    s['trainings']=[{'active_workers':[{'episode_id':'still-closing'}]}]
    assert decision(s,'a'*12)=='active'
    s=snapshot('operator_paused');assert decision(s,'a'*12)=='operator_pause'
    s=snapshot('budget_paused');s['queue_pause']={'paused':True}
    assert decision(s,'a'*12)=='operator_pause'
    assert decision(snapshot('simulation_qualified'),'a'*12)=='qualified'
    assert decision(snapshot('maintenance_paused'),'a'*12)=='ready'


def test_resource_pause_can_recover_only_with_memory_headroom_and_no_operator_pause():
    s=snapshot('memory_paused');s['memory']=dict(total_bytes=100,available_bytes=24)
    assert decision(s,'a'*12)=='waiting_for_memory'
    s['memory']['available_bytes']=25
    assert decision(s,'a'*12)=='ready'
    s['queue_pause']={'paused':True}
    assert decision(s,'a'*12)=='operator_pause'


def test_upgrade_rejects_staged_tampering_and_external_paths(tmp_path):
    import hashlib
    (tmp_path/'module.py').write_text('checked code')
    config=dict(source=str(tmp_path),files={'module.py':hashlib.sha256(b'checked code').hexdigest()})
    assert source_verified(config)==tmp_path
    (tmp_path/'module.py').write_text('unverified code')
    with pytest.raises(ValueError,match='checksum'):source_verified(config)
    config['files']={'../../etc/passwd':'bad'}
    with pytest.raises(ValueError):source_verified(config)


@pytest.mark.parametrize('state',['running','training','validating','queued','operator_paused'])
def test_activation_does_not_touch_the_service_or_lease_while_work_or_operator_pause_exists(tmp_path,monkeypatch,state):
    import simulator.repair_upgrade as upgrade
    source=tmp_path/'checked-source';source.mkdir();root=tmp_path/'data';root.mkdir()
    config=dict(source=str(source),files={},root=str(root),support=str(tmp_path),request='a'*12)
    monkeypatch.setattr(type(tmp_path),'is_mount',lambda p:True)
    monkeypatch.setattr(upgrade,'get',lambda path:snapshot(state))
    monkeypatch.setattr(upgrade.psutil,'Process',lambda *args:pytest.fail('touched an active service'))
    monkeypatch.setattr(upgrade.subprocess,'Popen',lambda *a,**kw:pytest.fail('started a second service'))
    assert upgrade.activate(config,tmp_path/'marker.json') is None
    assert not (root/'skill-training').exists()
