import importlib.util
import json
from pathlib import Path

import pytest

from simulator.automatic_learning import AutomaticLearning


def maintenance_module():
    p=Path(__file__).resolve().parents[1]/'bin/apply-simulator-update.py'
    spec=importlib.util.spec_from_file_location('simulator_update_cli',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_legacy_maintenance_handoff_preserves_record_and_never_overrides_changed_pause(tmp_path):
    module=maintenance_module();folder=tmp_path/'automatic-requests';folder.mkdir();(tmp_path/'maintenance').mkdir()
    identifier='ab'*6;receipt=dict(owner_id=identifier,paused=True,at=123)
    row=dict(id=identifier,state='operator_paused',events=[],tasks=[dict(state='operator_paused')],completed=167)
    (folder/(identifier+'.json')).write_text(json.dumps(row))
    (folder/'queue-pause.json').write_text(json.dumps(dict(receipt,at=124)))
    with pytest.raises(ValueError,match='пользователя'):module.finish_legacy_pause(tmp_path,identifier,receipt,'checked update')
    assert json.loads((folder/(identifier+'.json')).read_text())==row
    (folder/'queue-pause.json').write_text(json.dumps(receipt))
    module.finish_legacy_pause(tmp_path,identifier,receipt,'checked update')
    updated=json.loads((folder/(identifier+'.json')).read_text())
    assert updated['state']=='maintenance_paused' and updated['completed']==167
    assert (folder/'queue-pause.json').exists()  # caller must separately verify and clear its own receipt
    assert json.loads((tmp_path/'maintenance'/('authorized-update-'+identifier+'.json')).read_text())['previous_request']==row


def test_new_update_request_does_not_cancel_workers_or_create_new_jobs(tmp_path):
    book=AutomaticLearning(tmp_path);folder=tmp_path/'skill-training'/('bc'*6);folder.mkdir(parents=True)
    row=dict(id='ab'*6,state='running',events=[],tasks=[dict(training_id='bc'*6)],created=1)
    book.save(row);result=book.prepare_update(row['id'])
    assert result['state']=='running' and (folder/'update.request').exists()
    assert not book.pause_info() and not (folder/'pause.request').exists()
    assert len(book.list())==1
    book.pause(row['id'])
    with pytest.raises(ValueError,match='Ручная пауза'):book.prepare_update(row['id'])
