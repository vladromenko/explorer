"""Early diagnostics cannot overwrite a live learner or count as acceptance."""
import json
import shutil

import pytest

from simulator import live_room_diagnostic as diagnostic
from simulator.learning import write_json


def checkpoint(root,name):
    p=root/'model/checkpoints'/name/'pretrained_model';p.mkdir(parents=True)
    for n in ('model.safetensors','config.json','policy_preprocessor.json','policy_postprocessor.json'):
        (p/n).write_bytes(b'fixture only')
    return p


def test_snapshot_ignores_newest_checkpoint_and_last_link(tmp_path,monkeypatch):
    training=tmp_path/'training';old=checkpoint(training,'056000');new=checkpoint(training,'057000')
    (new.parent.parent/'last').symlink_to(new.parent)
    assert diagnostic.numeric_checkpoint(training)==old
    def stage(source,relative,target):shutil.copytree(source/relative,target/'copied')
    monkeypatch.setattr('simulator.checkpoint_experiment.stage_candidate',stage)
    monkeypatch.setattr(diagnostic,'read_bundle',lambda target:({},target/'copied'))
    before=diagnostic.manifest(training)
    target=tmp_path/'copy';assert diagnostic.snapshot(training,target)==56000
    assert diagnostic.manifest(training)==before
    receipt=json.loads((target/'early-diagnostic.json').read_text())
    assert receipt['simulation_qualified'] is False and receipt['source_training_untouched'] is True
    second=tmp_path/'second-copy'
    assert diagnostic.snapshot(training,second,previous=target)==56000
    assert diagnostic.manifest(second)==diagnostic.manifest(target)


def test_changed_source_is_rejected_and_all_evidence_is_retained(tmp_path,monkeypatch):
    training=tmp_path/'training';old=checkpoint(training,'056000');checkpoint(training,'057000')
    def changed(source,relative,target):
        shutil.copytree(source/relative,target/'copied');(old/'model.safetensors').write_bytes(b'changed')
    monkeypatch.setattr('simulator.checkpoint_experiment.stage_candidate',changed)
    monkeypatch.setattr(diagnostic,'read_bundle',lambda target:({},target/'copied'))
    with pytest.raises(ValueError,match='changed during snapshot'):diagnostic.snapshot(training,tmp_path/'copy')
    assert (tmp_path/'copy/copied/model.safetensors').exists()


def test_single_checkpoint_may_still_be_writing(tmp_path):
    checkpoint(tmp_path,'057000')
    with pytest.raises(ValueError,match='two numeric'):diagnostic.numeric_checkpoint(tmp_path)


def test_low_memory_never_admits_a_side_diagnostic():
    assert not diagnostic.can_start(dict(free_percent=24.9,available_gb=6.))
    assert not diagnostic.can_start(dict(free_percent=30.,available_gb=5.))
    assert diagnostic.can_start(dict(free_percent=26.,available_gb=6.))
    assert not diagnostic.can_start(dict(free_percent=26.,available_gb=6.),after_pressure=True)


def test_early_scene_is_reserved_from_future_final_tests(tmp_path,monkeypatch):
    import simulator.neural_recovery as recovery
    folder=tmp_path/'diagnostics/live-room-example'
    write_json(folder/'request.json',{'seed':1703000004})
    assert 1703000004 in recovery.reserved_scene_seeds(tmp_path)
    monkeypatch.setattr(recovery,'training_seeds',lambda _:set())
    seed=diagnostic.choose_seed(tmp_path,tmp_path/'model')
    assert seed==1703000014  # same user's geometry, different independent scene


def test_repaired_retry_cannot_relaunch_same_code_pause_or_accepted_scene():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'bin/diagnose-live-room.py'
    spec=importlib.util.spec_from_file_location('live_room_launcher_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=dict(revision='previous');state=dict(state='failed',training='/runs/aaaaaaaaaaaa')
    module.validate_repair_retry(old,state,'checked-new-revision','aaaaaaaaaaaa')
    for bad in (dict(state,state='running'),dict(state,state='paused'),dict(state,state='complete',mission_success=True)):
        with pytest.raises(ValueError):module.validate_repair_retry(old,bad,'checked-new-revision','aaaaaaaaaaaa')
    with pytest.raises(ValueError):module.validate_repair_retry(old,state,'previous','aaaaaaaaaaaa')
