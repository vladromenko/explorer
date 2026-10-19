import importlib.util
import json
from pathlib import Path


def module():
    spec=importlib.util.spec_from_file_location('audit_run_tests',Path(__file__).resolve().parents[1]/'bin/audit-learning-run.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_provenance_samples_already_include_replay(tmp_path):
    audit=module();path=tmp_path/'provenance.json'
    row=dict(input_sha256={'samples.jsonl':'a'*64},samples=12,recorded_frames=3,replay_weight=4,
        episode=dict(id='x',seed=12,source='simulation',success=True,outcome=dict(success=True),
            room_scene=True,full_mission_success=True))
    path.write_text(json.dumps(dict(episodes=[row])))
    value,hashes,seeds=audit.partition(path)
    assert value['samples']==3 and value['effective_replay_frames']==12
    assert value['recorded_simulator_success_labels']==1 and value['whole_room_outcome_labels']==1
    assert hashes=={'a'*64} and seeds=={12}


def test_confidence_does_not_turn_five_of_ten_into_reliability_claim():
    audit=module();low,high=audit.wilson(5,10)
    assert .23<low<.24 and .76<high<.77
    assert audit.wilson(100,100)[0]>.96
