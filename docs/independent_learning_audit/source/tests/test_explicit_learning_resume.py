import importlib.util
from pathlib import Path

import pytest


def validator():
    path=Path(__file__).resolve().parents[1]/'bin/resume-paused-learning.py'
    spec=importlib.util.spec_from_file_location('explicit_resume',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_states


def test_resume_accepts_only_both_stopped_operator_owners():
    validate=validator()
    local=dict(state='paused',active_pids=[],alive=False)
    room=dict(state='operator_paused',active_pids=[],alive=False)
    validate(local,room)
    for changed in (dict(state='running'),dict(active_pids=[123]),dict(alive=True)):
        with pytest.raises(ValueError):validate(dict(local,**changed),room)
        with pytest.raises(ValueError):validate(local,dict(room,**changed))
