import numpy as np
import pytest
from types import SimpleNamespace
from simulator.vision_backend import equivalent,PinnedDetectorBackend


def detection(score=.8,box=None):return dict(scores=np.array([score]),boxes=np.array([[1,2,3,4] if box is None else box]),text_labels=['sock'])


def test_same_labels_are_not_sufficient_for_accelerator_conformance():
    assert equivalent(detection(),detection(.8002,[1.0002,2,3,4]))
    assert not equivalent(detection(),detection(.81))
    assert not equivalent(detection(),detection(.8,[2,2,3,4]))
    assert not equivalent(detection(),dict(scores=[],boxes=[],text_labels=[]))


def test_cpu_only_host_does_not_warm_up_an_unavailable_accelerator(monkeypatch):
    torch=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False)))
    backend=PinnedDetectorBackend(torch,object(),object());calls=[]
    backend._infer=lambda inputs,size,device:(calls.append(device) or detection(),.1)
    backend.infer({},(12,16));backend.infer({},(12,16))
    assert calls==['cpu','cpu'] and backend.device=='cpu'


def test_runtime_accelerator_error_returns_to_same_fp32_cpu_model():
    moves=[];torch=SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda:None))
    model=SimpleNamespace(to=lambda device:moves.append(device));backend=PinnedDetectorBackend(torch,model,object())
    backend.checked=True;backend.device='mps'
    def infer(inputs,size,device):
        if device=='mps':raise RuntimeError('unsupported op')
        return detection(),.1
    backend._infer=infer
    assert equivalent(backend.infer({},(12,16)),detection())
    assert moves==['cpu'] and backend.device=='cpu' and 'fallback' in backend.measurement


def test_early_diagnostic_can_explicitly_keep_vision_off_training_gpu(monkeypatch):
    monkeypatch.setenv('EXPLORER_VISION_CPU_ONLY','1')
    torch=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(
        is_available=lambda:pytest.fail('Diagnostic must not probe or warm training GPU'))))
    backend=PinnedDetectorBackend(torch,object(),object());calls=[]
    backend._infer=lambda inputs,size,device:(calls.append(device) or detection(),.1)
    backend.infer({},(12,16));backend.infer({},(12,16))
    assert calls==['cpu','cpu']


def test_accelerator_diagnostics_are_json_serializable(monkeypatch):
    import json
    import simulator.vision_backend as module
    monkeypatch.setattr(module.sys,'platform','darwin')
    monkeypatch.setattr('simulator.resource_budget.memory_budget',lambda:dict(available_bytes=50,total_bytes=100))
    torch=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)),mps=SimpleNamespace(empty_cache=lambda:None))
    backend=PinnedDetectorBackend(torch,SimpleNamespace(to=lambda _:None),object())
    backend._infer=lambda inputs,size,device:(detection(),1. if device=='cpu' else .1)
    backend.infer({},(12,16))
    assert backend.device=='mps' and json.loads(json.dumps(backend.measurement))['same_recorded_detections'] is True
