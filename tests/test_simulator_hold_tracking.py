from types import SimpleNamespace

import numpy as np
import pytest

from simulator.skill_port import SimulatorSkillPort


def candidate(z=.16,color=(.18,.09,.18)):
    return dict(center=[.21,.02,z],bounds_min=[.20,.01,z-.03],bounds_max=[.23,.04,z+.01],
        uncertainty_m=.012,confidence=.45,appearance={'rgb_median':list(color)},
        evidence={'self_occlusion_not_resolved':False,'visible_pixels':500})


def setup(tmp_path,second):
    jaw=np.array([.2,0.,.18])
    env=SimpleNamespace(phase='',pending_command=np.zeros(9),command=np.zeros(9),
        data=SimpleNamespace(ctrl=np.zeros(4)),wheel_ids=[0,1,2,3],jaw_center=lambda:jaw.copy())
    port=SimulatorSkillPort(env,tmp_path/'execution')
    port.target={'appearance':{'rgb_median':[.6,.3,.6]}}
    frames=iter([dict(frame_id='first',object_candidates=[candidate()]),
                 dict(frame_id='second',object_candidates=second)])
    port.scene=lambda ctx:next(frames)
    def move(xyz,*args,**kwargs):jaw[:]=xyz
    port.move=move
    return port


def test_partial_shadowed_material_requires_and_passes_measured_motion(tmp_path):
    port=setup(tmp_path,[candidate(z=.18)])
    try:
        result=port.verify_hold(None)
        assert result['outcome']=='success'
        assert result['evidence']['confirmed'] is True
        assert result['evidence']['frames']==['first','second']
    finally:port.close()


@pytest.mark.parametrize('second',[[candidate()],[],[candidate(z=.18),candidate(z=.18)],
    [candidate(z=.18,color=(.05,.1,.8))]])
def test_static_missing_ambiguous_or_different_material_does_not_pass(tmp_path,second):
    port=setup(tmp_path,second)
    try:assert port.verify_hold(None)['outcome']!='success'
    finally:port.close()
