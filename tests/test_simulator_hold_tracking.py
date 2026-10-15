from types import SimpleNamespace

import numpy as np
import pytest

from simulator.skill_port import SimulatorSkillPort


def candidate(z=.16,color=(.18,.09,.18)):
    return dict(center=[.21,.02,z],bounds_min=[.20,.01,z-.03],bounds_max=[.23,.04,z+.01],
        uncertainty_m=.012,confidence=.45,appearance={'rgb_median':list(color)},
        evidence={'self_occlusion_not_resolved':False,'visible_pixels':500})


def setup(tmp_path,second,monkeypatch,delta=(0.,0.,.02)):
    from simulator.manipulation_planning import HoldProbePlan
    jaw=np.array([.2,0.,.18])
    env=SimpleNamespace(phase='',pending_command=np.zeros(9),command=np.zeros(9),
        data=SimpleNamespace(ctrl=np.zeros(4),body=lambda name:SimpleNamespace(xmat=np.eye(3).ravel())),
        wheel_ids=[0,1,2,3],jaw_center=lambda:jaw.copy())
    port=SimulatorSkillPort(env,tmp_path/'execution')
    port.target={'appearance':{'rgb_median':[.6,.3,.6]}}
    frames=iter([dict(frame_id='first',object_candidates=[candidate()]),
                 dict(frame_id='second',object_candidates=second)])
    port.scene=lambda ctx:next(frames)
    monkeypatch.setattr('simulator.manipulation_planning.plan_hold_probe',lambda *args,**kwargs:
        HoldProbePlan(np.r_[np.ones(5)*90.,170.],jaw+delta,np.eye(3),
            'up' if delta[2] else 'toward_base'))
    def move_angles(*args,**kwargs):jaw[:]+=delta
    port.move_angles=move_angles
    return port


def test_partial_shadowed_material_requires_and_passes_measured_motion(tmp_path,monkeypatch):
    port=setup(tmp_path,[candidate(z=.18)],monkeypatch)
    try:
        result=port.verify_hold(None)
        assert result['outcome']=='success'
        assert result['evidence']['confirmed'] is True
        assert result['evidence']['frames']==['first','second']
    finally:port.close()


@pytest.mark.parametrize('second',[[candidate()],[],[candidate(z=.18),candidate(z=.18)],
    [candidate(z=.18,color=(.05,.1,.8))]])
def test_static_missing_ambiguous_or_different_material_does_not_pass(tmp_path,second,monkeypatch):
    port=setup(tmp_path,second,monkeypatch)
    try:assert port.verify_hold(None)['outcome']!='success'
    finally:port.close()


def test_horizontal_probe_still_requires_same_measured_object_displacement(tmp_path,monkeypatch):
    second=candidate();second['center'][0]-=.02
    port=setup(tmp_path,[second],monkeypatch,delta=(-.02,0.,0.))
    try:
        result=port.verify_hold(None)
        assert result['outcome']=='success'
        assert result['evidence']['probe_direction']=='toward_base'
        assert result['evidence']['measured_translation_m']==pytest.approx(.02)
    finally:port.close()


def test_matching_object_motion_cannot_hide_a_turning_hand(tmp_path,monkeypatch):
    from scipy.spatial.transform import Rotation
    port=setup(tmp_path,[candidate(z=.18)],monkeypatch)
    original=port.move_angles
    def turning(*args,**kwargs):
        original(*args,**kwargs)
        port.env.data.body=lambda name:SimpleNamespace(xmat=Rotation.from_euler('y',10.,degrees=True).as_matrix().ravel())
    port.move_angles=turning
    try:assert port.verify_hold(None)['outcome']=='unknown'
    finally:port.close()
