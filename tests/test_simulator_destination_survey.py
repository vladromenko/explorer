"""Finite camera exploration never upgrades partial or cached geometry to proof."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.rgbd_observation import combine_container_views,ObservationRefused
from simulator.skill_port import SimulatorSkillPort


def candidate(center=(.44,.35,.1)):
    return dict(center=list(center),bounds_min=[.29,.19,.076],bounds_max=[.595,.514,.12],
        uncertainty_m=.023,confidence=.75,evidence={'shape':'raised_walls_with_visible_opening',
        'rim_sectors':4,'opening_pixels':3000},semantic_identity='unverified')


def view(name,position,box=None):
    return dict(candidate=box or candidate(),frame_id=name,simulation_time=1.,camera_position=position)


def frame(name,position,containers=None,hints=None):
    transform=np.eye(4);transform[:3,3]=position
    return dict(frame_id=name,simulation_time=1.,optical_to_world=transform.tolist(),
                containers=containers or [],container_hypotheses=hints or [])


def make_port(tmp_path,frames):
    env=SimpleNamespace(command=np.r_[np.ones(5)*90.,170.,0.,0.,0.],
        pending_command=np.zeros(9),data=SimpleNamespace(ctrl=np.zeros(4)),wheel_ids=[0,1,2,3],
        phase='',last_error=None,base_pose=lambda:[0.,0.,0.])
    port=SimulatorSkillPort(env,tmp_path/'execution')
    sequence=iter(frames);port.scene=lambda ctx:next(sequence)
    calls=[];port.look_at=lambda *args,**kwargs:calls.append((args,kwargs))
    return port,calls


def test_two_independent_complete_views_measure_consistency_without_changing_gate():
    views=[view('a',[.23,.13,.28]),view('b',[.15,.12,.345])]
    result=combine_container_views(views)
    assert result['uncertainty_m']==.012
    assert result['evidence']['camera_baseline_m']>.02
    assert result['evidence']['source_frames']==['a','b']
    assert result['evidence']['real_calibration_validated'] is False


@pytest.mark.parametrize('second',[
    view('same-pose',[.23,.13,.28]),
    view('almost-same-pose',[.231,.13,.28]),
    view('displaced',[.15,.12,.345],candidate((.6,.35,.1))),
    view('partial',[.15,.12,.345],dict(candidate(),confidence=.3,
        evidence={'shape':'partial_raised_geometry_for_camera_aim_only'})),
])
def test_same_pose_disagreement_or_partial_wall_cannot_become_destination(second):
    with pytest.raises(ObservationRefused):combine_container_views([view('a',[.23,.13,.28]),second])


def test_bounded_survey_uses_fresh_complete_views_and_preserves_gripper(tmp_path):
    first=frame('first',[.23,.13,.28],[candidate()])
    second=frame('second',[.15,.12,.345],[candidate()])
    port,calls=make_port(tmp_path,[first,second])
    try:
        result=port.survey_destination(None)
        assert result['outcome']=='success'
        assert result['destination']['frame_id']=='second'
        assert len(calls)==2
        assert all(kwargs['distance_m']>=.4 for args,kwargs in calls)
        assert port.env.command[5]==170.
    finally:port.close()


def test_partial_hypothesis_only_guides_next_camera_view(tmp_path):
    hint=dict(candidate(),confidence=.3,evidence={'actionable':False})
    port,calls=make_port(tmp_path,[frame('partial',[.23,.13,.28],hints=[hint]),
        frame('empty',[.15,.12,.345])])
    try:
        result=port.survey_destination(None,max_views=2)
        assert result['outcome']=='unknown'
        assert port.destination is None
        assert len(calls)==2
        np.testing.assert_allclose(calls[1][0][0],hint['center'])
    finally:port.close()


def test_old_cache_cannot_replace_new_complete_measurement(tmp_path):
    port,calls=make_port(tmp_path,[frame('missing',[.15,.12,.345])])
    port.destination=candidate()
    port.destination_views=[view('old-a',[.23,.13,.28]),view('old-b',[.15,.12,.345])]
    try:
        result=port.survey_destination(None,max_views=1)
        assert result['outcome']=='unknown'
        assert result['evidence']['new_complete_views']==0
    finally:port.close()


def test_look_at_uses_current_closed_gripper_in_fk_and_final_command(tmp_path,monkeypatch):
    from simulator.environment import ExplorerEnv
    from scipy.optimize import OptimizeResult
    env=ExplorerEnv(render=False)
    port=SimulatorSkillPort(env,tmp_path/'execution')
    env.command[5]=170.
    goals=[];port.move_angles=lambda goal,ctx:goals.append(goal)
    monkeypatch.setattr('scipy.optimize.least_squares',lambda fun,start,**kwargs:
        OptimizeResult(x=np.asarray(start),fun=np.zeros(8)))
    try:
        port.look_at([.4,.24,.15],None)
        assert goals[-1][5]==170.
        port.look_at([.4,.24,.15],None,gripper=30.)
        assert goals[-1][5]==30.
    finally:port.close();env.close()
