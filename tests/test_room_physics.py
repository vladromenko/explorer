"""Finite geometry/sensor checks; these do not qualify a cleanup policy."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from simulator.environment import ExplorerEnv
from simulator.explorer_model import build_mjcf
from simulator.inspection_planning import check_observed_box_path,CameraViewUnavailable
from simulator.room_cleanup_port import RoomCleanupPort
from simulator.room_validation import room_scene,CollisionAudit


def test_room_adds_real_collision_walls_only_when_explicitly_requested():
    ordinary=ET.fromstring(build_mjcf());room=ET.fromstring(build_mjcf(room_layout=True))
    base={g.attrib['name'] for g in ordinary.findall('.//geom') if 'name' in g.attrib}
    added=[g for g in room.findall('.//geom') if g.attrib.get('name','').startswith('cleanup_room_')]
    assert len(added)==6 and all(g.attrib['contype']=='2' and g.attrib['conaffinity']=='5' for g in added)
    assert not any(name.startswith('cleanup_room_') for name in base)


def test_local_base_sweep_checks_intermediate_robot_volume_without_mutating_state():
    env=ExplorerEnv(render=False);before=env.data.qpos.copy()
    try:
        obstacle=dict(bounds_min=[.44,-.1,0.],bounds_max=[.55,.1,.3],uncertainty_m=.005)
        # Start is clear, but a 30 cm translation would enter the basket box.
        check_observed_box_path(env,env.command[:6],obstacle)
        with pytest.raises(CameraViewUnavailable,match='пересекает'):
            check_observed_box_path(env,env.command[:6],obstacle,base_target=[.30,0.])
        np.testing.assert_array_equal(before,env.data.qpos)
        with pytest.raises(CameraViewUnavailable,match='finite'):
            check_observed_box_path(env,env.command[:6],obstacle,base_target=[1.,0.])
    finally:env.close()


def test_own_footprint_and_room_scan_do_not_read_hidden_object_outcomes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import simulator.room_cleanup_port as module
    env=ExplorerEnv(render=False,config={'room_layout':True});env.reset(seed=1600000004,options=dict(
        task_id='sock_basket',scene_profile='sock_variants_v1',generated_scene=room_scene(1600000004)))
    monkeypatch.setattr(module,'RoomActor',lambda *a,**k:SimpleNamespace(sha='unit'))
    monkeypatch.setattr(env,'state',lambda:pytest.fail('Actor cannot inspect hidden task state'))
    port=RoomCleanupPort(env,tmp_path/'room',tmp_path/'unused',{},tmp_path/'unused-vision')
    try:
        assert port.own_radius()>=.35
        port._scan();assert port.grid.free.any() and port.grid.occupied.any()
        assert port.grid.radius>=.35
        outside=port.grid.cell([3.,3.]);assert not port.grid.free[outside]
        frontiers=[g for g in port.survey_candidates() if g.get('information_source')=='observed_scan_frontier']
        assert frontiers and any(port.plan(g)['reachable'] for g in frontiers)
        assert all(g['information_gain']>0 for g in frontiers)
    finally:port.close();env.close()


def test_heading_feedback_overcomes_rotational_resistance_without_relaxing_tolerance():
    from simulator.room_navigation import HeadingFeedback
    controller=HeadingFeedback();yaw=0.;goal=.5
    for _ in range(1000):
        error=goal-yaw
        if abs(error)<.03:break
        velocity=controller.update(error,.05)
        assert abs(velocity)<=.15
        actual=np.sign(velocity)*max(0.,abs(velocity)-.047)*.8
        yaw+=actual*.05
    assert abs(goal-yaw)<.03


def test_room_navigation_actually_arrives_in_physics(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import simulator.room_cleanup_port as module
    env=ExplorerEnv(scene='empty',render=False,config={'room_layout':True})
    monkeypatch.setattr(module,'RoomActor',lambda *a,**k:SimpleNamespace(sha='unit'))
    port=RoomCleanupPort(env,tmp_path/'drive',tmp_path/'unused',{},tmp_path/'unused-vision')
    try:
        port.begin(lambda:None)
        result=port.navigate(dict(x=-.25,y=.10,yaw=.50),check=lambda:None)
        assert result['completed'] and np.linalg.norm(np.asarray(env.base_pose())[:2]-[-.25,.10])<.025
        assert abs(env.base_pose()[2]-.50)<.04
    finally:port.close();env.close()


def test_near_target_return_is_exempt_only_inside_the_prechecked_observed_box():
    from types import SimpleNamespace
    port=RoomCleanupPort.__new__(RoomCleanupPort)
    site=SimpleNamespace(xpos=np.array([0.,0.,.1]),xmat=np.eye(3).ravel())
    env=SimpleNamespace(data=SimpleNamespace(site=lambda _:site),lidar_scan=lambda *a,**k:
        dict(ranges=[.05],angle_min=0.,angle_increment=0.))
    port.env=env
    with pytest.raises(ValueError):port.lidar_clearance()
    target=dict(bounds_min=[.04,-.01,0.],bounds_max=[.06,.01,.3],uncertainty_m=.005)
    port.lidar_clearance(target)
    target['bounds_min'][0]=.2;target['bounds_max'][0]=.3
    with pytest.raises(ValueError):port.lidar_clearance(target)


def test_outcome_audit_rejects_arm_floor_collision_but_allows_chassis_support():
    from types import SimpleNamespace
    env=ExplorerEnv(render=False)
    try:
        audit=CollisionAudit(env);finger=env.model.geom('finger_left').id
        assert finger in audit.arm
        base=next(iter(audit.robot-audit.arm))
        audit.env=SimpleNamespace(data=SimpleNamespace(time=1.,contact=[
            SimpleNamespace(geom=[audit.floor,base],dist=-.002),
            SimpleNamespace(geom=[audit.floor,finger],dist=-.002)]))
        audit.sample();assert len(audit.contacts)==1 and audit.contacts[0]['geoms']==[audit.floor,finger]
    finally:env.close()


def test_unconfirmed_basket_hint_does_not_become_a_navigation_obstacle():
    from types import SimpleNamespace
    from src.room_cleanup import ViewUnavailable
    port=RoomCleanupPort.__new__(RoomCleanupPort);port._stop_base=lambda:None
    port._ctx=lambda c:c
    port.geometry=SimpleNamespace(destination=None,survey_destination=lambda *a,**k:dict(outcome='unknown',reason='partial opening'))
    with pytest.raises(ViewUnavailable):
        port.look_at(dict(kind='basket',usable_for='camera_aim_only',bounds_max=[1.,1.,.3]),check=lambda:None)
    assert port.geometry.destination is None
    assert port.geometry.destination_camera_min_height==pytest.approx(.42)


def test_user_basket_alignment_does_not_accept_a_chassis_intersection():
    env=ExplorerEnv(render=False)
    port=RoomCleanupPort.__new__(RoomCleanupPort);port.env=env;port.check=lambda:None
    try:
        for distance,expected in ((.259,False),(.325,False),(.35,True)):
            target=dict(center=[distance,0.,.15],bounds_min=[distance-.125,-.125,0.],
                bounds_max=[distance+.125,.125,.30],uncertainty_m=.012)
            assert port.local_admission('basket',target)['inside_envelope'] is expected
    finally:env.close()
