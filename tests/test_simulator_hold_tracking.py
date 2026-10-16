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


@pytest.mark.parametrize('slip_m,expected',[(0.,'success'),(.03,'unknown')])
def test_tracked_hold_compares_in_rotating_hand_frame(tmp_path,monkeypatch,slip_m,expected):
    from scipy.spatial.transform import Rotation
    port=setup(tmp_path,[candidate(z=.18)],monkeypatch)
    try:
        assert port.verify_hold(None)['outcome']=='success'
        rotation=Rotation.from_euler('y',120.,degrees=True).as_matrix()
        jaw=port.env.jaw_center()
        offset=np.asarray(port.before_hold['center'])-jaw
        patch=candidate(z=.18)
        patch['center']=(jaw+rotation@(offset+[slip_m,0.,0.])).tolist()
        port.env.data.body=lambda name:SimpleNamespace(xmat=rotation.ravel())
        port.scene=lambda ctx:dict(frame_id='after_turn',object_candidates=[patch])
        result=port.verify_hold(None,probe=False)
        assert result['outcome']==expected
        if expected=='success':
            assert result['evidence']['comparison_frame']=='arm5'
            assert result['evidence']['relative_displacement_m']==pytest.approx(0.,abs=1e-10)
    finally:port.close()


def test_untracked_hold_without_orientation_reference_is_unknown(tmp_path,monkeypatch):
    port=setup(tmp_path,[candidate(z=.18)],monkeypatch)
    try:
        assert port.verify_hold(None)['outcome']=='success'
        del port.hold_rotation
        port.scene=lambda ctx:dict(frame_id='later',object_candidates=[candidate(z=.18)])
        assert port.verify_hold(None,probe=False)['outcome']=='unknown'
    finally:port.close()


@pytest.mark.parametrize('kind,expected', [('stable','success'),('lost','unknown'),('duplicate','unknown')])
def test_stationary_hold_requires_a_second_of_fresh_visible_evidence(tmp_path,monkeypatch,kind,expected):
    port=setup(tmp_path,[candidate(z=.18)],monkeypatch)
    ticks=[];checks=[]
    port.tick=lambda arm,ctx=None:ticks.append(arm.copy())
    def check(ctx,probe=False):
        checks.append(len(ticks))
        if kind=='lost' and len(checks)==2:return dict(outcome='unknown',reason='lost')
        return dict(outcome='success',evidence={'frame_id':'same' if kind=='duplicate' else str(len(ticks))})
    port.verify_hold=check
    try:
        result=port.verify_stationary_hold(None)
        assert result['outcome']==expected
        if kind=='stable':
            assert checks==[5,10,15,20]
            assert result['evidence']['duration_s']==1.
        else:assert len(ticks)==10
    finally:port.close()


def test_measured_motion_guard_interrupts_the_arm_segment(tmp_path,monkeypatch):
    port=setup(tmp_path,[candidate(z=.18)],monkeypatch)
    port.env.velocity_limit=np.full(6,100.)
    ticks=[]
    port.tick=lambda arm,ctx=None:ticks.append(arm.copy())
    def guard():
        if len(ticks)==3:raise ValueError('measured hand tilt')
    try:
        with pytest.raises(ValueError,match='measured hand tilt'):
            # setup replaces the instance method for the probe fixture; call
            # the actual movement loop to verify immediate guarded interruption.
            SimulatorSkillPort.move_angles(port,np.full(6,90.),None,duration=.2,motion_check=guard)
        assert len(ticks)==3
        assert not np.array_equal(ticks[-1],np.full(6,90.))
    finally:port.close()


def test_measured_view_violation_aborts_survey_instead_of_trying_another_motion(tmp_path):
    from simulator.environment import ExplorerEnv
    from simulator.inspection_planning import ObservedMotionViolation
    env=ExplorerEnv(render=False);port=SimulatorSkillPort(env,tmp_path/'abort')
    port.before_hold={};port.hold_rotation=np.eye(3)
    port.destination=dict(center=[.46,.34,.2],bounds_min=[.33,.21,.09],bounds_max=[.59,.47,.30],uncertainty_m=.012)
    calls=[]
    def unsafe(*args,**kwargs):calls.append('yaw');raise ObservedMotionViolation('measured tilt')
    port.held_yaw_view=unsafe
    port.look_at=lambda *args,**kwargs:calls.append('another motion')
    try:
        with pytest.raises(ObservedMotionViolation,match='measured tilt'):port.survey_destination(None)
        assert calls==['yaw']
    finally:port.close();env.close()


def test_internal_viewpoint_drive_is_declared_to_shared_executor(tmp_path,monkeypatch):
    port=setup(tmp_path,[],monkeypatch)
    def support(ctx):
        port.base_motion_sequence+=1
        return dict(outcome='success')
    port.support=support
    try:assert port.operations()['support'](None)==dict(outcome='success',base_moved=True)
    finally:port.close()


@pytest.mark.parametrize('time,evidence',[(0.,{}),(130.,{'shape':'raised_walls_with_visible_opening'})])
def test_navigation_requires_observed_recent_destination_even_if_camera_points_up(tmp_path,monkeypatch,time,evidence):
    port=setup(tmp_path,[],monkeypatch)
    port.env.data.time=time
    port.env.model=SimpleNamespace(camera=lambda name:SimpleNamespace(id=0))
    port.env.data.cam_xmat=np.array([np.diag([1.,1.,-1.]).ravel()])
    port.destination={'bounds_max':[.6,.4,.30],'evidence':evidence,'simulation_time':0.}
    port.survey_destination=lambda *args,**kwargs:pytest.fail('No navigation hint is permitted')
    try:
        result=port.carry(None)
        assert result['outcome']=='unknown'
        assert 'измеренного положения' in result['reason']
    finally:port.close()


def test_unreachable_preferred_optical_distance_does_not_prevent_a_safe_view(tmp_path,monkeypatch):
    port=setup(tmp_path,[],monkeypatch)
    port.env.command=np.r_[np.full(5,90.),170.,0.,0.,0.]
    port.destination=dict(center=[.6,.3,.2],bounds_max=[.7,.4,.3])
    calls=[]
    def view(env,point,**kwargs):
        calls.append(kwargs['distance_m'])
        if len(calls)<3:raise ValueError('Requested optical distance is outside the arm workspace')
        return SimpleNamespace(command=env.command[:6].copy())
    monkeypatch.setattr('simulator.inspection_planning.plan_camera_view',view)
    monkeypatch.setattr('simulator.inspection_planning.check_observed_box_path',lambda *a,**kw:None)
    port.verify_hold=lambda *a,**kw:dict(outcome='success',evidence={'frame_id':'tracked'})
    port.verify_stationary_hold=port.verify_hold
    port.move_angles=lambda *a,**kw:None
    try:
        assert port.reorient_for_destination(None)['outcome']=='success'
        assert len(calls)==3 and len(set(calls))==3
    finally:port.close()


def test_empty_grasp_requires_two_fresh_unambiguous_floor_observations(tmp_path,monkeypatch):
    port=setup(tmp_path,[],monkeypatch)
    floor=candidate(z=.015);floor['bounds_min'][2]=.005;floor['bounds_max'][2]=.025
    port.target=dict(floor);port.env.data.time=0.
    frames=iter([dict(frame_id='floor1',simulation_time=0.,object_candidates=[floor]),
                 dict(frame_id='floor2',simulation_time=.5,object_candidates=[floor])])
    port.scene=lambda ctx:next(frames)
    port.tick=lambda *a,**kw:setattr(port.env.data,'time',port.env.data.time+.05)
    try:
        result=port.verify_hold(None)
        assert result['outcome']=='failure'
        assert result['evidence']['source']=='two_frame_ground_reacquisition'
        assert result['evidence']['frames']==['floor1','floor2']
        assert port.before_hold is None
    finally:port.close()


@pytest.mark.parametrize('second',['same_frame','different_color','two_objects','missing'])
def test_uncertain_floor_reacquisition_cannot_authorise_opening_or_retry(tmp_path,monkeypatch,second):
    port=setup(tmp_path,[],monkeypatch)
    floor=candidate(z=.015);floor['bounds_min'][2]=.005;floor['bounds_max'][2]=.025
    port.target=dict(floor);port.env.data.time=0.
    other=dict(floor,appearance={'rgb_median':[.05,.1,.8]})
    choices={'same_frame':[floor],'different_color':[other],'two_objects':[floor,floor],'missing':[]}
    frames=iter([dict(frame_id='floor1',simulation_time=0.,object_candidates=[floor]),
                 dict(frame_id='floor1' if second=='same_frame' else 'floor2',simulation_time=.5,object_candidates=choices[second])])
    port.scene=lambda ctx:next(frames);port.tick=lambda *a,**kw:None
    port.move_angles=lambda *a,**kw:pytest.fail('opened or moved under ambiguous evidence')
    try:
        assert port.verify_hold(None)['outcome']=='unknown'
        assert port.recover_grasp(None)['outcome']=='unknown'
    finally:port.close()


def test_impossible_complete_closure_is_refused_before_descending(tmp_path,monkeypatch):
    port=setup(tmp_path,[],monkeypatch);port.grasp_xyz=np.array([.21,0.,.008])
    monkeypatch.setattr('simulator.grasp_planning.plan_gripper_closure',lambda *a,**kw:
        (_ for _ in ()).throw(ValueError('interior opening infeasible')))
    port.move_angles=lambda *a,**kw:pytest.fail('descended before complete closure preflight')
    try:
        with pytest.raises(ValueError,match='interior opening'):port.grasp(None)
    finally:port.close()


def test_initial_lift_does_not_rotate_into_high_transport_before_visual_proof(tmp_path,monkeypatch):
    port=setup(tmp_path,[],monkeypatch);port.grasp_xyz=np.array([.21,0.,.008])
    calls=[]
    monkeypatch.setattr('simulator.grasp_planning.solve_pick_ik',lambda env,point,gripper,**kw:
        calls.append(list(point)) or np.r_[np.full(5,90.),170.])
    port.env.solve_ik=lambda *a,**kw:pytest.fail('attempted unconfirmed high transport')
    port.move_angles=lambda *a,**kw:None
    try:
        assert port.lift(None)['outcome']=='success'
        assert calls==[[.21,0.,port.parameters['initial_lift_height_m']]]
        assert port.raise_verified_hold(None)['outcome']=='unknown'
    finally:port.close()
