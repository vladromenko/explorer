import numpy as np
import pytest

from simulator.grasp_planning import observed_pick_patch,plan_pick_stance
from simulator.rgbd_observation import extract_candidates,ObservationRefused


def test_low_chassis_mesh_keeps_intersecting_edges_without_projecting_raised_front():
    from simulator.grasp_planning import mesh_below_plane
    vertices=np.array([[0.,0.,0.],[1.,0.,1.],[0.,1.,1.]])
    clipped=mesh_below_plane(vertices,[[0,1,2]],.25)
    np.testing.assert_allclose(clipped,[[0.,0.,0.],[.25,0.,.25],[0.,.25,.25]])
    assert not len(mesh_below_plane(vertices,[[0,1,2]],-.1))


def material():
    depth=np.full((128,256),.4);depth[44:84,46:210]=.392
    pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[.34,0.,.4]
    k=dict(fx=400.,fy=400.,cx=127.5,cy=63.5)
    robot=np.full(depth.shape,8.)
    target=extract_candidates(depth,k,pose,robot_depth=robot)['low_objects'][0]
    record=dict(frame={'intrinsics':k,'optical_to_world':pose.tolist()},depth=depth,robot_depth=robot)
    return record,target


def test_pick_is_on_measured_near_material_rather_than_an_assumed_box_center():
    record,target=material()
    point=observed_pick_patch(record,target,[0.,0.,0.])
    assert target['bounds_min'][0]<point[0]<target['center'][0]
    assert point[2]==pytest.approx(.008)


def test_own_robot_or_missing_depth_cannot_supply_a_pick_patch():
    record,target=material();record['robot_depth']=record['depth'].copy()
    with pytest.raises(ObservationRefused):observed_pick_patch(record,target,[0.,0.,0.])


def test_pick_stance_keeps_chassis_away_from_the_entire_cloth_and_checks_high_lift():
    from simulator.environment import ExplorerEnv
    env=ExplorerEnv(render=False);record,target=material();before=env.data.qpos.copy()
    try:
        stance=plan_pick_stance(env,target,record,{'bounds_min':[.33,.21,.08],
            'bounds_max':[.59,.47,.3],'uncertainty_m':.012})
        assert stance.chassis_clearance_m>=.04
        assert stance.lift_xyz[2]==pytest.approx(.42)
        assert np.linalg.norm(stance.base_xy-np.asarray(env.base_pose()[:2]))<=.22
        assert target['bounds_min'][0]<stance.grasp_xyz[0]<target['bounds_max'][0]
        np.testing.assert_array_equal(env.data.qpos,before)
    finally:env.close()


@pytest.mark.parametrize('gripper',[30.,170.])
def test_position_first_pick_reaches_material_without_trading_position_for_wrist_angle(gripper):
    import mujoco
    from simulator.environment import ExplorerEnv
    from simulator.grasp_planning import solve_pick_ik
    from simulator.explorer_model import servo_to_joint
    env=ExplorerEnv(render=False);before=env.data.qpos.copy();point=np.array([.21,0.,.017])
    try:
        command=solve_pick_ik(env,point,gripper)
        scratch=mujoco.MjData(env.model);scratch.qpos[:]=before
        q=servo_to_joint(command);scratch.qpos[env.joint_q]=q
        for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
            scratch.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
        mujoco.mj_forward(env.model,scratch)
        assert np.linalg.norm(env.jaw_center(scratch)-point)<.003
        assert -scratch.body('arm5').xmat.reshape(3,3)[2,2]>=np.cos(np.deg2rad(15.))
        np.testing.assert_array_equal(env.data.qpos,before)
        with pytest.raises(ValueError):solve_pick_ik(env,[.6,0.,.017],gripper)
    finally:env.close()


def test_closure_keeps_pad_center_on_observed_material_throughout_linkage_motion():
    import mujoco
    from simulator.environment import ExplorerEnv
    from simulator.explorer_model import servo_to_joint
    from simulator.grasp_planning import plan_gripper_closure
    env=ExplorerEnv(render=False);before=env.data.qpos.copy();commands=env.command.copy()
    point=np.array([.21,0.,.017])
    try:
        path=plan_gripper_closure(env,point)
        assert path[0,5]==30. and path[-1,5]==170.
        assert np.all(np.diff(path[:,5])>0.)
        scratch=mujoco.MjData(env.model);scratch.qpos[:]=before
        # Include interior positions actually followed by the controller, not
        # just the optimised waypoints. Never read cloth state or contacts.
        for a,b in zip(path[:-1],path[1:]):
            for u in (0.,.5,1.):
                q=servo_to_joint(a+(b-a)*u);scratch.qpos[env.joint_q]=q
                for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
                    scratch.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
                mujoco.mj_forward(env.model,scratch)
                assert np.linalg.norm(env.jaw_center(scratch)-point)<=.003
        np.testing.assert_array_equal(env.data.qpos,before)
        np.testing.assert_array_equal(env.command,commands)
    finally:env.close()
