from types import SimpleNamespace

import numpy as np
import pytest

from simulator.environment import ExplorerEnv
from simulator.inspection_planning import plan_camera_view,CameraViewUnavailable


def test_feasible_view_satisfies_each_constraint_and_preserves_state():
    env=ExplorerEnv(render=False)
    before=env.data.qpos.copy();command=env.command.copy()
    try:
        view=plan_camera_view(env,[.4,.24,.15],gripper=170.,distance_m=.4,min_camera_height_m=.3)
        assert view.lateral_error_m<=.04 and view.range_error_m<=.04
        assert view.position[2]>=.292 and view.command[5]==170.
        np.testing.assert_array_equal(before,env.data.qpos)
        np.testing.assert_array_equal(command,env.command)
    finally:env.close()


def test_unreachable_camera_height_is_not_accepted_as_an_approximate_view():
    env=ExplorerEnv(render=False)
    try:
        with pytest.raises(CameraViewUnavailable,match='нехватка высоты'):
            plan_camera_view(env,[.4,.24,.15],gripper=30.,distance_m=.4,min_camera_height_m=1.0)
    finally:env.close()


def test_optimizer_success_cannot_hide_wrong_pose(monkeypatch):
    env=ExplorerEnv(render=False)
    monkeypatch.setattr('scipy.optimize.least_squares',lambda fun,start,**kwargs:
        SimpleNamespace(x=np.asarray(start),fun=np.zeros(8),success=True))
    try:
        with pytest.raises(CameraViewUnavailable):
            plan_camera_view(env,[.4,.24,.15],gripper=30.,distance_m=.4,min_camera_height_m=1.0)
    finally:env.close()


def test_cancelled_planning_does_not_change_robot():
    env=ExplorerEnv(render=False);before=env.data.qpos.copy()
    def stop():raise InterruptedError('operator pause')
    try:
        with pytest.raises(InterruptedError):
            plan_camera_view(env,[.4,.24,.15],gripper=30.,distance_m=.4,min_camera_height_m=.3,check=stop)
        np.testing.assert_array_equal(before,env.data.qpos)
    finally:env.close()


def held_pose(env):
    import mujoco
    from simulator.explorer_model import servo_to_joint
    env.command[:6]=[91.23,81.02,28.21,5.34,81.53,170.]
    q=servo_to_joint(env.command[:6]);env.data.qpos[env.joint_q]=q
    for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
        env.data.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
    mujoco.mj_forward(env.model,env.data)


def test_grip_preserving_camera_view_accepts_an_already_visible_target():
    from simulator.manipulation_planning import hand_rotation
    env=ExplorerEnv(render=False)
    try:
        held_pose(env);before=env.data.qpos.copy()
        camera=env.model.camera('wrist_depth').id
        point=env.data.cam_xpos[camera]-.24*env.data.cam_xmat[camera].reshape(3,3)[:,2]
        gravity=hand_rotation(env).T@np.array([0.,0.,1.])
        plan=plan_camera_view(env,point,gripper=170.,distance_m=.24,min_camera_height_m=.12,
            maximum_hand_tilt_deg=2.,reference_hand_gravity=gravity)
        assert plan.hand_tilt_change_deg<=2.
        assert plan.command[5]==170.
        np.testing.assert_array_equal(env.data.qpos,before)
    finally:env.close()


def test_destination_view_cannot_tip_a_verified_hold_to_satisfy_camera_aim():
    env=ExplorerEnv(render=False)
    try:
        held_pose(env);before=env.data.qpos.copy()
        point=[.4,.24,.15]
        # The unconstrained view is attainable but tips the hand. The same
        # target must be refused under the held-object constraint.
        free=plan_camera_view(env,point,gripper=170.,distance_m=.45,min_camera_height_m=.30)
        assert free.lateral_error_m<=.04
        with pytest.raises(CameraViewUnavailable,match='с сохранением наклона'):
            plan_camera_view(env,point,gripper=170.,distance_m=.45,min_camera_height_m=.30,
                maximum_hand_tilt_deg=2.)
        np.testing.assert_array_equal(env.data.qpos,before)
    finally:env.close()
