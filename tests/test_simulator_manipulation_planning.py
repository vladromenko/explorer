import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simulator.environment import ExplorerEnv
from simulator.explorer_model import servo_to_joint
from simulator.manipulation_planning import plan_hold_probe,hand_rotation


def apply_kinematic_pose(env,command):
    q=servo_to_joint(command);env.data.qpos[env.joint_q]=q
    for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
        env.data.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
    mujoco.mj_forward(env.model,env.data)


def test_hold_probe_preserves_orientation_instead_of_using_free_pitch():
    env=ExplorerEnv(render=False)
    try:
        env.command[:6]=[91.0791214,65.3229147,50.8938333,0.,81.53469,170.]
        apply_kinematic_pose(env,env.command[:6])
        before=env.data.qpos.copy();jaw=env.jaw_center().copy();reference=hand_rotation(env)
        plan=plan_hold_probe(env)
        np.testing.assert_array_equal(env.data.qpos,before)
        assert plan.command[5]==170.
        apply_kinematic_pose(env,plan.command)
        assert np.linalg.norm(env.jaw_center()-plan.target)<=.008
        assert Rotation.from_matrix(reference.T@hand_rotation(env)).magnitude()<=np.deg2rad(2.)
        assert np.linalg.norm(plan.target-jaw)==pytest.approx(.02)
    finally:env.close()


def test_cancelled_probe_planning_never_mutates_robot():
    env=ExplorerEnv(render=False);before=env.data.qpos.copy()
    def stop():raise InterruptedError('operator pause')
    try:
        with pytest.raises(InterruptedError):plan_hold_probe(env,check=stop)
        np.testing.assert_array_equal(env.data.qpos,before)
    finally:env.close()
