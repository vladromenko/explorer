"""Small hold-verification motions preserving measured hand orientation.

Only robot FK, joint state and base odometry enter the planner. A probe must
not turn a translation request into an unconstrained change of wrist pitch.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HoldProbePlan:
    command: np.ndarray
    target: np.ndarray
    reference_rotation: np.ndarray
    direction: str


def hand_rotation(env):
    return np.asarray(env.data.body('arm5').xmat).reshape(3,3).copy()


def plan_hold_probe(env,*,check=lambda:None):
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .explorer_model import servo_to_joint
    jaw=np.asarray(env.jaw_center(),float);reference=hand_rotation(env)
    start=env.command[:5].copy();gripper=float(env.command[5])
    radial=jaw[:2]-np.asarray(env.base_pose()[:2])
    directions=[('up',np.array([0.,0.,.02]))]
    if np.linalg.norm(radial)>.19:
        directions.append(('toward_base',np.r_[-.02*radial/np.linalg.norm(radial),0.]))
    scratch=mujoco.MjData(env.model);scratch.qpos[:]=env.data.qpos
    for name,delta in directions:
        target=jaw+delta
        def errors(angles):
            q=servo_to_joint([*angles,gripper]);scratch.qpos[env.joint_q]=q
            for link,multiplier in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
                scratch.qpos[env.model.joint(link).qposadr[0]]=multiplier*q[5]
            mujoco.mj_forward(env.model,scratch)
            position=env.jaw_center(scratch)-target
            rotation=reference.T@scratch.body('arm5').xmat.reshape(3,3)
            return position,Rotation.from_matrix(rotation).as_rotvec()
        def residual(angles):
            position,rotation=errors(angles)
            return np.r_[position*10.,rotation,(angles-start)*.00003]
        plans=[]
        for seed in (start,[90.,125.,165.,150.,90.],[90.,45.,20.,45.,90.]):
            check()
            fit=least_squares(residual,np.clip(seed,.001,[179.999]*4+[269.999]),
                bounds=([0.]*5,[180.]*4+[270.]),max_nfev=150)
            position,rotation=errors(fit.x)
            # Preserve the existing 8 mm positional limit; additionally demand
            # less than two degrees of orientation error, rather than free IK.
            if np.linalg.norm(position)<=.008 and np.linalg.norm(rotation)<=np.deg2rad(2.):
                plans.append((float(np.linalg.norm(position)*10+np.linalg.norm(rotation)),fit.x))
        if plans:
            angles=min(plans,key=lambda row:row[0])[1]
            return HoldProbePlan(np.r_[angles,gripper],target,reference,name)
    raise ValueError('Проверочное движение недостижимо без изменения ориентации захвата')
