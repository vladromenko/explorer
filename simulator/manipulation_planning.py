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


def _plan_oriented_translation(env,target,reference,*,check,check_path=False):
    """Return robot-only IK with fixed hand orientation, or no feasible plan."""
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .explorer_model import servo_to_joint
    start=env.command[:5].copy();gripper=float(env.command[5])
    scratch=mujoco.MjData(env.model);scratch.qpos[:]=env.data.qpos
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
        if np.linalg.norm(position)<=.008 and np.linalg.norm(rotation)<=np.deg2rad(2.):
            if check_path:
                # move_angles follows this straight joint-space segment with
                # a smooth time law. Check its interior, not just the endpoint.
                count=max(2,int(np.ceil(np.sum(np.abs(fit.x-start))/.5)))
                clear=True
                for fraction in np.linspace(0.,1.,count+1):
                    check()
                    if np.linalg.norm(errors(start+(fit.x-start)*fraction)[1])>np.deg2rad(2.):
                        clear=False;break
                if not clear:continue
            plans.append((float(np.linalg.norm(position)*10+np.linalg.norm(rotation)),fit.x))
    if plans:
        angles=min(plans,key=lambda row:row[0])[1]
        return np.r_[angles,gripper]
    return None


def plan_hold_probe(env,*,check=lambda:None):
    jaw=np.asarray(env.jaw_center(),float);reference=hand_rotation(env)
    radial=jaw[:2]-np.asarray(env.base_pose()[:2])
    directions=[('up',np.array([0.,0.,.02]))]
    if np.linalg.norm(radial)>.19:
        directions.append(('toward_base',np.r_[-.02*radial/np.linalg.norm(radial),0.]))
    for name,delta in directions:
        target=jaw+delta
        command=_plan_oriented_translation(env,target,reference,check=check)
        if command is not None:
            return HoldProbePlan(command,target,reference,name)
    raise ValueError('Проверочное движение недостижимо без изменения ориентации захвата')


@dataclass(frozen=True)
class TransportPosePlan:
    command: np.ndarray
    reference_rotation: np.ndarray
    retained: bool
    minimum_height_m: float
    maximum_reach_m: float


def transport_pose_satisfied(env,minimum_height_m,maximum_reach_m=.20):
    jaw=np.asarray(env.jaw_center(),float)
    reach=float(np.linalg.norm(jaw[:2]-np.asarray(env.base_pose()[:2])))
    return bool(np.isfinite(jaw).all() and jaw[2]>=minimum_height_m and reach<=maximum_reach_m)


def plan_transport_pose(env,minimum_height_m,*,maximum_reach_m=.20,check=lambda:None):
    """Keep a compact, raised verified hold; otherwise translate without a flip.

    These are own-robot pose constraints, not a collision-free path certificate.
    The caller must recheck measured pose and visible hold before base motion.
    """
    if (not np.isfinite([minimum_height_m,maximum_reach_m]).all() or
        not .18<=minimum_height_m<=.6 or not .10<=maximum_reach_m<=.20):
        raise ValueError('Некорректные ограничения транспортной позы')
    check();reference=hand_rotation(env)
    if transport_pose_satisfied(env,minimum_height_m,maximum_reach_m):
        return TransportPosePlan(env.command[:6].copy(),reference,True,minimum_height_m,maximum_reach_m)
    jaw=np.asarray(env.jaw_center(),float);base=np.asarray(env.base_pose()[:2])
    radial=jaw[:2]-base;length=float(np.linalg.norm(radial))
    # Reserve the unchanged IK positional tolerance inside the required envelope.
    target=jaw.copy();target[2]=max(jaw[2],minimum_height_m+.008)
    if length>maximum_reach_m-.008:
        target[:2]=base+radial/length*(maximum_reach_m-.008)
    command=_plan_oriented_translation(env,target,reference,check=check,check_path=True)
    if command is None:
        raise ValueError('Транспортная поза недостижима с сохранением ориентации захвата')
    return TransportPosePlan(command,reference,False,minimum_height_m,maximum_reach_m)
