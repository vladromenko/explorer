"""Robot-only camera pose planning with explicit feasibility gates.

Targets are observed geometry or search directions. Model object coordinates,
labels and contacts are never queried. A low optimisation cost is not itself
evidence that a camera position is reachable.
"""
from dataclasses import dataclass

import numpy as np


class CameraViewUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class CameraViewPlan:
    command: np.ndarray
    position: np.ndarray
    lateral_error_m: float
    range_error_m: float


def plan_camera_view(env,point,*,gripper,distance_m,min_camera_height_m,check=lambda:None):
    import mujoco
    from scipy.optimize import least_squares
    from .explorer_model import servo_to_joint
    point=np.asarray(point,float)
    if (point.shape!=(3,) or not np.isfinite(point).all() or
        not np.isfinite([gripper,distance_m,min_camera_height_m]).all() or
        not 30.<=gripper<=170. or not .08<=distance_m<=1. or min_camera_height_m<.08):
        raise ValueError('Некорректная цель обзора камеры')
    scratch=mujoco.MjData(env.model);scratch.qpos[:]=env.data.qpos
    camera=env.model.camera('wrist_depth').id
    start=env.command[:5].copy()
    def geometry(angles):
        q=servo_to_joint([*angles,gripper]);scratch.qpos[env.joint_q]=q
        for name,multiplier in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
            scratch.qpos[env.model.joint(name).qposadr[0]]=multiplier*q[5]
        mujoco.mj_forward(env.model,scratch)
        position=scratch.cam_xpos[camera].copy()
        optical=scratch.cam_xmat[camera].reshape(3,3).T@(point-position)
        return position,optical
    def residual(angles):
        position,optical=geometry(angles)
        return np.r_[optical[:2]*3.,(-optical[2]-distance_m),
            max(min_camera_height_m-position[2],0.)*30.,(angles-start)*.00003]
    seeds=(start,[90.,90.,90.,90.,90.],
           [50.,120.,30.,0.,90.],[130.,120.,30.,0.,90.],
           [50.,45.,120.,120.,90.],[130.,45.,120.,120.,90.])
    candidates=[];best_errors=None;best_cost=float('inf')
    for seed in seeds:
        check()
        fit=least_squares(residual,np.clip(seed,.001,[179.999]*4+[269.999]),
            bounds=([0.]*5,[180.]*4+[270.]),max_nfev=150)
        # Recompute FK: an optimiser's success flag or reported residual can
        # never replace the actual geometric constraints.
        position,optical=geometry(fit.x)
        lateral=float(np.linalg.norm(optical[:2]));range_error=abs(float(-optical[2]-distance_m))
        height_deficit=max(float(min_camera_height_m-position[2]),0.)
        cost=lateral+range_error+height_deficit
        if cost<best_cost:
            best_cost=cost;best_errors=(lateral,range_error,height_deficit)
        if optical[2]<-.035 and lateral<=.04 and range_error<=.04 and height_deficit<=.008:
            candidates.append(CameraViewPlan(np.r_[fit.x,gripper],position,lateral,range_error))
    if not candidates:
        lateral,range_error,height_deficit=best_errors
        raise CameraViewUnavailable('Обзорная поза недостижима: ошибка наведения '
            f'{lateral*100:.1f} см, расстояния {range_error*100:.1f} см, нехватка высоты {height_deficit*100:.1f} см')
    return min(candidates,key=lambda row:row.lateral_error_m+row.range_error_m+
               float(np.linalg.norm(row.command[:5]-start))*.00003)
