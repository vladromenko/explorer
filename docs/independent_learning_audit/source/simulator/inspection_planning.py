"""Robot-only camera pose planning with explicit feasibility gates.

Targets are observed geometry or search directions. Model object coordinates,
labels and contacts are never queried. A low optimisation cost is not itself
evidence that a camera position is reachable.
"""
from dataclasses import dataclass

import numpy as np


class CameraViewUnavailable(ValueError):
    pass


class ObservedMotionViolation(CameraViewUnavailable):
    """An executed motion violated its observed guard; never try another pose."""


class CameraViewTrackingFailure(CameraViewUnavailable):
    """Unheld camera missed its requested view, within measured floor clearance."""


@dataclass(frozen=True)
class CameraViewPlan:
    command: np.ndarray
    position: np.ndarray
    lateral_error_m: float
    range_error_m: float
    hand_tilt_change_deg: float | None = None


def observed_rim_aims(target):
    """Frame a measured rim below the optical axis of a high wrist camera.

    A framing direction is not an object position or opening proof. All
    reach, height and independent RGB-D confirmation gates remain required.
    """
    center=np.asarray(target['center'],float);lo=np.asarray(target['bounds_min'],float);hi=np.asarray(target['bounds_max'],float)
    if center.shape!=(3,) or lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite(np.r_[center,lo,hi]).all() or np.any(hi<=lo):
        raise CameraViewUnavailable('Invalid observed rim framing extent')
    framing=np.r_[center[:2],hi[2]+.14]
    return [(framing.copy(),distance) for distance in (.45,.50,.40)]+[(center.copy(),.45)]


def _mesh_box_separated(env,scratch,geom,lo,hi):
    """Prove separation using an actual CAD convex-hull supporting plane.

    A world AABB is only a broad phase: its empty corners can cover a floor
    item under an inclined chassis surface. The mesh hull still encloses the
    complete CAD surface; no cavity, physical contact or object label is used.
    Absence of a separating face remains a conservative collision refusal.
    """
    import mujoco
    from scipy.spatial import ConvexHull
    if env.model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:return False
    cache=getattr(env,'_observed_cad_hulls',None)
    if cache is None or cache[0] is not env.model:
        cache=(env.model,{})
        env._observed_cad_hulls=cache
    mesh=int(env.model.geom_dataid[geom])
    if mesh not in cache[1]:
        start=int(env.model.mesh_vertadr[mesh]);count=int(env.model.mesh_vertnum[mesh])
        vertices=np.asarray(env.model.mesh_vert[start:start+count],dtype=float)
        if not np.isfinite(vertices).all():raise CameraViewUnavailable('Некорректная CAD-геометрия')
        cache[1][mesh]=ConvexHull(vertices).equations.copy()
    planes=cache[1][mesh]
    rotation=np.asarray(scratch.geom_xmat[geom]).reshape(3,3)
    normals=np.einsum('nj,ij->ni',planes[:,:3],rotation)
    center=(lo+hi)/2-scratch.geom_xpos[geom];half=(hi-lo)/2
    # Minimum signed distance of the entire padded observed box to each
    # outward hull plane. A positive gap proves disjoint volumes. Use an
    # additional numerical reserve, never an optimisation success flag.
    gap=np.einsum('ij,j->i',normals,center)+planes[:,3]-np.einsum('ij,j->i',abs(normals),half)
    return bool(np.any(gap>1e-6))


def check_observed_box_path(env,command,obstacle,*,check=lambda:None,base_target=None):
    """Conservative own-geometry sweep outside one measured occupied extent.

    No free cavity is assumed: even empty interior is treated as occupied.
    This checks the known occupied extent only, not an unobserved whole room.
    """
    import mujoco
    from itertools import product
    from .explorer_model import servo_to_joint
    command=np.asarray(command,float)
    lo=np.asarray(obstacle['bounds_min'],float).copy();hi=np.asarray(obstacle['bounds_max'],float).copy()
    uncertainty=float(obstacle['uncertainty_m'])
    if (command.shape!=(6,) or lo.shape!=(3,) or hi.shape!=(3,) or
        not np.isfinite(np.r_[command,lo,hi,uncertainty]).all() or
        np.any(hi<=lo) or not 0.<=uncertainty<=.02):
        raise CameraViewUnavailable('Нет достаточно точной границы препятствия для проверки обзора')
    margin=uncertainty+.012;lo[:2]-=margin;lo[2]=-.01;hi+=margin
    own=[]
    for i in range(env.model.ngeom):
        body=int(env.model.geom_bodyid[i])
        while body and env.model.body(body).name!='base_link':body=int(env.model.body_parentid[body])
        if body:own.append(i)
    own=np.array(own,dtype=int)
    aabb=np.asarray(env.model.geom_aabb[own],float)
    signs=np.array(list(product((-1.,1.),repeat=3)))
    corners=aabb[:,None,:3]+aabb[:,None,3:]*signs
    scratch=mujoco.MjData(env.model);scratch.qpos[:]=env.data.qpos
    start=env.command[:6].copy()
    count=max(2,int(np.ceil(np.sum(np.abs(command[:5]-start[:5]))/.5)))
    base_start=scratch.qpos[:2].copy()
    if base_target is not None:
        base_target=np.asarray(base_target,float)
        if base_target.shape!=(2,) or not np.isfinite(base_target).all() or np.linalg.norm(base_target-base_start)>.45:
            raise CameraViewUnavailable('Invalid finite local base alignment')
        count=max(count,int(np.ceil(np.linalg.norm(base_target-base_start)/.005)))
    for fraction in np.linspace(0.,1.,count+1):
        if base_target is not None:scratch.qpos[:2]=base_start+(base_target-base_start)*fraction
        check();q=servo_to_joint(start+(command-start)*fraction);scratch.qpos[env.joint_q]=q
        for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
            scratch.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
        mujoco.mj_kinematics(env.model,scratch);mujoco.mj_camlight(env.model,scratch)
        rotation=np.asarray(scratch.geom_xmat[own],float).reshape(-1,3,3)
        world=np.einsum('nvj,nij->nvi',corners,rotation)+scratch.geom_xpos[own,None,:]
        if not np.isfinite(world).all():raise CameraViewUnavailable('Некорректная геометрия пути обзора')
        intersects=np.all(world.max(1)>=lo,axis=1)&np.all(world.min(1)<=hi,axis=1)
        for geom in own[intersects]:
            if not _mesh_box_separated(env,scratch,int(geom),lo,hi):
                raise CameraViewUnavailable('Путь обзора пересекает измеренную границу препятствия')


def plan_camera_view(env,point,*,gripper,distance_m,min_camera_height_m,check=lambda:None,
                     maximum_hand_tilt_deg=None,reference_hand_gravity=None,minimum_jaw_height_m=None,
                     maximum_jaw_reach_m=None):
    import mujoco
    from scipy.optimize import least_squares
    from .explorer_model import servo_to_joint
    point=np.asarray(point,float)
    if (point.shape!=(3,) or not np.isfinite(point).all() or
        not np.isfinite([gripper,distance_m,min_camera_height_m]).all() or
        not 30.<=gripper<=170. or not .08<=distance_m<=1. or min_camera_height_m<.08):
        raise ValueError('Некорректная цель обзора камеры')
    if maximum_hand_tilt_deg is not None and (not np.isfinite(maximum_hand_tilt_deg) or not 0.<maximum_hand_tilt_deg<=3.):
        raise ValueError('Некорректный предел наклона удерживаемого предмета')
    if minimum_jaw_height_m is not None and (not np.isfinite(minimum_jaw_height_m) or minimum_jaw_height_m<.045):
        raise ValueError('Некорректная высота удерживаемого предмета')
    if maximum_jaw_reach_m is not None and (not np.isfinite(maximum_jaw_reach_m) or not .10<=maximum_jaw_reach_m<=.30):
        raise ValueError('Некорректная компактность удерживаемого предмета')
    scratch=mujoco.MjData(env.model);scratch.qpos[:]=env.data.qpos
    camera=env.model.camera('wrist_depth').id
    start=env.command[:5].copy()
    reference_gravity=env.data.body('arm5').xmat.reshape(3,3).T@np.array([0.,0.,1.])
    if reference_hand_gravity is not None:
        reference_gravity=np.asarray(reference_hand_gravity,float)
        if (maximum_hand_tilt_deg is None or reference_gravity.shape!=(3,) or
            not np.isfinite(reference_gravity).all() or abs(np.linalg.norm(reference_gravity)-1.)>1e-5):
            raise ValueError('Некорректная исходная ориентация захвата')
    def geometry(angles):
        q=servo_to_joint([*angles,gripper]);scratch.qpos[env.joint_q]=q
        for name,multiplier in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
            scratch.qpos[env.model.joint(name).qposadr[0]]=multiplier*q[5]
        mujoco.mj_kinematics(env.model,scratch);mujoco.mj_camlight(env.model,scratch)
        position=scratch.cam_xpos[camera].copy()
        optical=scratch.cam_xmat[camera].reshape(3,3).T@(point-position)
        gravity=scratch.body('arm5').xmat.reshape(3,3).T@np.array([0.,0.,1.])
        tilt=float(np.arccos(np.clip(reference_gravity@gravity,-1.,1.)))
        jaw=env.jaw_center(scratch)
        reach=float(np.linalg.norm(jaw[:2]-np.asarray(env.base_pose())[:2]))
        return position,optical,gravity,tilt,float(jaw[2]),reach
    def residual(angles):
        position,optical,gravity,_,jaw_height,reach=geometry(angles)
        return np.r_[optical[:2]*3.,(-optical[2]-distance_m),
            max(min_camera_height_m-position[2],0.)*30.,(angles-start)*.00003,
            [max(minimum_jaw_height_m-jaw_height,0.)*30.] if minimum_jaw_height_m is not None else np.empty(0),
            [max(reach-maximum_jaw_reach_m,0.)*30.] if maximum_jaw_reach_m is not None else np.empty(0),
            (gravity-reference_gravity)*8. if maximum_hand_tilt_deg is not None else np.empty(0)]
    seeds=[start,[90.,90.,90.,90.,90.],
           [50.,120.,30.,0.,90.],[130.,120.,30.,0.,90.],
           [50.,45.,120.,120.,90.],[130.,45.,120.,120.,90.]]
    priors=[]
    for prior in getattr(env,'observed_ik_seeds',[]):
        prior=np.asarray(prior,float)
        if prior.shape!=(5,) or not np.isfinite(prior).all() or np.any(prior<0) or np.any(prior>[180.,180.,180.,180.,270.]):continue
        priors.append((float(np.linalg.norm(residual(prior))),prior))
    seeds.extend(p for _,p in sorted(priors,key=lambda row:row[0])[:3])
    candidates=[];best_errors=None;best_cost=float('inf')
    for seed in seeds:
        check()
        fit=least_squares(residual,np.clip(seed,.001,[179.999]*4+[269.999]),
            bounds=([0.]*5,[180.]*4+[270.]),max_nfev=150)
        # Recompute FK: an optimiser's success flag or reported residual can
        # never replace the actual geometric constraints.
        position,optical,_,tilt,jaw_height,reach=geometry(fit.x)
        lateral=float(np.linalg.norm(optical[:2]));range_error=abs(float(-optical[2]-distance_m))
        height_deficit=max(float(min_camera_height_m-position[2]),0.)
        cost=lateral+range_error+height_deficit
        if cost<best_cost:
            best_cost=cost;best_errors=(lateral,range_error,height_deficit)
        if optical[2]<-.035 and lateral<=.04 and range_error<=.04 and height_deficit<=.008:
            if minimum_jaw_height_m is not None and jaw_height<minimum_jaw_height_m-.008:continue
            if maximum_jaw_reach_m is not None and reach>maximum_jaw_reach_m:continue
            if maximum_hand_tilt_deg is not None:
                if np.rad2deg(tilt)>maximum_hand_tilt_deg:continue
                # Check the same joint-space segment used by move_angles;
                # yaw about gravity is allowed, tipping the held cloth is not.
                count=max(2,int(np.ceil(np.sum(np.abs(fit.x-start))/.5)))
                allowed=True
                for fraction in np.linspace(0.,1.,count+1):
                    check()
                    if np.rad2deg(geometry(start+(fit.x-start)*fraction)[3])>maximum_hand_tilt_deg:
                        allowed=False;break
                if not allowed:continue
            candidates.append(CameraViewPlan(np.r_[fit.x,gripper],position,lateral,range_error,
                None if maximum_hand_tilt_deg is None else float(np.rad2deg(tilt))))
    if not candidates:
        lateral,range_error,height_deficit=best_errors
        qualifier=' с сохранением наклона захвата' if maximum_hand_tilt_deg is not None else ''
        raise CameraViewUnavailable('Обзорная поза недостижима'+qualifier+': ошибка наведения '
            f'{lateral*100:.1f} см, расстояния {range_error*100:.1f} см, нехватка высоты {height_deficit*100:.1f} см')
    return min(candidates,key=lambda row:row.lateral_error_m+row.range_error_m+
               float(np.linalg.norm(row.command[:5]-start))*.00003)
