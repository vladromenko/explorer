"""Pick an observed material patch without driving the chassis over the cloth."""
from dataclasses import dataclass
import copy

import numpy as np

from .rgbd_observation import deproject_depth,ObservationRefused


@dataclass(frozen=True)
class PickStance:
    base_xy: np.ndarray
    grasp_xyz: np.ndarray
    lift_xyz: np.ndarray
    fraction: float
    chassis_clearance_m: float


def solve_pick_ik(env,point,gripper,*,check=lambda:None):
    """Position-first floor approach, bounded to a 15 degree downward cone.

    A weighted orientation preference alone can trade centimetres of position
    for a prettier wrist angle. Here both independently measured FK bounds must
    pass; geometry, joint bounds and actuator physics remain unchanged.
    """
    import mujoco
    from scipy.optimize import least_squares
    from .explorer_model import servo_to_joint
    target=np.asarray(point,float);seed=env.command[:5].copy()
    if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('Invalid pick target')
    scratch=mujoco.MjData(env.model);scratch.qpos[:]=env.data.qpos
    def errors(angles):
        q=servo_to_joint([*angles,gripper]);scratch.qpos[env.joint_q]=q
        for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
            scratch.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
        mujoco.mj_kinematics(env.model,scratch)
        return env.jaw_center(scratch)-target,scratch.body('arm5').xmat.reshape(3,3)[:,2]
    def residual(angles):
        position,axis=errors(angles)
        return np.r_[position*100.,(axis-[0.,0.,-1.])*.65,(angles-seed)*.00003]
    plans=[]
    for start in (seed,[90.,125.,165.,150.,90.],[90.,45.,20.,45.,90.]):
        check();fit=least_squares(residual,np.clip(start,.001,[179.999]*4+[269.999]),
            bounds=([0.]*5,[180.]*4+[270.]),max_nfev=150)
        position,axis=errors(fit.x)
        if np.linalg.norm(position)<=.003 and -axis[2]>=np.cos(np.deg2rad(15.)):
            plans.append((float(np.linalg.norm(residual(fit.x))),fit.x))
    if not plans:raise ValueError('Нет точного захвата в допустимом наклоне кисти')
    return np.r_[min(plans,key=lambda row:row[0])[1],gripper]


def plan_gripper_closure(env,point,*,check=lambda:None):
    """Compensate the four-bar jaw translation throughout closure.

    Interpolating just the two endpoint joint poses sweeps the pads away from
    the measured material patch. Each waypoint instead solves the same world
    point at the actual linkage opening. Only own-robot FK is consulted.
    """
    branch=copy.copy(env);branch.command=env.command.copy()
    commands=[]
    for opening in np.linspace(30.,170.,29):
        check()
        command=solve_pick_ik(branch,point,float(opening),check=check)
        commands.append(command);branch.command[:6]=command
    commands=np.asarray(commands)
    return commands


def observed_pick_patch(record,target,base_pose,*,fraction=.25,lateral_fraction=.5,preferred_xyz=None):
    if not .10<=fraction<=.60:raise ValueError('Invalid material percentile')
    if not .20<=lateral_fraction<=.80:raise ValueError('Invalid lateral material percentile')
    frame=record['frame'];depth=np.asarray(record['depth'],float);robot=np.asarray(record['robot_depth'],float)
    points=deproject_depth(depth,frame['intrinsics'],frame['optical_to_world'])
    lo,hi=np.asarray(target['bounds_min'],float),np.asarray(target['bounds_max'],float)
    if robot.shape!=depth.shape:raise ObservationRefused('Нет фильтра собственной геометрии')
    valid=(np.isfinite(points).all(-1)&(depth>=.035)&(depth<=2.5)&
        (points[:,:,2]>=.004)&(points[:,:,2]<=.075)&
        np.all(points>=lo-.003,axis=-1)&np.all(points<=hi+.003,axis=-1)&
        ~(np.isfinite(robot)&(robot>0)&(np.abs(depth-robot)<=.008)))
    cloud=points[valid]
    if len(cloud)<30:raise ObservationRefused('Недостаточно измеренной поверхности для захвата')
    direction=np.asarray(target['center'],float)[:2]-np.asarray(base_pose,float)[:2]
    length=float(np.linalg.norm(direction))
    if length<.03:raise ObservationRefused('Предмет слишком близко к центру шасси')
    direction/=length;lateral=np.array([-direction[1],direction[0]])
    aim=np.quantile(np.einsum('ij,j->i',cloud[:,:2],direction),fraction)*direction+np.quantile(
        np.einsum('ij,j->i',cloud[:,:2],lateral),lateral_fraction)*lateral
    if preferred_xyz is not None:aim=np.asarray(preferred_xyz,float)[:2]
    nearest=cloud[np.argmin(np.linalg.norm(cloud[:,:2]-aim,axis=1))]
    if preferred_xyz is not None and np.linalg.norm(nearest[:2]-aim)>.015:
        raise ObservationRefused('Ранее выбранный участок ткани закрыт или сместился')
    patch=cloud[np.linalg.norm(cloud[:,:2]-nearest[:2],axis=1)<=.008]
    if len(patch)<8:raise ObservationRefused('Точка не подтверждена измеренным участком ткани')
    return np.r_[nearest[:2],np.median(patch[:,2])]


def mesh_below_plane(vertices,faces,height):
    """Clip CAD triangle surfaces at cloth height, retaining crossing edges.

    Projecting an entire raised chassis just because one rear vertex is low
    fills empty air under its front. The clipped mesh preserves actual edges;
    it does not remove any surface that could touch the observed low target.
    """
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    selected=[vertices[vertices[:,2]<=height]]
    triangles=vertices[faces]
    for first,second in ((0,1),(1,2),(2,0)):
        a,b=triangles[:,first],triangles[:,second]
        crosses=(a[:,2]<=height)!=(b[:,2]<=height)
        a,b=a[crosses],b[crosses]
        if len(a):selected.append(a+(b-a)*((height-a[:,2])/(b[:,2]-a[:,2]))[:,None])
    return np.concatenate(selected)


def _chassis_geometry_bounds(env,data,*,maximum_height=None):
    """Known rigid chassis only; arm reach is checked separately by IK."""
    own=[]
    for i in range(env.model.ngeom):
        body=int(env.model.geom_bodyid[i]);arm=False
        while body and env.model.body(body).name!='base_link':
            arm|=env.model.body(body).name=='arm1';body=int(env.model.body_parentid[body])
        if body and not arm:own.append(i)
    import mujoco
    bounds=[]
    for i in own:
        rotation=np.asarray(data.geom_xmat[i],float).reshape(3,3);position=np.asarray(data.geom_xpos[i],float)
        kind=env.model.geom_type[i];size=np.asarray(env.model.geom_size[i],float)
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(env.model.geom_dataid[i]);a,n=int(env.model.mesh_vertadr[mesh]),int(env.model.mesh_vertnum[mesh])
            vertices=np.asarray(env.model.mesh_vert[a:a+n],dtype=float)
            world=np.einsum('vj,ij->vi',vertices,rotation)+position
            if maximum_height is not None:
                fa,fn=int(env.model.mesh_faceadr[mesh]),int(env.model.mesh_facenum[mesh])
                world=mesh_below_plane(world,env.model.mesh_face[fa:fa+fn],maximum_height)
                if not len(world):continue
            bounds.append([world.min(0),world.max(0)]);continue
        if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent=np.abs(rotation[:,2])*size[1]+size[0]
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent=np.linalg.norm(rotation[:,:2],axis=1)*size[0]+np.abs(rotation[:,2])*size[1]
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:extent=np.full(3,size[0])
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:extent=np.einsum('ij,j->i',np.abs(rotation),size)
        else:raise ValueError('Unsupported own chassis geometry')
        bounds.append([position-extent,position+extent])
    bounds=np.asarray(bounds).reshape(-1,2,3)
    if not np.isfinite(bounds).all():raise ValueError('Invalid chassis geometry')
    return bounds


def chassis_bounds(env,data):
    bounds=_chassis_geometry_bounds(env,data)
    return bounds[:,0,:2].min(0),bounds[:,1,:2].max(0)


def low_chassis_clearance(env,data,target):
    """Check parts that can touch low cloth, without filling the gap between wheels."""
    lo,hi=np.asarray(target['bounds_min']),np.asarray(target['bounds_max'])
    uncertainty=float(target['uncertainty_m'])
    bounds=_chassis_geometry_bounds(env,data,maximum_height=hi[2]+uncertainty+.012)
    low=bounds[bounds[:,0,2]<=hi[2]+uncertainty+.012]
    if not len(low):return float('inf')
    gaps=np.maximum(np.maximum(lo[:2]-uncertainty-low[:,1,:2],
                               low[:,0,:2]-hi[:2]-uncertainty),0.)
    return float(np.linalg.norm(gaps,axis=1).min())


def plan_pick_stance(env,target,record,destination,*,check=lambda:None,preferred_fraction=.50,lift_clearance=.12,surface_offset=.010,
                     lateral_fraction=.5,preferred_standoff=.24,preferred_angle=0.,initial_lift_height=.10):
    import mujoco
    from .explorer_model import servo_to_joint
    pose=np.asarray(env.base_pose(),float);body_direction=np.array([np.cos(pose[2]),np.sin(pose[2])])
    branch=copy.copy(env);branch.data=mujoco.MjData(env.model);branch.data.qpos[:]=env.data.qpos
    address=env.model.joint('base_free').qposadr[0]
    lo,hi=np.asarray(target['bounds_min'],float),np.asarray(target['bounds_max'],float)
    uncertainty=float(target['uncertainty_m']);choices=[]
    def reserve_ik(point,gripper,downward=True):
        command=(solve_pick_ik(branch,point,gripper,check=check) if downward else
                 branch.solve_ik(point,gripper=gripper,downward=False))
        scratch=mujoco.MjData(env.model);scratch.qpos[:]=branch.data.qpos
        q=servo_to_joint(command);scratch.qpos[env.joint_q]=q
        for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
            scratch.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
        mujoco.mj_kinematics(env.model,scratch)
        if np.linalg.norm(branch.jaw_center(scratch)-point)>.003:
            raise ValueError('No remaining IK margin for odometry and fresh depth correction')
    # Use the actual learnable fraction, rather than snapping every proposal
    # to one of five hard-coded percentiles. Mecanum translation can also
    # approach the middle from a side instead of always pinching the near end.
    low_bounds=_chassis_geometry_bounds(env,env.data,maximum_height=hi[2]+uncertainty+.012)
    basket_bounds=_chassis_geometry_bounds(env,env.data,
        maximum_height=float(destination['bounds_max'][2])+float(destination['uncertainty_m'])+.012)
    def swept_clearance(bounds,base,obstacle):
        delta=base-pose[:2]
        lower=bounds[:,0,:2]+np.minimum(delta,0.)
        upper=bounds[:,1,:2]+np.maximum(delta,0.)
        padding=float(obstacle['uncertainty_m'])
        gaps=np.maximum(np.maximum(np.asarray(obstacle['bounds_min'])[:2]-padding-upper,
            lower-np.asarray(obstacle['bounds_max'])[:2]-padding),0.)
        return float(np.linalg.norm(gaps,axis=1).min())
    fractions=sorted(set((.12,.20,.30,.40,.50,float(preferred_fraction))),
        key=lambda f:abs(f-preferred_fraction))
    for fraction in fractions:
        point=observed_pick_patch(record,target,pose,fraction=fraction,lateral_fraction=lateral_fraction)
        grasp=np.r_[point[:2],max(.017,float(point[2])+surface_offset)]
        choices=[]
        angles=sorted(set((0.,-40.,40.,-65.,65.,float(preferred_angle))),key=lambda a:abs(a-preferred_angle))
        for angle in angles:
            yaw=pose[2]+np.deg2rad(angle);direction=np.array([np.cos(yaw),np.sin(yaw)])
            for stand_off in sorted(set((.24,.22,.26,.20,.28,.18,float(preferred_standoff))),key=lambda d:abs(d-preferred_standoff)):
                check();base=point[:2]-stand_off*direction
                if np.linalg.norm(base-pose[:2])>.22:continue
                clearance=swept_clearance(low_bounds,base,target)
                if clearance<.04 or swept_clearance(basket_bounds,base,destination)<.012:continue
                branch.data.qpos[address:address+2]=base;mujoco.mj_kinematics(branch.model,branch.data)
                lift=np.r_[base+.16*body_direction,max(.24,float(destination['bounds_max'][2])+lift_clearance)]
                try:
                    reserve_ik(grasp,30.);reserve_ik(grasp,170.)
                    reserve_ik([*point[:2],initial_lift_height],170.)
                    reserve_ik(lift,170.,downward=False)
                except ValueError:continue
                choices.append(((abs(stand_off-preferred_standoff),abs(angle-preferred_angle),float(np.linalg.norm(base-pose[:2]))),
                    PickStance(base,point,lift,fraction,clearance)))
        # Endpoints alone can be feasible while an intermediate linkage
        # opening is unreachable. Reserve the WHOLE closing trajectory before
        # moving the base or descending. No cloth/contact state is consulted.
        for _,stance in sorted(choices,key=lambda choice:choice[0]):
            check();branch.data.qpos[address:address+2]=stance.base_xy
            branch.command=env.command.copy();mujoco.mj_kinematics(branch.model,branch.data)
            try:plan_gripper_closure(branch,grasp,check=check)
            except ValueError:continue
            return stance
    raise ObservationRefused('Не найден подъезд без наезда на ткань с достижимым захватом и подъёмом')
