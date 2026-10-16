"""Own-robot, observation-anchored motion repairs; no task truth inputs."""
import numpy as np


def predicted_anchor_visible(env,data,relative_anchor,*,raycast=True):
    import mujoco
    from .explorer_model import model_config
    camera=env.model.camera('wrist_depth').id
    rotation=data.body('arm5').xmat.reshape(3,3)
    anchor=env.jaw_center(data)+rotation@np.asarray(relative_anchor,float)
    origin=data.cam_xpos[camera];view=data.cam_xmat[camera].reshape(3,3).T@(anchor-origin)
    z=float(-view[2]);config=model_config()['camera']
    if not .035<z<2.5:return False
    pixel=np.array([config['fx']*view[0]/z+config['cx'],config['cy']-config['fy']*view[1]/z])
    if not (16<=pixel[0]<config['width']-16 and 16<=pixel[1]<config['height']-16):return False
    if raycast:
        direction=anchor-origin;length=float(np.linalg.norm(direction));geom=np.full(1,-1,dtype=np.int32)
        # Group 2 contains own robot CAD, as in the existing self-depth mask.
        # Scene objects, floor and cloth are excluded from the ray query.
        near=float(config['near_m'])*length/z
        hit=mujoco.mj_ray(env.model,data,origin+direction/length*near,direction/length,
            np.array([0,0,1,0,0,0],dtype=np.uint8),0,-1,geom)
        if hit>=0 and hit<length-near-.008:return False
    return True


def plan_visible_lift(env,target,relative_anchor,*,check=lambda:None):
    """Search IK branches with a camera-visible material anchor along the path.

    Prediction merely reserves a view; actual RGBD hold checks during execution
    still decide whether a grasp survives. No predicted view grants evidence.
    """
    import mujoco
    from .explorer_model import servo_to_joint
    target=np.asarray(target,float);relative_anchor=np.asarray(relative_anchor,float)
    if (target.shape!=(3,) or relative_anchor.shape!=(3,) or not np.isfinite(np.r_[target,relative_anchor]).all()
        or np.linalg.norm(relative_anchor)>=.08):raise ValueError('Нет измеренного участка ткани для планирования обзора')
    start=env.command[:6].copy();scratch=mujoco.MjData(env.model);scratch.qpos[:]=env.data.qpos
    seeds=[start[:5].copy(),[90.,125.,165.,150.,90.],[90.,45.,20.,45.,90.]]
    seeds+=list(getattr(env,'observed_ik_seeds',[]))
    for offset in (-45.,45.,-90.,90.):
        seed=start[:5].copy();seed[4]=np.clip(seed[4]+offset,0.,270.);seeds.append(seed)
    candidates=[];seen=[]
    for seed in seeds[:14]:
        check()
        try:command=env.solve_ik(target,seed,gripper=170.,downward=False)
        except ValueError:continue
        if any(np.linalg.norm(command-old)<.2 for old in seen):continue
        seen.append(command)
        count=max(2,int(np.ceil(np.sum(abs(command[:5]-start[:5]))/.5)))
        clear=True
        for fraction in np.linspace(0.,1.,count+1):
            check();q=servo_to_joint(start+(command-start)*fraction);scratch.qpos[env.joint_q]=q
            for name,m in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
                scratch.qpos[env.model.joint(name).qposadr[0]]=m*q[5]
            mujoco.mj_kinematics(env.model,scratch);mujoco.mj_camlight(env.model,scratch)
            if not predicted_anchor_visible(env,scratch,relative_anchor):clear=False;break
        if clear:candidates.append((float(np.linalg.norm(command-start)),command,count+1))
    if not candidates:raise ValueError('Не найден высокий подъём с наблюдаемым участком ткани на всём пути')
    _,command,count=min(candidates,key=lambda x:x[0])
    return command,count
