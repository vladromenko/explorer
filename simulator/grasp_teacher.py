"""Offline expert planning with independent MuJoCo data, never a learned input.

The real trajectory is recorded only after planning. No object is attached,
teleported, made lighter or accepted using a planning result.
"""
import copy

import mujoco
import numpy as np

from .environment import ExplorerEnv,PERIOD


def fork_physics(env):
    branch=object.__new__(ExplorerEnv)
    branch.__dict__=env.__dict__.copy()
    branch.data=mujoco.MjData(env.model)
    specification=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(env.model,specification))
    mujoco.mj_getState(env.model,env.data,state,specification)
    mujoco.mj_setState(env.model,branch.data,state,specification)
    mujoco.mj_forward(env.model,branch.data)
    for name in ('command','pending_command','arm_velocity','arm_acceleration'):
        setattr(branch,name,getattr(env,name).copy())
    branch.contacts=list(env.contacts)
    branch.placement_diagnostics=copy.deepcopy(getattr(env,'placement_diagnostics',{}))
    branch.renderer=None;branch.overview_renderer=None;branch.render_enabled=False;branch._rgb_cache={}
    return branch


def move(branch,goal,duration=2.):
    start=branch.pending_command[:6].copy()
    count=max(round(duration/PERIOD),round(np.max(np.abs(goal-start)/branch.velocity_limit)/PERIOD)+10)
    for index in range(count):
        u=(index+1)/count;blend=u*u*u*(10.-15.*u+6.*u*u)
        branch.step(np.r_[start+(goal-start)*blend,0.,0.,0.])
        if branch.last_error:raise ValueError(branch.last_error)
    for _ in range(12):branch.step(np.r_[goal,0.,0.,0.])


def plan_grasp(env,center,approach,lift_goal,cancelled=None,downward_lift=False):
    # The first candidate is the unchanged legacy motion, preserving every
    # existing grasp that its full-physics forecast can already hold.
    candidates=[(.017,0.,0.),(.014,0.,0.),(.020,0.,0.),
                (.014,-.008,0.),(.014,.008,0.),(.012,0.,0.)]
    diagnostics=[];first=None
    for height,dx,dy in candidates:
        if cancelled:cancelled()
        branch=fork_physics(env)
        try:
            xy=np.asarray(center[:2])+[dx,dy]
            lower_height=max(.024,float(center[2])+.018) if first is None else max(.024,height+.007)
            lowered=branch.solve_ik([*xy,lower_height],approach[:5])
            move(branch,lowered)
            closed=branch.solve_ik([*xy,height],lowered[:5],gripper=170.)
            move(branch,closed)
            lifted=branch.solve_ik(lift_goal,closed[:5],gripper=170.,downward=downward_lift)
            move(branch,lifted)
            for _ in range(30):branch.step(np.r_[lifted,0.,0.,0.])
            plan=dict(lowered=lowered,closed=closed,lifted=lifted)
            if first is None:first=plan
            held=branch.ever_lifted
            diagnostics.append(dict(height_m=height,offset_m=[dx,dy],predicted_held=held,error=branch.last_error))
            if held:return plan,diagnostics
        except ValueError as error:
            diagnostics.append(dict(height_m=height,offset_m=[dx,dy],predicted_held=False,error=str(error)))
        finally:branch.close()
    if first is None:raise ValueError('No feasible grasp trajectory')
    # A failed forecast remains an attempted, possibly failed recording. It is
    # never turned into success by changing acceptance or simulation state.
    return first,diagnostics


def plan_release(env,cancelled=None):
    """Search open-finger withdrawal motions using isolated full physics.

    Commands returned here must still be replayed in the real episode. A
    predicted release never counts as a completed demonstration.
    """
    origin=np.asarray(env.jaw_center())
    retreat=[origin[0]-.035,origin[1],origin[2]+.07]
    lowest=max(float(env.bin_position[2]+env.bin_height+.012),float(origin[2]-.05))
    choices=(
        ('legacy_retreat',[(retreat,0.,2.,160)]),
        ('lower_before_retreat', [([origin[0],origin[1],lowest],0.,1.5,12),
            ([origin[0]-.045,origin[1],lowest],0.,1.5,12),
            ([origin[0]-.055,origin[1],origin[2]+.07],0.,2.,160)]),
        ('side_left', [([origin[0],origin[1]+.045,origin[2]],0.,1.5,12),
            ([retreat[0],origin[1]+.045,retreat[2]],0.,2.,160)]),
        ('side_right', [([origin[0],origin[1]-.045,origin[2]],0.,1.5,12),
            ([retreat[0],origin[1]-.045,retreat[2]],0.,2.,160)]),
        ('roll_left', [([*origin],-30.,1.,12),(retreat,0.,2.,160)]),
        ('roll_right', [([*origin],30.,1.,12),(retreat,0.,2.,160)]),
    )
    diagnostics=[];first=None
    for name,targets in choices:
        if cancelled:cancelled()
        branch=fork_physics(env);trajectory=[]
        try:
            for xyz,roll,duration,hold in targets:
                goal=branch.solve_ik(xyz,branch.pending_command[:5],gripper=30.,downward=False)
                goal[4]=np.clip(goal[4]+roll,0.,270.)
                move(branch,goal,duration)
                for _ in range(max(0,hold-12)):branch.step(np.r_[goal,0.,0.,0.])
                trajectory.append((goal,duration,hold))
            if name=='legacy_retreat' and not branch.task_success:
                reference=trajectory[-1][0].copy()
                for roll in (15.,-15.,0.):
                    goal=reference.copy();goal[4]=np.clip(reference[4]+roll,0.,270.)
                    move(branch,goal,1.)
                    for _ in range(28):branch.step(np.r_[goal,0.,0.,0.])
                    trajectory.append((goal,1.,40))
                    if branch.task_success:break
            if first is None:first=trajectory
            diagnostics.append(dict(name=name,predicted_delivered=bool(branch.task_success),error=branch.last_error))
            if branch.task_success:return trajectory,diagnostics
        except ValueError as error:
            diagnostics.append(dict(name=name,predicted_delivered=False,error=str(error)))
        finally:branch.close()
    if first is None:raise ValueError('No feasible release trajectory')
    return first,diagnostics
