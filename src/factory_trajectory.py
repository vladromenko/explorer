"""Retain MoveIt's timed path and stream a smooth look-ahead to factory robotio."""
import math
import numpy as np
from arm_commissioning import HARD_LIMITS
from servo_coordinates import SIGNS
from timed_trajectory import Limits,TimedPath

DEFAULT_MOTION=dict(
    velocity_deg_s=[28,28,28,35,35,40],
    acceleration_deg_s2=[100,100,100,140,140,160],
    jerk_deg_s3=[1000,1000,1000,1400,1400,1800],
    publish_period_s=.08,lookahead_s=.20,min_duration_s=.35)

def motion_profile(value=None):
    value=dict(DEFAULT_MOTION,**(value or {}))
    arrays=[]
    for key in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3'):
        array=np.asarray(value[key],dtype=float)
        if array.shape!=(6,) or not np.isfinite(array).all() or np.any(array<=0):
            raise ValueError('Invalid factory arm '+key)
        arrays.append(array)
    period=float(value['publish_period_s']);lookahead=float(value['lookahead_s']);minimum=float(value['min_duration_s'])
    if not .04<=period<=.15 or not period<=lookahead<=.35 or not .04<=minimum<=1.:
        raise ValueError('Invalid factory arm timing profile')
    return dict(value,velocity_deg_s=arrays[0].tolist(),acceleration_deg_s2=arrays[1].tolist(),
                jerk_deg_s3=arrays[2].tolist(),publish_period_s=period,lookahead_s=lookahead,min_duration_s=minimum)

def compile_path(start,goal,trajectory,model,motion=None):
    motion=motion_profile(motion)
    if len(start)!=6 or len(goal)!=6:raise ValueError('Six joint positions required')
    final_pose=np.rint(goal).astype(int).tolist()
    a=np.radians(start);b=np.radians(goal)
    if trajectory is None:
        # TimedPath expands this minimum duration until the quintic satisfies
        # velocity, acceleration and jerk limits. This gives zero velocity and
        # acceleration at both ends without artificially slow 8 deg/s motion.
        times=[0.,motion['min_duration_s']]
        positions=[a,b];velocities=np.zeros((2,6));accelerations=np.zeros((2,6))
    else:
        names=trajectory['names'];expected=['arm'+str(i)+'_Joint' for i in range(1,6)]
        if len(names)!=5 or set(names)!=set(expected):raise ValueError('Invalid MoveIt joint names')
        order=[names.index(n) for n in expected];times=list(trajectory['times'])
        positions=[];velocities=[];accelerations=[]
        for field,target in [('positions',positions),('velocities',velocities),('accelerations',accelerations)]:
            rows=np.asarray(trajectory[field],dtype=float)
            if rows.shape!=(len(times),5):raise ValueError('Incomplete MoveIt timed path')
            values=rows[:,order]*SIGNS
            if field=='positions':values+=math.pi/2
            target.extend(np.column_stack([values,np.full(len(times),a[5] if field=='positions' else 0.)]))
        if not np.allclose(positions[0],a,atol=math.radians(.1)) or not np.allclose(positions[-1][:5],b[:5],atol=math.radians(.1)):
            raise ValueError('MoveIt endpoints disagree with current pose')
        # TOTG may report a tiny nonzero first velocity and nonzero endpoint
        # acceleration. Blend the two boundary segments from/to rest. Interior
        # derivatives and every waypoint remain; TimedPath rechecks the new
        # polynomial hull, dynamics and collisions before any publication.
        # Jazzy TOTG getVelocity(0) reports the end of its first 1 ms
        # integration step, not velocity at t=0. Accept only that bounded
        # numerical artifact; a materially moving boundary remains invalid.
        first=np.asarray(velocities[0]);first_acc=np.asarray(accelerations[0])
        artifact=(np.max(np.abs(first))<=.005 and
            np.allclose(first,first_acc*.001,rtol=.05,atol=1e-6))
        if ((np.max(np.abs(first))>1e-3 and not artifact) or
                np.max(np.abs(velocities[-1]))>1e-3):
            raise ValueError("MoveIt boundary velocity is not a stationary start/finish")
        for i in (0,-1):
            velocities[i]=np.zeros(6);accelerations[i]=np.zeros(6)
        if abs(a[5]-b[5])>1e-8:
            times.append(times[-1]+max(.5,abs(a[5]-b[5])/math.radians(8)))
            positions.append(b);velocities.append(np.zeros(6));accelerations.append(np.zeros(6))
    def clear(x,y):
        x,y=np.degrees(x),np.degrees(y)
        return all(model.path(x[:5],y[:5],shape)['valid'] for shape in (0.,-.2,-.4,-.6,-.8))
    limits=Limits(np.radians([v[0] for v in HARD_LIMITS]),np.radians([v[1] for v in HARD_LIMITS]),
                  np.radians(motion['velocity_deg_s']),np.radians(motion['acceleration_deg_s2']),
                  np.radians(motion['jerk_deg_s3']))
    path=TimedPath(['servo'+str(i) for i in range(1,7)],times,positions,velocities,accelerations,limits,clear)
    # robotio exposes integer-degree targets only. Send a receding target ahead
    # of the desired state, before the preceding finite interpolation expires.
    # The overlap avoids the stop/start seam that made the arm visibly twitch.
    period=motion['publish_period_s'];lookahead=motion['lookahead_s']
    send_times=np.arange(0.,max(0.,path.duration-lookahead)+period*.5,period).tolist()
    send_times.append(max(0.,path.duration-lookahead))
    commands=[];previous=list(start)
    for at in sorted(set(round(float(t),9) for t in send_times)):
        target=min(path.duration,at+lookahead)
        pose=np.rint(np.degrees(path.sample(target)['position'])).astype(int).tolist()
        if pose==previous:continue
        if not clear(np.radians(previous),np.radians(pose)):raise ValueError('Rounded factory path collides')
        commands.append(dict(at=at,end=target,pose=pose,runtime_ms=max(20,round((target-at)*1000))))
        previous=pose
    if not commands or commands[-1]['pose']!=final_pose:
        at=max(0.,path.duration-lookahead)
        commands.append(dict(at=at,end=path.duration,pose=final_pose,runtime_ms=max(20,round((path.duration-at)*1000))))
    return dict(commands=commands,duration=path.duration,source_sha256=path.source_sha256,
                time_scale=path.scale,full_moveit_path_retained=trajectory is not None,
                profile='coordinated_quintic_lookahead',motion=motion)
