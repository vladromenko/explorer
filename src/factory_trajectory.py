"""Retain MoveIt's timed path and stream a smooth look-ahead to factory robotio."""
import math
import hashlib
import json
import numpy as np
from arm_commissioning import HARD_LIMITS
from servo_coordinates import SIGNS
from timed_trajectory import Limits,TimedPath,quintic,derivative,exact_bounds


def _straight_retime(times, positions, velocities, accelerations, limits):
    """Retain a proved straight monotonic path with synchronized jerk timing.

    A sampled TOTG stop has nonzero deceleration at its final point. Forcing
    that acceleration to zero across a tiny last sample creates a numerical
    jerk spike, slowing every segment. A collinear path permits exact scalar
    retiming without changing any of its geometric curves or skipping points.
    General curved and reversing paths retain their original quintics.
    """
    q,v,a=(np.asarray(value,dtype=float) for value in (positions,velocities,accelerations))
    t=np.asarray(times,dtype=float)
    delta=q[-1]-q[0]
    norm=float(delta @ delta)
    if norm<1e-12:return None
    progress=(q-q[0]) @ delta / norm
    if (np.max(np.abs(q-(q[0]+progress[:,None]*delta)))>1e-9
            or np.any(np.diff(progress)<-1e-10)):
        return None
    for values in (v,a):
        projected=(values @ delta / norm)[:,None]*delta
        if np.max(np.abs(values-projected))>1e-9:return None
    for index,dt in enumerate(np.diff(t)):
        coefficients=quintic(q[index],q[index+1],v[index],v[index+1],a[index],a[index+1],dt)
        scalar=derivative(coefficients) @ delta[:,None] / norm
        low,_=exact_bounds(scalar)
        if low[0]<-1e-9:return None
    moving=np.abs(delta)>1e-10
    speed=float(np.min(limits.velocity[moving]/np.abs(delta[moving])))
    acceleration=float(np.min(limits.acceleration[moving]/np.abs(delta[moving])))
    jerk=float(np.min(limits.jerk[moving]/np.abs(delta[moving])))
    tj=min(acceleration/jerk,math.sqrt(speed/jerk))
    ta=max(0.0,speed/(jerk*tj)-tj)
    distance=speed*(2*tj+ta)
    if distance<=1.0:
        cruise=(1.0-distance)/speed
    else:
        cruise=0.0
        tj=acceleration/jerk
        if 1.0>=2*acceleration*tj*tj:
            ta=max(0.0,(-3*tj+math.sqrt(tj*tj+4/acceleration))/2)
        else:
            tj=(1/(2*jerk))**(1/3)
            ta=0.0
    phases=[]
    stamp=position=velocity=accel=0.0
    for length,j in zip((tj,ta,tj,cruise,tj,ta,tj),(jerk,0.0,-jerk,0.0,-jerk,0.0,jerk)):
        if length>1e-12:
            phases.append((stamp,length,position,velocity,accel,j))
            position+=velocity*length+accel*length**2/2+j*length**3/6
            velocity+=accel*length+j*length**2/2
            accel+=j*length
            stamp+=length
    duration=stamp
    def sample(at):
        phase=phases[-1]
        for item in phases:
            if item[0]<=at<item[0]+item[1]:
                phase=item
                break
        beginning,length,p0,v0,a0,j=phase
        dt=min(length,max(0.0,at-beginning))
        return p0+v0*dt+a0*dt**2/2+j*dt**3/6,v0+a0*dt+j*dt**2/2,a0+j*dt
    phase_stamps=[0.0,duration]+[phase[0] for phase in phases]
    stamps=list(phase_stamps)
    source_stamps=[]
    for point in progress:
        lo,hi=0.0,duration
        for _ in range(55):
            mid=(lo+hi)/2
            if sample(mid)[0]<point:lo=mid
            else:hi=mid
        at=(lo+hi)/2
        if abs(point)<1e-10:at=0.0
        if abs(point-1.0)<1e-10:at=duration
        source_stamps.append(at)
        # A source waypoint one microsecond from a jerk switch is still
        # crossed exactly by this curve. It must not create a tiny ill-
        # conditioned Hermite segment beside that exact phase boundary.
        if min(abs(at-knot) for knot in stamps)>=0.001:
            stamps.append(at)
    stamps=sorted(set(round(at,10) for at in stamps))
    if np.min(np.diff(stamps))<0.001:return None
    points=[];rates=[];accelerations_out=[]
    for at in stamps:
        p,rate,acc=sample(at)
        points.append(q[0]+p*delta)
        rates.append(rate*delta)
        accelerations_out.append(acc*delta)
    points[0]=q[0];points[-1]=q[-1]
    rates[0]=rates[-1]=np.zeros(len(delta))
    accelerations_out[0]=accelerations_out[-1]=np.zeros(len(delta))
    return dict(times=stamps,positions=points,velocities=rates,accelerations=accelerations_out,
        source_waypoint_times=source_stamps,algorithm="collinear_synchronized_jerk_limited")

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
    source_duration=float(times[-1]);source_count=len(times)
    retimed=_straight_retime(times,positions,velocities,accelerations,limits) if trajectory is not None else None
    if retimed:
        times=retimed["times"];positions=retimed["positions"]
        velocities=retimed["velocities"];accelerations=retimed["accelerations"]
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
                profile="coordinated_quintic_lookahead",motion=motion,
                source_duration_s=source_duration, timing_limiter=path.timing_limiter,
                source_waypoint_count=source_count,
                retiming_algorithm=retimed["algorithm"] if retimed else "uniform_quintic_time_dilation",
                original_moveit_sha256=hashlib.sha256(json.dumps(trajectory,sort_keys=True,
                    allow_nan=False).encode()).hexdigest() if trajectory is not None else None,
                retained_source_waypoint_times_s=[at*path.scale for at in retimed["source_waypoint_times"]] if retimed else None)
