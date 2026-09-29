"""Retain MoveIt's timed path, quantize only at the factory ArmJoints boundary."""
import math
import numpy as np
from arm_commissioning import HARD_LIMITS
from servo_coordinates import SIGNS
from timed_trajectory import Limits,TimedPath

def compile_path(start,goal,trajectory,model):
    if len(start)!=6 or len(goal)!=6:raise ValueError('Six joint positions required')
    a=np.radians(start);b=np.radians(goal)
    if trajectory is None:
        times=[0.,max(.5,float(np.max(np.abs(b-a)))/math.radians(8))]
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
        if max(float(np.max(np.abs(velocities[i]))) for i in (0,-1))>1e-3:
            raise ValueError('MoveIt boundary velocity is not a stationary start/finish')
        for i in (0,-1):
            velocities[i]=np.zeros(6);accelerations[i]=np.zeros(6)
        if abs(a[5]-b[5])>1e-8:
            times.append(times[-1]+max(.5,abs(a[5]-b[5])/math.radians(8)))
            positions.append(b);velocities.append(np.zeros(6));accelerations.append(np.zeros(6))
    def clear(x,y):
        x,y=np.degrees(x),np.degrees(y)
        return all(model.path(x[:5],y[:5],shape)['valid'] for shape in (0.,-.2,-.4,-.6,-.8))
    limits=Limits(np.radians([v[0] for v in HARD_LIMITS]),np.radians([v[1] for v in HARD_LIMITS]),
                  np.full(6,math.radians(10)),np.full(6,math.radians(30)),np.full(6,math.radians(120)))
    path=TimedPath(['servo'+str(i) for i in range(1,7)],times,positions,velocities,accelerations,limits,clear)
    # Continuous servo interpolation is bounded to 200 ms outstanding at a time.
    times=np.linspace(0,path.duration,max(2,math.ceil(path.duration/.2)+1))
    commands=[];previous=list(start)
    for t0,t1 in zip(times,times[1:]):
        pose=np.rint(np.degrees(path.sample(float(t1))['position'])).astype(int).tolist()
        if not clear(np.radians(previous),np.radians(pose)):raise ValueError('Rounded factory path collides')
        commands.append(dict(at=float(t0),end=float(t1),pose=pose,runtime_ms=max(20,round((t1-t0)*1000))))
        previous=pose
    return dict(commands=commands,duration=path.duration,source_sha256=path.source_sha256,
                time_scale=path.scale,full_moveit_path_retained=trajectory is not None)
