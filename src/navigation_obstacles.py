"""Directional swept-envelope check using real scans and accepted lidar mounts."""
import math
import numpy as np


def project_scan(message,frames):
    frame=message.header.frame_id
    if frame not in frames:raise ValueError("Unaccepted lidar frame")
    offset=frames[frame];angles=message.angle_min+np.arange(len(message.ranges))*message.angle_increment
    ranges=np.asarray(message.ranges,dtype=float)
    valid=np.isfinite(ranges)&(ranges>message.range_min)&(ranges<message.range_max)
    return np.column_stack([offset[0]+ranges[valid]*np.cos(angles[valid]),offset[1]+ranges[valid]*np.sin(angles[valid])])


def swept_obstacle(points,velocity,polygon,horizon=.8,margin=.04):
    if len(points)<10 or len(velocity)!=3 or not np.isfinite(velocity).all():return True
    if max(abs(value) for value in velocity)<1e-5:return False
    a=np.asarray(points,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():return True
    # Only the rigid chassis volume is self-filtered, not the arm reach envelope.
    outside=(np.abs(a[:,0])>.153)|(np.abs(a[:,1])>.111)
    a=a[outside]
    lower=np.min(polygon,axis=0);upper=np.max(polygon,axis=0)
    def distance(local):return np.linalg.norm(np.maximum(np.maximum(lower-local,local-upper),0.),axis=1)
    initial=distance(a)
    if np.any(initial<1e-6):return True
    vx,vy,wz=velocity
    for t in np.linspace(0,horizon,9):
        angle=wz*t;c=math.cos(angle);s=math.sin(angle)
        if abs(wz)<1e-6:dx,dy=vx*t,vy*t
        else:dx=(vx*s-vy*(1-c))/wz;dy=(vx*(1-c)+vy*s)/wz
        delta=a-[dx,dy];local=delta@np.array([[c,-s],[s,c]])
        gap=distance(local)
        # A return already in the extra margin may be passed or moved away
        # from, never approached. The hard accepted footprint remains blocked.
        collision=(gap<1e-6)|((initial>margin)&(gap<=margin))|((initial<=margin)&(gap<initial-1e-6))
        if np.any(collision):return True
    return False
