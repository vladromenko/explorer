"""Frontier candidates restricted to connected, known, footprint-clear free space."""
import heapq, math
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion, label
from scipy.spatial import ConvexHull

def footprint_kernel(polygon,resolution,yaw,padding=.02):
    points=np.asarray(polygon,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():raise ValueError("Invalid footprint")
    c,s=math.cos(yaw),math.sin(yaw);points=points@np.array([[c,s],[-s,c]])
    half=int(math.ceil((np.max(np.linalg.norm(points,axis=1))+padding)/resolution))+1
    ys,xs=np.mgrid[-half:half+1,-half:half+1];cells=np.column_stack([xs.ravel(),ys.ravel()])*resolution
    equations=ConvexHull(points).equations
    tolerance=padding+resolution*.5*np.abs(equations[:,:2]).sum(axis=1)
    touched=np.all(cells@equations[:,:2].T+equations[:,2]<=tolerance,axis=1)
    return touched.reshape(xs.shape)


def departure_candidate(grid,resolution,origin,pose,footprint,heading,origin_yaw=0.):
    """Find a short known-free forward staging point when the start touches a mapped obstacle.

    This proposes a target only. Nav2's live local collision check and the
    independent controller lidar guard still decide whether movement occurs.
    """
    a=np.asarray(grid,dtype=np.int16)
    if a.ndim!=2 or not a.size or not math.isfinite(resolution) or resolution<=0:return None
    if not all(math.isfinite(v) for v in (*origin,*pose,heading,origin_yaw)):return None
    c,s=math.cos(origin_yaw),math.sin(origin_yaw)
    def cell(x,y):
        dx=x-origin[0];dy=y-origin[1]
        return int(math.floor((c*dx+s*dy)/resolution)),int(math.floor((-s*dx+c*dy)/resolution))
    sx,sy=cell(*pose)
    h,w=a.shape
    if not (0<=sx<w and 0<=sy<h) or a[sy,sx]!=0:return None
    allowed=binary_erosion(a==0,footprint_kernel(footprint,resolution,heading-origin_yaw),border_value=0)
    if allowed[sy,sx]:return None
    forward=(math.cos(heading),math.sin(heading))
    for distance in (.25,.30,.35,.40):
        x=pose[0]+distance*forward[0];y=pose[1]+distance*forward[1]
        tx,ty=cell(x,y)
        if 0<=tx<w and 0<=ty<h and allowed[ty,tx]:
            samples=[cell(pose[0]+q*forward[0],pose[1]+q*forward[1]) for q in np.linspace(0,distance,max(2,int(math.ceil(distance/resolution))*2))]
            if all(0<=ix<w and 0<=iy<h and a[iy,ix]==0 for ix,iy in samples):
                return dict(x=x,y=y,yaw=heading,distance_m=distance,policy="known_free_forward_staging")
    return None


def candidates(grid,resolution,origin,pose,radius=.38,origin_yaw=0.,footprint=None,heading=0.):
    a=np.asarray(grid,dtype=np.int16)
    if a.ndim!=2 or not a.size or a.size>4_000_000 or not math.isfinite(resolution) or resolution<=0:raise ValueError('Invalid occupancy grid')
    if not math.isfinite(origin_yaw):raise ValueError("Invalid map origin yaw")
    cosine=math.cos(origin_yaw);sine=math.sin(origin_yaw)
    dx=pose[0]-origin[0];dy=pose[1]-origin[1]
    local_pose=[cosine*dx+sine*dy,-sine*dx+cosine*dy]
    free=a==0;unknown=a<0
    # Unknown space also blocks the inflated centre path.
    clearance=distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]*resolution
    allowed=free&(clearance>=radius) if footprint is None else binary_erosion(free,footprint_kernel(footprint,resolution,heading-origin_yaw),border_value=0)
    h,w=a.shape;sx=int(math.floor(local_pose[0]/resolution));sy=int(math.floor(local_pose[1]/resolution))
    if not (0<=sx<w and 0<=sy<h) or not allowed[sy,sx]:return []
    distances=np.full(a.shape,np.inf);distances[sy,sx]=0.;queue=[(0.,sy,sx)]
    steps=[(-1,0),(1,0),(0,-1),(0,1)]
    while queue:
        d,y,x=heapq.heappop(queue)
        if d!=distances[y,x]:continue
        for dy,dx in steps:
            ny,nx=y+dy,x+dx
            if 0<=ny<h and 0<=nx<w and allowed[ny,nx] and d+resolution<distances[ny,nx]:
                distances[ny,nx]=d+resolution;heapq.heappush(queue,(d+resolution,ny,nx))
    frontier=free&binary_dilation(unknown)
    groups,count=label(frontier,np.ones((3,3),dtype=int));out=[]
    ys_all,xs_all=np.indices(a.shape)
    # A winding free-cell route must not turn a nearly coincident point into
    # an in-place rotation goal beside an obstacle.
    radial=np.hypot((xs_all+.5)*resolution-local_pose[0],(ys_all+.5)*resolution-local_pose[1])
    reachable=np.isfinite(distances)&(distances>.25)&(radial>=.30)
    for i in range(1,count+1):
        region=groups==i;n=int(region.sum())
        if n*resolution<.3:continue
        to_frontier=distance_transform_edt(~region)*resolution
        approach=reachable&(to_frontier<=max(.9,radius*2))
        if not approach.any():continue
        score=np.where(approach,distances+3.0*to_frontier,np.inf)
        y,x=np.unravel_index(np.argmin(score),a.shape)
        ys,xs=np.nonzero(region);cx=float(xs.mean());cy=float(ys.mean())
        margin=float(clearance[y,x]-radius)
        risk=1/(.05+max(0.,margin))
        utility=float(n*resolution/(1+score[y,x]+.08*risk))
        lx=(x+.5)*resolution;ly=(y+.5)*resolution
        # Mapping keeps the camera facing along the accepted chassis heading;
        # the stopped arm pans toward frontier sides after arrival. A forced
        # turn at a close frontier can sweep the arm into adjacent furniture.
        target_yaw=heading if footprint is not None else math.atan2(cy-y,cx-x)+origin_yaw
        out.append(dict(x=origin[0]+cosine*lx-sine*ly,y=origin[1]+sine*lx+cosine*ly,
                        yaw=target_yaw,path_distance_m=float(distances[y,x]),
                        frontier_cells=n,information_gain_cells=n,clearance_m=float(clearance[y,x]),
                        risk_cost=float(risk),score=utility,
                        policy='risk_adjusted_information_gain'))
    return sorted(out,key=lambda p:-p['score'])[:30]
