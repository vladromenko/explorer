"""Frontier candidates restricted to connected, known, footprint-clear free space."""
import heapq, math
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, label

def candidates(grid,resolution,origin,pose,radius=.38):
    a=np.asarray(grid,dtype=np.int16)
    if a.ndim!=2 or not a.size or a.size>4_000_000 or not math.isfinite(resolution) or resolution<=0:raise ValueError('Invalid occupancy grid')
    free=a==0;unknown=a<0
    # Unknown space also blocks the inflated centre path.
    clearance=distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]*resolution
    allowed=free&(clearance>=radius)
    h,w=a.shape;sx=int(math.floor((pose[0]-origin[0])/resolution));sy=int(math.floor((pose[1]-origin[1])/resolution))
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
    reachable=np.isfinite(distances)&(distances>.25)
    for i in range(1,count+1):
        region=groups==i;n=int(region.sum())
        if n*resolution<.3:continue
        to_frontier=distance_transform_edt(~region)*resolution
        approach=reachable&(to_frontier<=max(.9,radius*2))
        if not approach.any():continue
        score=np.where(approach,distances+3.0*to_frontier,np.inf)
        y,x=np.unravel_index(np.argmin(score),a.shape)
        ys,xs=np.nonzero(region);cx=float(xs.mean());cy=float(ys.mean())
        out.append(dict(x=origin[0]+(x+.5)*resolution,y=origin[1]+(y+.5)*resolution,
                        yaw=math.atan2(cy-y,cx-x),path_distance_m=float(distances[y,x]),
                        frontier_cells=n,score=float(n*resolution/(1+score[y,x]))))
    return sorted(out,key=lambda p:-p['score'])[:30]
