"""Separate a small object's depth from a surrounding planar surface.

Camera-frame hypothesis only: a fitted plane is not automatically the floor,
and a connected depth component is not a verified semantic object or grasp.
"""
import cv2
import numpy as np

def locate(sample,box):
    def unknown(reason):return dict(outcome='unknown',reason=reason,position=None)
    b=np.asarray(box,dtype=float);depth=np.asarray(sample['depth']);h,w=depth.shape
    if b.shape!=(4,) or not np.isfinite(b).all():return unknown('INVALID_BOX')
    x1,y1,x2,y2=np.rint(b).astype(int)
    if not (0<=x1<x2<=w and 0<=y1<y2<=h):return unknown('INVALID_BOX')
    margin=max(12,min(60,int(max(x2-x1,y2-y1)*.3)))
    xa,ya,xb,yb=max(0,x1-margin),max(0,y1-margin),min(w,x2+margin),min(h,y2+margin)
    v,u=np.mgrid[ya:yb,xa:xb];z=depth[ya:yb,xa:xb]
    valid=np.isfinite(z)&(z>.15)&(z<2.)
    inside=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
    ring=valid&~inside
    if np.count_nonzero(ring)<80:return unknown('INSUFFICIENT_SURROUNDING_DEPTH')
    k=np.asarray(sample['k'],dtype=float);d=np.asarray(sample['d'],dtype=float)
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        return unknown('INVALID_INTRINSICS')
    rays=cv2.undistortPoints(np.stack([u,v],axis=-1).astype(float).reshape(-1,1,2),k,d).reshape(*z.shape,2)
    xyz=np.concatenate([rays*z[:,:,None],z[:,:,None]],axis=-1)
    cloud=xyz[ring];cloud=cloud[::max(1,len(cloud)//2500)]
    rng=np.random.default_rng(17);best=np.zeros(len(cloud),dtype=bool)
    for _ in range(100):
        a,bp,c=cloud[rng.choice(len(cloud),3,replace=False)]
        n=np.cross(bp-a,c-a);norm=np.linalg.norm(n)
        if norm>1e-7:
            n/=norm;support=np.abs((cloud-a)@n)<.004
            if support.sum()>best.sum():best=support
    if best.mean()<.7:return unknown('SURROUNDING_SURFACE_NOT_PLANAR')
    center=np.mean(cloud[best],axis=0)
    _,_,vt=np.linalg.svd(cloud[best]-center,full_matrices=False);normal=vt[-1]
    if np.dot(normal,center)>0:normal=-normal
    offset=-float(np.dot(normal,center))
    residual=(cloud@normal)+offset
    noise=float(np.median(np.abs(residual[best]))*1.4826)
    threshold=max(.006,noise*4)
    height=xyz@normal+offset
    foreground=(valid&inside&(height>threshold)&(height<.25)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(foreground,connectivity=8)
    sizes=sorted(((int(stats[i,cv2.CC_STAT_AREA]),i) for i in range(1,count)),reverse=True)
    evidence=dict(method='surrounding_plane_ransac_connected_depth',
                  plane_camera=[*normal.tolist(),offset],plane_inlier_fraction=float(best.mean()),
                  plane_noise_m=noise,min_separation_m=threshold,
                  support_plane_semantics_verified=False,identity_verified=False)
    if not sizes or sizes[0][0]<max(20,.03*(x2-x1)*(y2-y1)):
        return dict(unknown('NO_RESOLVED_FOREGROUND'),evidence=evidence)
    if len(sizes)>1 and sizes[1][0]>.5*sizes[0][0]:
        return dict(unknown('AMBIGUOUS_COMPONENTS'),evidence=evidence)
    selected=labels==sizes[0][1];points=xyz[selected];position=np.median(points,axis=0)
    evidence.update(pixels=int(selected.sum()),median_surface_height_m=float(np.median(height[selected])),
                    image_centroid_px=[float(np.median(u[selected])),float(np.median(v[selected]))])
    return dict(outcome='candidate',evidence=evidence,position=dict(
        x=float(position[0]),y=float(position[1]),z=float(position[2]),frame=str(sample['frame']),
        depth_spread_m=float(np.percentile(points[:,2],90)-np.percentile(points[:,2],10)),
        depth_is_object_verified=False,foreground_separated=True))
