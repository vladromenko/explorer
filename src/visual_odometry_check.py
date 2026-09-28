"""Independent RGB-D PnP displacement check; never changes motion calibration."""
import cv2
import numpy as np

def compare(a,b):
    orb=cv2.ORB_create(nfeatures=2500)
    ka,da=orb.detectAndCompute(a['rgb'],None);kb,db=orb.detectAndCompute(b['rgb'],None)
    if da is None or db is None:return dict(outcome='unknown',reason='No visual features')
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    xyz=[];uv=[]
    h,w=a['depth'].shape
    for pair in matches:
        if len(pair)==2 and pair[0].distance<.7*pair[1].distance:
            m=pair[0];u,v=ka[m.queryIdx].pt;x,y=int(u),int(v)
            patch=a['depth'][max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]
            valid=patch[np.isfinite(patch)&(patch>.15)&(patch<4)]
            if len(valid)>=5 and np.ptp(valid)<.03:
                z=float(np.median(valid))
                ray=cv2.undistortPoints(np.array([[[u,v]]],dtype=float),a['k'],a['d'])[0,0]
                xyz.append([ray[0]*z,ray[1]*z,z]);uv.append(kb[m.trainIdx].pt)
    if len(xyz)<20:return dict(outcome='unknown',reason='Insufficient depth-supported matches',matches=len(xyz))
    xyz=np.asarray(xyz,dtype=float);uv=np.asarray(uv,dtype=float)
    ok,r,t,inliers=cv2.solvePnPRansac(xyz,uv,b['k'],b['d'],iterationsCount=200,
                                     reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<15:return dict(outcome='unknown',reason='PnP consensus failed')
    ids=inliers[:,0];r,t=cv2.solvePnPRefineLM(xyz[ids],uv[ids],b['k'],b['d'],r,t)
    projected,_=cv2.projectPoints(xyz[ids],r,t,b['k'],b['d'])
    errors=np.linalg.norm(projected[:,0]-uv[ids],axis=1);rotation=cv2.Rodrigues(r)[0]
    position=(-rotation.T@t).ravel()
    return dict(outcome='estimate',camera_displacement_m=position.tolist(),distance_m=float(np.linalg.norm(t)),
                rotation_rad=float(np.linalg.norm(r)),matches=len(xyz),inliers=len(ids),
                median_reprojection_px=float(np.median(errors)),metric_ground_truth=False,
                camera_lever_arm_compensated=False)
