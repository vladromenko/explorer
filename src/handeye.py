"""RGB-D natural-feature hand-eye fit with held-out poses; no actuator access."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def visual_pose(reference, current):
    sift=cv2.SIFT_create(nfeatures=1800)
    ka,da=sift.detectAndCompute(reference['rgb'],None)
    kb,db=sift.detectAndCompute(current['rgb'],None)
    if da is None or db is None:raise ValueError('Insufficient visual texture')
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    pairs=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
    objects=[];pixels=[]
    for m in pairs:
        u,v=ka[m.queryIdx].pt;x,y=int(round(u)),int(round(v))
        depth=reference['depth'][max(0,y-1):y+2,max(0,x-1):x+2]
        good=depth[np.isfinite(depth)&(depth>.15)&(depth<1.5)]
        if len(good)>=5 and np.ptp(good)<.025:
            uv=cv2.undistortPoints(np.array([[[u,v]]],dtype=float),reference['k'],reference['d'])[0,0]
            z=float(np.median(good));objects.append([uv[0]*z,uv[1]*z,z]);pixels.append(kb[m.trainIdx].pt)
    if len(objects)<40:raise ValueError('Fewer than 40 valid RGB-D matches')
    objects=np.asarray(objects);pixels=np.asarray(pixels)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(objects,pixels,current['k'],current['d'],
                                         iterationsCount=300,reprojectionError=2.,confidence=.999)
    if not ok or inliers is None or len(inliers)<35:raise ValueError('Unreliable visual pose')
    idx=inliers[:,0]
    rvec,tvec=cv2.solvePnPRefineLM(objects[idx],pixels[idx],current['k'],current['d'],rvec,tvec)
    projected=cv2.projectPoints(objects[idx],rvec,tvec,current['k'],current['d'])[0][:,0]
    error=np.linalg.norm(projected-pixels[idx],axis=1)
    if np.median(error)>1. or np.percentile(error,95)>2.5:raise ValueError('High visual reprojection residual')
    transform=np.eye(4);transform[:3,:3]=cv2.Rodrigues(rvec)[0];transform[:3,3]=tvec[:,0]
    predicted=(objects[idx]@transform[:3,:3].T+transform[:3,3])[:,2]
    observed=[]
    for u,v in pixels[idx]:
        x,y=int(round(u)),int(round(v));observed.append(current['depth'][y,x])
    observed=np.asarray(observed);valid=np.isfinite(observed)&(observed>.15)
    depth_error=np.abs(predicted[valid]-observed[valid])
    if len(depth_error)<25 or np.median(depth_error)>.015:raise ValueError('RGB pose disagrees with independent depth')
    coverage=np.prod(np.ptp(pixels[idx],axis=0))/np.prod(current['rgb'].shape[:2])
    if coverage<.08:raise ValueError('Matches cover too little of the image')
    return transform,dict(inliers=len(idx),reprojection_median_px=float(np.median(error)),
                          depth_median_error_m=float(np.median(depth_error)),coverage=float(coverage))


def mount_poses(samples):
    explicit=['base_mount' in sample for sample in samples]
    if any(explicit):
        if not all(explicit) or any(str(sample.get('mount_frame'))!='arm4' for sample in samples):
            raise ValueError('Mixed or unknown hand-eye mount frames')
        return [sample['base_mount'] for sample in samples],'arm4'
    return [sample['base_tool'] for sample in samples],'Gripping with servo5 held at 90 degrees'


def fit(paths):
    samples=[dict(np.load(p,allow_pickle=False)) for p in paths]
    if len(samples)<8:raise ValueError('At least eight independently observed arm poses required')
    base=np.array([s['base_pose'] for s in samples])
    if np.max(np.abs(base-base[0]))>.005:raise ValueError('Base moved during collection')
    visual=[np.eye(4)];metrics=[dict(reference=True)]
    for i,sample in enumerate(samples[1:],1):
        result=None
        references=[0]+list(range(i-1,max(0,i-4),-1))
        for j in references:
            if result is None:
                try:
                    pose,quality=visual_pose(samples[j],sample)
                    result=pose@visual[j],dict(quality,reference_index=j)
                except ValueError:pass
        if result is None:raise ValueError(f'No reliable visual overlap for sample {i}')
        visual.append(result[0]);metrics.append(result[1])
    tools,mount_frame=mount_poses(samples)
    rotation_axes=np.array([Rotation.from_matrix(tools[0][:3,:3].T@t[:3,:3]).as_rotvec() for t in tools[1:-2]])
    singular=np.linalg.svd(rotation_axes,compute_uv=False)
    if singular[1]<.15:raise ValueError('Insufficient independent rotation axes')
    def solve(method):
        r,t=cv2.calibrateHandEye([p[:3,:3] for p in tools[:-2]],[p[:3,3] for p in tools[:-2]],
                                [p[:3,:3] for p in visual[:-2]],[p[:3,3] for p in visual[:-2]],method=method)
        result=np.eye(4);result[:3,:3]=r;result[:3,3]=t[:,0]
        if not np.all(np.isfinite(result)):raise ValueError('Degenerate hand-eye fit')
        return result
    result=solve(cv2.CALIB_HAND_EYE_PARK);second=solve(cv2.CALIB_HAND_EYE_TSAI)
    disagreement=np.linalg.inv(result)@second
    if np.linalg.norm(disagreement[:3,3])>.025 or Rotation.from_matrix(disagreement[:3,:3]).magnitude()>.05:
        raise ValueError('Independent hand-eye methods disagree')
    world=[g@result@v for g,v in zip(tools,visual)]
    errors=[np.linalg.inv(world[0])@w for w in world]
    translation=[float(np.linalg.norm(e[:3,3])) for e in errors]
    angle=[float(np.degrees(Rotation.from_matrix(e[:3,:3]).magnitude())) for e in errors]
    accepted=max(translation)<.01 and max(angle)<2 and np.linalg.norm(result[:3,3])<.25
    return dict(camera_to_mount_reference=result.tolist(),reference_mount=mount_frame,
                samples=[str(p) for p in paths],visual_quality=metrics,
                translation_residual_m=translation,rotation_residual_deg=angle,
                held_out_indices=[len(samples)-2,len(samples)-1],consistent=accepted,
                measured_joint_positions=False,execution_authorized=False,
                note='Command-estimated joint poses; fit consistency is not independent absolute joint calibration')


if __name__=='__main__':
    root=Path('/home/vlad/Explorer/data')
    paths=sorted((root/'handeye').glob('*.npz'))
    try:report=fit(paths)
    except (ValueError,cv2.error) as exc:report=dict(consistent=False,error=str(exc),execution_authorized=False)
    (root/'handeye-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
