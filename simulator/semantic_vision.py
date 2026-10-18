"""Observed RGB-only semantics; detections are hypotheses, not scene labels."""
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def projected_bounds(candidate,frame):
    lo,hi=np.asarray(candidate['bounds_min']),np.asarray(candidate['bounds_max'])
    corners=np.array(list(itertools.product(*zip(lo,hi))))
    transform=np.asarray(frame['optical_to_world']);optical=(corners-transform[:3,3])@transform[:3,:3]
    if np.any(optical[:,2]<=.015):raise ValueError('Candidate intersects camera plane')
    k=frame['intrinsics'];xy=optical[:,:2]/optical[:,2:]
    pixels=xy*np.array([k['fx'],k['fy']])+np.array([k['cx'],k['cy']])
    return np.r_[pixels.min(axis=0),pixels.max(axis=0)]


def associate(detections,candidates,frame):
    matches=[]
    for candidate in candidates:
        try:box=projected_bounds(candidate,frame)
        except (KeyError,ValueError):continue
        area=float(np.prod(np.maximum(box[2:]-box[:2],0.)))
        if area<4:continue
        for detection in detections:
            b=np.asarray(detection.get('bbox'),float)
            if b.shape!=(4,) or not np.isfinite(b).all() or detection.get('confidence',0)<.35:continue
            intersection=float(np.prod(np.maximum(np.minimum(box[2:],b[2:])-np.maximum(box[:2],b[:2]),0.)))
            if intersection/area>=.60:
                matches.append((candidate,detection));break
    if len(matches)!=1:raise ValueError('Смысл запроса и геометрия не совпали однозначно; захват не разрешён')
    candidate,detection=matches[0]
    return dict(candidate,semantic_identity='two_stage_model_hypothesis',
        semantic_evidence=dict(model='GroundingDINO-tiny',detection=detection,
            frame_id=frame['frame_id'],world_geometry_from='registered_depth',identity_proven=False))


def floor_material_candidates(frame,containers):
    """Exclude observed wall bases from floor-cloth crop proposals.

    A close crop can lose the visual context of a tall basket. Its bottom is
    still part of the observed raised extent, rather than a second free item.
    This gate uses RGB-D extents only; object labels/contact truth are absent.
    Unsupported high folds remain unconfirmed instead of being misclassified.
    """
    output=[]
    for candidate in frame.get('low_objects',[]):
        lo,hi=np.asarray(candidate['bounds_min']),np.asarray(candidate['bounds_max'])
        if hi[2]>.06 or hi[2]-lo[2]>.04:continue
        area=float(np.prod(hi[:2]-lo[:2]))
        if area<=0:continue
        excluded=False
        for container in containers:
            a,b=np.asarray(container['bounds_min']),np.asarray(container['bounds_max'])
            if b[2]<.08:continue
            overlap=np.prod(np.maximum(0.,np.minimum(hi[:2],b[:2]+.015)-np.maximum(lo[:2],a[:2]-.015)))
            if overlap/area>=.5:excluded=True;break
        if not excluded:output.append(candidate)
    return output


def detect_metric_crops(detector,image,query,frame,candidates,*,maximum=2):
    """RGB-only grounding at observed component scale, without identity labels.

    Metric candidates propose crops, never semantic answers. The same pinned
    detector and threshold classify the actual pixels; boxes map back to the
    original exposure before the unchanged association gate is applied.
    """
    image=np.asarray(image);h,w=image.shape[:2];results=[];count=0
    for candidate in candidates:
        if count>=maximum:break
        if candidate.get('confidence',0)<.7 or candidate.get('uncertainty_m',1.)>.02:continue
        try:box=projected_bounds(candidate,frame)
        except (KeyError,ValueError):continue
        if not np.isfinite(box).all() or np.any(box[:2]<0) or np.any(box[2:]>[w,h]) or np.any(box[2:]-box[:2]<8):continue
        margin=np.maximum((box[2:]-box[:2])*.15,8.)
        x0,y0=np.maximum(0,np.floor(box[:2]-margin)).astype(int)
        x1,y1=np.minimum([w,h],np.ceil(box[2:]+margin)).astype(int)
        if x1<=x0 or y1<=y0:continue
        token=frame['frame_id']+':metric-crop:'+str(count);count+=1
        for detection in detector.detect(image[y0:y1,x0:x1].copy(),query,token):
            b=np.asarray(detection.get('bbox'),float)
            if b.shape!=(4,) or not np.isfinite(b).all():continue
            b=np.clip(b,[0,0,0,0],[x1-x0,y1-y0,x1-x0,y1-y0])+[x0,y0,x0,y0]
            if np.any(b[2:]<=b[:2]):continue
            results.append(dict(detection,bbox=b.tolist(),source_frame_id=frame['frame_id'],
                crop_bounds=[int(x0),int(y0),int(x1),int(y1)],source='RGB_model_on_observed_metric_crop'))
    return results


def refresh_stationary_detections(reference,current,detections):
    """Associate slow RGB inference with a NEW, matching RGB-D exposure.

    This is not a timestamp refresh. Camera pose, calibration, per-pixel depth
    and appearance must agree. The model exposure and tracking exposure remain
    separately recorded; an unmatched detection supplies no current label.
    """
    a,b=reference['frame'],current['frame']
    if a['frame_id']==b['frame_id'] or a['intrinsics']!=b['intrinsics']:return []
    ta,tb=np.asarray(a['optical_to_world']),np.asarray(b['optical_to_world'])
    if np.linalg.norm(ta[:3,3]-tb[:3,3])>.002 or np.linalg.norm(ta[:3,:3]-tb[:3,:3])>.01:return []
    rgb_a,rgb_b=np.asarray(reference['rgb']),np.asarray(current['rgb'])
    depth_a,depth_b=np.asarray(reference['depth']),np.asarray(current['depth'])
    if rgb_a.shape!=rgb_b.shape or depth_a.shape!=depth_b.shape:return []
    h,w=depth_a.shape;result=[]
    for detection in detections:
        box=np.asarray(detection.get('bbox'),float)
        if box.shape!=(4,) or not np.isfinite(box).all():continue
        x0,y0=np.maximum(0,np.floor(box[:2])).astype(int);x1,y1=np.minimum([w,h],np.ceil(box[2:])).astype(int)
        if x1<=x0 or y1<=y0:continue
        x=depth_a[y0:y1,x0:x1];y=depth_b[y0:y1,x0:x1]
        va=np.isfinite(x)&(x>.03)&(x<3.);vb=np.isfinite(y)&(y>.03)&(y<3.);valid=va&vb
        if valid.sum()<100 or valid.sum()/max(1,(va|vb).sum())<.90:continue
        delta=np.abs(x[valid]-y[valid])
        color=np.abs(rgb_a[y0:y1,x0:x1][valid].astype(float)-rgb_b[y0:y1,x0:x1][valid].astype(float))
        if np.median(delta)>.004 or np.percentile(delta,95)>.008 or color.mean()>8 or np.percentile(color,90)>18:continue
        result.append(dict(detection,model_frame_id=a['frame_id'],tracked_to_frame_id=b['frame_id'],
            association='stationary_registered_RGBD_and_RGB',identity_proven=False))
    return result


class SemanticDetector:
    def __init__(self,root,folder,cancelled=lambda:False):
        self.root=Path(root);self.folder=Path(folder);self.cancelled=cancelled

    def detect(self,image,query,frame_id):
        import shutil
        token=str(time.time_ns());audit=self.folder/token;folder=audit
        # The pinned inference service deliberately accepts files only inside
        # its model storage root. Live mission logs live elsewhere; stage the
        # exact same RGB there and preserve a copy in the mission audit trail.
        if not audit.resolve().is_relative_to(self.root.resolve()):folder=self.root/'requests'/token
        folder.mkdir(parents=True)
        Image.fromarray(np.asarray(image,dtype=np.uint8)).save(folder/'image.png')
        request=dict(query=query,frame_id=frame_id,model=str(self.root/'models/grounding-dino-tiny'))
        (folder/'request.json').write_text(json.dumps(request))
        from .semantic_cache import infer
        try:infer(self.root,folder,cancelled=self.cancelled)
        except (InterruptedError,TimeoutError):raise
        except (OSError,RuntimeError,ValueError) as error:
            raise RuntimeError('Vision backend unavailable: '+str(error)) from error
        finally:
            if folder!=audit:
                audit.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(folder,audit)
        result=json.loads((folder/'result.json').read_text())
        if result.get('frame_id')!=frame_id:raise ValueError('Semantic frame mismatch')
        return result['detections']


def vision_python(root):
    internal=Path.home()/'Library/Application Support/ExplorerSimulator/vision-runtime/bin/python'
    return internal if sys.platform=='darwin' and internal.is_file() else Path(root).parent/'env-vision/bin/python'
