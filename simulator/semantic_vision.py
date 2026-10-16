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


class SemanticDetector:
    def __init__(self,root,folder,cancelled=lambda:False):
        self.root=Path(root);self.folder=Path(folder);self.cancelled=cancelled

    def detect(self,image,query,frame_id):
        token=str(time.time_ns());folder=self.folder/token;folder.mkdir(parents=True)
        Image.fromarray(np.asarray(image,dtype=np.uint8)).save(folder/'image.png')
        request=dict(query=query,frame_id=frame_id,model=str(self.root/'models/grounding-dino-tiny'))
        (folder/'request.json').write_text(json.dumps(request))
        from .semantic_cache import infer
        try:infer(self.root,folder,cancelled=self.cancelled)
        except (InterruptedError,TimeoutError):raise
        except (OSError,RuntimeError,ValueError) as error:
            raise RuntimeError('Vision backend unavailable: '+str(error)) from error
        result=json.loads((folder/'result.json').read_text())
        if result.get('frame_id')!=frame_id:raise ValueError('Semantic frame mismatch')
        return result['detections']


def vision_python(root):
    internal=Path.home()/'Library/Application Support/ExplorerSimulator/vision-runtime/bin/python'
    return internal if sys.platform=='darwin' and internal.is_file() else Path(root).parent/'env-vision/bin/python'
