"""Fresh depth/feature-associated semantic hypotheses for display and recording.

Recognition names never constitute verified identity or permission to move.
Lost targets cannot inherit a new object's label without a new recognition.
"""
import time
import cv2
import numpy as np
from delivery_vision import FeatureObject, reacquire_candidate


class StationaryDepthTarget:
    """Re-observe a texture-poor target only while the observer remains settled."""
    def __init__(self, sample, box, geometry=None):
        if sample.get("observer_key") is None:
            raise ValueError("Texture-poor target requires a stationary observer")
        self.sample, self.box = sample, list(box)
        self.object_id = "depth-" + str(time.time_ns())
        if geometry is None:
            from surface_foreground import locate
            geometry=locate(sample,box)
        evidence=geometry.get("depth_evidence",geometry.get("evidence"))
        if evidence is None:
            raise ValueError("Stationary depth target lacks a separated support plane")
        self.plane=np.array(evidence["plane_camera"])
        self.separation=evidence["min_separation_m"]
        x1,y1,x2,y2=np.rint(box).astype(int)
        h,w=sample["depth"].shape
        self.region=(max(0,x1-12),max(0,y1-12),min(w,x2+12),min(h,y2+12))
        a,b,c,d=self.region
        v,u=np.mgrid[b:d,a:c]
        self.inside=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
        self.rays=cv2.undistortPoints(np.stack([u,v],axis=-1).astype(float).reshape(-1,1,2),
            sample["k"],sample["d"]).reshape(*u.shape,2)
        self.mask,self.point=self.segment(sample)
        self.gray=cv2.cvtColor(sample["rgb"],cv2.COLOR_BGR2GRAY)[y1:y2,x1:x2].astype(float).ravel()
        if np.std(self.gray)<8:
            raise ValueError("Stationary target appearance lacks contrast")

    def segment(self,sample):
        a,b,c,d=self.region
        z=sample["depth"][b:d,a:c]
        valid=np.isfinite(z)&(z>.15)&(z<2)
        xyz=np.concatenate([self.rays*z[:,:,None],z[:,:,None]],axis=-1)
        height=xyz@self.plane[:3]+self.plane[3]
        ring=valid&~self.inside
        if ring.sum()<80 or np.mean(np.abs(height[ring])<.004)<.7:
            raise ValueError("Stationary target support plane changed")
        mask=valid&self.inside&(height>self.separation)&(height<.25)
        if mask.sum()<20:
            raise ValueError("No separated object in stationary target region")
        return mask,np.median(xyz[mask],axis=0)

    def update(self, sample):
        if not float(self.sample["stamp"]) < float(sample["stamp"]) <= float(self.sample["stamp"]) + .4:
            raise ValueError("Stationary depth target frame gap")
        if sample.get("observer_key") != self.sample.get("observer_key"):
            raise ValueError("Observer moved; texture-poor target needs new recognition")
        mask,point=self.segment(sample)
        if np.count_nonzero(mask&self.mask)/max(1,np.count_nonzero(mask|self.mask))<.7:
            raise ValueError("Stationary object depth silhouette changed")
        if np.linalg.norm(point-self.point)>.01:
            raise ValueError("Stationary object moved")
        x1,y1,x2,y2=np.rint(self.box).astype(int)
        gray=cv2.cvtColor(sample["rgb"],cv2.COLOR_BGR2GRAY)[y1:y2,x1:x2].astype(float).ravel()
        correlation=float(np.corrcoef(self.gray,gray)[0,1]) if np.std(gray)>=8 else 0.
        if not np.isfinite(correlation) or correlation<.92:
            raise ValueError("Stationary object appearance changed")
        self.sample = sample
        self.mask,self.point,self.gray=mask,point,gray
        return {"point_camera": point,
            "association_fraction": None, "object_position_uncertainty_m": None,
            "association_method": "stationary_RGBD_dense_appearance",
            "appearance_correlation": correlation}


class GroundedTargets:
    def __init__(self):
        self.targets = []
        self.request_id = None
        self.deadline = 0.
        self.phase = "idle"
        self.reason = None
        self.calibration = None

    def clear(self, phase="idle", reason=None):
        self.targets = []
        self.phase, self.reason = phase, reason

    def install(self, identifier, reference, sample, objects, now=None):
        self.clear()
        self.request_id = identifier
        self.calibration = (np.array(sample["k"]), np.array(sample["d"]))
        self.deadline = (time.time() if now is None else now) + 120
        for item in objects:
            if item.get("position") is not None and item.get("confidence", 0) >= .35:
                try:
                    association = reacquire_candidate(reference, sample, item["bbox"])
                    try:
                        tracker = FeatureObject(sample, association["bbox"])
                    except ValueError as exc:
                        if "insufficient visible tracking texture" not in str(exc):
                            raise
                        tracker = StationaryDepthTarget(sample, association["bbox"],association)
                    self.targets.append({"tracker": tracker, "box": np.array(item["bbox"], dtype=float),
                        "label": item.get("requested_label", item["label"]),
                        "model_label": item["label"], "confidence": item["confidence"],
                        "recognition_stamp": float(reference["stamp"])})
                except ValueError as exc:
                    self.reason = str(exc)
        self.phase = "tracking" if self.targets else "unavailable"

    def update(self, sample, now=None):
        now = time.time() if now is None else now
        if self.targets and now >= self.deadline:
            self.clear("expired")
        if self.targets and (not np.array_equal(sample["k"], self.calibration[0])
                or not np.array_equal(sample["d"], self.calibration[1])):
            self.clear("lost", "Camera calibration changed after recognition")
        rows, surviving = [], []
        for item in self.targets:
            tracker = item["tracker"]
            previous = None if isinstance(tracker,StationaryDepthTarget) else np.median(tracker.points[:, 0], axis=0)
            try:
                observation = tracker.update(sample)
                if previous is not None:
                    item["box"] += np.tile(np.median(tracker.points[:, 0], axis=0) - previous, 2)
                h, w = sample["rgb"].shape[:2]
                box = item["box"]
                if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                    raise ValueError("Recognized target left the camera image")
                point = observation["point_camera"]
                rows.append({"label": item["label"], "model_label": item["model_label"],
                    "confidence": item["confidence"], "bbox": box.tolist(),
                    "track": tracker.object_id, "image_stamp": float(sample["stamp"]),
                    "recognition_stamp": item["recognition_stamp"], "request_id": self.request_id,
                    "position": {"x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
                        "frame": str(sample["frame"])},
                    "association_fraction": observation["association_fraction"],
                    "position_uncertainty_m": observation["object_position_uncertainty_m"],
                    "label_source": "GroundingDINO_hypothesis",
                    "association_method": observation.get("association_method","reciprocal_RGBD_features"),
                    "identity_verified": False, "motion_authorized": False})
                surviving.append(item)
            except ValueError as exc:
                self.reason = str(exc)
        self.targets = surviving
        if self.phase == "tracking" and not surviving:
            self.phase = "lost"
        return {"at": now, "image_stamp": float(sample["stamp"]), "frame": str(sample["frame"]),
            "phase": self.phase, "request_id": self.request_id, "reason": self.reason,
            "objects": rows, "identity_verified": False, "motion_authorized": False}
