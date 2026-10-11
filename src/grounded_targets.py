"""Fresh depth/feature-associated semantic hypotheses for display and recording.

Recognition names never constitute verified identity or permission to move.
Lost targets cannot inherit a new object's label without a new recognition.
"""
import time
import numpy as np
from delivery_vision import FeatureObject, reacquire_candidate


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
                    tracker = FeatureObject(sample, association["bbox"])
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
            previous = np.median(tracker.points[:, 0], axis=0)
            try:
                observation = tracker.update(sample)
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
                    "label_source": "GroundingDINO_hypothesis_reciprocal_RGBD_tracking",
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
