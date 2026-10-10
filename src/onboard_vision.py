"""Read-only fresh camera tools and a fast selected-ROI tracker; no camera owner."""
from collections import deque
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np


def deproject(depth, intrinsic, distortion, stride=8):
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or not 1 <= stride <= 32:
        raise ValueError("Invalid depth image or sampling stride")
    k = np.asarray(intrinsic, dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Camera intrinsics unavailable")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > .15) & (z < 5.)
    pixels = np.column_stack([u[valid], v[valid]]).astype(float)
    if len(pixels) == 0:
        raise ValueError("No valid measured depth")
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, np.asarray(distortion, dtype=float))[:, 0]
    points = np.column_stack([rays * z[valid, None], z[valid]])
    return points, pixels.astype(int)


def fit_plane(points):
    cloud = np.asarray(points, dtype=float)
    cloud = cloud[::max(1, len(cloud) // 2000)]
    if len(cloud) < 80 or not np.isfinite(cloud).all():
        raise ValueError("Insufficient measured points for a plane")
    rng = np.random.default_rng(17)
    best = np.zeros(len(cloud), dtype=bool)
    for _ in range(80):
        a, b, c = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            mask = np.abs((cloud-a) @ (normal/norm)) < .008
            if mask.sum() > best.sum():
                best = mask
    if best.mean() < .5:
        raise ValueError("No dominant measured plane")
    center = cloud[best].mean(axis=0)
    _, _, directions = np.linalg.svd(cloud[best]-center, full_matrices=False)
    normal = directions[-1]
    if normal @ center > 0:
        normal = -normal
    return {"coefficients_camera": [*normal.tolist(), -float(normal @ center)],
            "inlier_fraction": float(best.mean()), "surface_semantics": "unknown", "is_floor": False}


def color_objects(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    ranges = {"red": [((0, 100, 65), (10, 255, 255)), ((170, 100, 65), (179, 255, 255))],
              "blue": [((95, 85, 55), (130, 255, 255))],
              "green": [((35, 70, 45), (85, 255, 255))],
              "yellow": [((18, 95, 75), (34, 255, 255))]}
    rows = []
    for color, bands in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in bands:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(contour)
            if area >= 100:
                x, y, w, h = cv2.boundingRect(contour)
                rect = cv2.minAreaRect(contour)
                perimeter = cv2.arcLength(contour, True)
                vertices = len(cv2.approxPolyDP(contour, .04*perimeter, True))
                shape = "triangle" if vertices == 3 else "quadrilateral" if vertices == 4 else "round" if area/max(perimeter*perimeter, 1) > .065 else "irregular"
                rows.append({"color": color, "bbox": [x, y, x+w, y+h], "shape_2d": shape,
                             "orientation_image_deg": float(rect[2]), "oriented_box_px": cv2.boxPoints(rect).tolist(),
                             "area_px": float(area), "semantic_identity_verified": False})
    return rows


def tag_objects(rgb):
    if not hasattr(cv2, "aruco"):
        raise ValueError("Installed OpenCV lacks the aruco module")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, identifiers, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
    else:
        corners, identifiers, _ = cv2.aruco.detectMarkers(gray, dictionary)
    return [] if identifiers is None else [{"tag_id": int(identifier), "family": "AprilTag36h11",
        "corners_px": corners[index].reshape(4, 2).tolist(), "pose_measured": False}
        for index, identifier in enumerate(identifiers.flatten())]


class OpticalTarget:
    def __init__(self, rgb, bbox, stamp):
        h, w = rgb.shape[:2]
        box = np.asarray(bbox, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Select a finite ROI box")
        x1, y1, x2, y2 = np.rint(box).astype(int)
        if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h or min(x2-x1, y2-y1) < 12:
            raise ValueError("ROI outside displayed frame or too small")
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        self.points = cv2.goodFeaturesToTrack(self.gray, 80, .015, 3, mask=mask)
        if self.points is None or len(self.points) < 8:
            raise ValueError("Selected region lacks texture; choose object edges")
        self.initial = len(self.points)
        self.stamp = stamp
        self.identifier = uuid.uuid4().hex
        self.box = box
        self.phase = "tracking"

    def update(self, rgb, stamp):
        if self.phase != "tracking":
            raise ValueError("Target association was lost; select it again")
        if not self.stamp < stamp <= self.stamp+.5:
            self.phase = "lost"
            raise ValueError("Camera gap; blind tracking cancelled")
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        nxt, valid, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None)
        if nxt is None:
            self.phase = "lost"
            raise ValueError("Target disappeared")
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None)
        if back is None:
            self.phase = "lost"
            raise ValueError("Target reciprocal association unavailable")
        good = (valid[:, 0] > 0) & (reverse[:, 0] > 0) & (np.linalg.norm(back[:, 0]-self.points[:, 0], axis=1) < .8)
        good &= (nxt[:, 0, 0] >= 0) & (nxt[:, 0, 0] < gray.shape[1]) & (nxt[:, 0, 1] >= 0) & (nxt[:, 0, 1] < gray.shape[0])
        if good.sum() < 8 or good.sum()/self.initial < .65:
            self.phase = "lost"
            raise ValueError("Target occluded or correspondence ambiguous")
        delta = np.median(nxt[good, 0]-self.points[good, 0], axis=0)
        self.box += np.tile(delta, 2)
        self.points, self.gray, self.stamp = nxt[good], gray, stamp
        center = (self.box[:2]+self.box[2:])/2
        return {"target_id": self.identifier, "phase": self.phase, "bbox": self.box.tolist(),
                "image_stamp": stamp, "center_px": center.tolist(), "image_size": [gray.shape[1], gray.shape[0]],
                "association_fraction": float(good.sum()/self.initial), "depth_required": False, "identity_kind": "operator_selected_2d_roi"}


class OnboardVision:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.frames = deque(maxlen=8)
        self.target = None
        self.state = {"phase": "idle"}

    def frame(self):
        directory = self.root / "data"
        metadata = json.loads((directory / "camera-frame.json").read_text())
        stamp = metadata.get("image_stamp", 0)
        if not 0 <= time.time()-stamp <= .7:
            raise ValueError("Onboard camera frame is stale")
        jpeg = (directory / "frame-raw.jpg").read_bytes()
        if metadata.get("jpeg_sha256") != hashlib.sha256(jpeg).hexdigest():
            raise ValueError("Camera frame changed during reading; retry")
        rgb = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError("Invalid camera JPEG")
        return rgb, jpeg, dict(metadata,same_frame_hash_verified=True)

    def capture(self):
        rgb, jpeg, metadata = self.frame()
        identifier = metadata["frame_id"]
        with self.lock:
            self.frames.append({"id": identifier, "stamp": metadata["image_stamp"], "rgb": rgb, "metadata": metadata})
        return identifier, jpeg

    def select(self, identifier, bbox):
        with self.lock:
            frame = next((item for item in self.frames if item["id"] == identifier), None)
            if frame is None or not 0 <= time.time()-frame["stamp"] < .5:
                raise ValueError("Displayed frame expired; select on a fresh frame")
            self.target = OpticalTarget(frame["rgb"], bbox, frame["stamp"])
            self.state = {"phase": "selected", "target_id": self.target.identifier, "image_stamp": frame["stamp"], "bbox": list(bbox)}
            return dict(self.state)

    def tick(self):
        with self.lock:
            if self.target is not None and self.target.phase == "tracking":
                try:
                    rgb, _, metadata = self.frame()
                    if metadata["image_stamp"] > self.target.stamp:
                        self.state = self.target.update(rgb, metadata["image_stamp"])
                        self.state.update(camera_info_version=metadata.get("camera_info_version"),
                            intrinsic_k=metadata.get("intrinsic_k"), distortion_d=metadata.get("distortion_d"))
                except (OSError, ValueError, cv2.error) as exc:
                    if time.time()-self.target.stamp > .5 or self.target.phase == "lost":
                        self.target.phase = "lost"
                        self.state.update(phase="lost", reason=str(exc), at=time.time())
            return dict(self.state)

    def cancel(self):
        with self.lock:
            self.target = None
            self.state = {"phase": "cancelled"}
            return dict(self.state)

    def colors(self):
        rgb, _, metadata = self.frame()
        return {"frame_id": metadata["frame_id"], "image_stamp": metadata["image_stamp"], "objects": color_objects(rgb)}

    def tags(self):
        rgb, _, metadata = self.frame()
        return {"frame_id": metadata["frame_id"], "image_stamp": metadata["image_stamp"], "tags": tag_objects(rgb)}

    def cloud(self, max_points=1500):
        if not 80 <= max_points <= 2000:
            raise ValueError("Point limit must be 80..2000")
        with np.load(self.root / "data/rgbd-snapshot.npz", allow_pickle=False) as raw:
            sample = {name: raw[name].copy() for name in ("rgb", "depth", "k", "d", "stamp", "depth_stamp", "frame")}
        stamp = float(sample["stamp"])
        depth_stamp = float(sample["depth_stamp"])
        if not np.isfinite([stamp, depth_stamp]).all() or abs(stamp-depth_stamp) > .04:
            raise ValueError("RGB/depth timestamps are not synchronized")
        if not 0 <= time.time()-min(stamp, depth_stamp) < 1.5:
            raise ValueError("Aligned RGB-D observation is stale")
        points, pixels = deproject(sample["depth"], sample["k"], sample["d"], 8)
        step = max(1, int(np.ceil(len(points)/max_points)))
        points, pixels = points[::step], pixels[::step]
        colors = sample["rgb"][pixels[:, 1], pixels[:, 0], ::-1]
        try:
            plane = fit_plane(points)
        except ValueError as exc:
            plane = {"available": False, "reason": str(exc)}
        return {"image_stamp": stamp, "depth_stamp": depth_stamp, "frame": str(sample["frame"].item()), "units": "metres",
                "xyz": points.tolist(), "rgb": colors.tolist(), "plane": plane,
                "source": "registered_rgbd", "map_transform_applied": False,
                "volume": None, "volume_reason": "One visible surface cannot establish complete object volume"}
