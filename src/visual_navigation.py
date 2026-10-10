"""Fresh RGB-D geometry for bounded visual navigation proposals, never actuator I/O."""
from dataclasses import dataclass
import math
from types import MappingProxyType
import time

import cv2
import numpy as np

from click_navigation import floor_goal, selected_depth_point


HSV_BANDS = {
    "red": (((0, 110, 65), (10, 255, 255)), ((170, 110, 65), (179, 255, 255))),
    "yellow": (((18, 100, 75), (34, 255, 255)),),
    "green": (((35, 85, 55), (85, 255, 255)),),
    "blue": (((95, 100, 60), (130, 255, 255)),),
    "white": (((0, 0, 200), (179, 50, 255)),),
    "black": (((0, 0, 0), (179, 255, 45)),),
}


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class NavigationLimits:
    max_step_m: float = .4
    max_turn_rad: float = .35
    heading_tolerance_rad: float = .18
    corridor_half_width_m: float = .24
    frame_max_age_s: float = .5
    max_target_distance_m: float = 3.5
    proposal_lifetime_s: float = .4

    def validate(self):
        bounds = {"max_step_m": (.08, .5), "max_turn_rad": (.05, .5), "heading_tolerance_rad": (.05, .25),
                  "corridor_half_width_m": (.18, .5), "frame_max_age_s": (.1, .7),
                  "max_target_distance_m": (.8, 3.5), "proposal_lifetime_s": (.1, .5)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid navigation limit: " + name)


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    k: np.ndarray
    d: np.ndarray
    image_stamp: float
    depth_stamp: float
    frame_id: str
    camera_frame: str
    camera_to_base: np.ndarray
    transform_stamp: float
    transform_version: str
    transform_accepted: bool
    pose: dict
    pose_stamp: float
    pose_verified: bool
    map_epoch: str

    def __post_init__(self):
        # Freeze the exact aligned image/geometry that generated the proposal.
        for name in ("rgb", "depth", "k", "d", "camera_to_base"):
            frozen = np.array(getattr(self, name), copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "pose", MappingProxyType(dict(self.pose)))


class ObservationFailure(ValueError):
    def __init__(self, reason, phase="blocked"):
        super().__init__(reason)
        self.phase = phase


class VisualNavigation:
    def __init__(self, limits=None):
        self.limits = limits or NavigationLimits()
        self.limits.validate()

    def _validate(self, frame, now, live_pose, live_epoch, live_transform_version):
        if not isinstance(frame, RGBDFrame):
            raise ObservationFailure("An immutable aligned RGBDFrame is required")
        stamps = (frame.image_stamp, frame.depth_stamp, frame.transform_stamp, frame.pose_stamp)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in stamps):
            raise ObservationFailure("Invalid observation timestamps")
        if not 0 <= now - frame.image_stamp < self.limits.frame_max_age_s or not 0 <= now - frame.depth_stamp < self.limits.frame_max_age_s:
            raise ObservationFailure("RGB-D frame is stale", "lost")
        if abs(frame.image_stamp - frame.depth_stamp) > .04:
            raise ObservationFailure("Color and depth timestamps are not aligned")
        if not isinstance(frame.frame_id, str) or not frame.frame_id or frame.camera_frame != "camera_color_optical_frame":
            raise ObservationFailure("Missing frame provenance or unsupported camera coordinate frame")
        if frame.rgb.dtype != np.uint8 or frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3 or min(frame.rgb.shape[:2]) < 80 or max(frame.rgb.shape[:2]) > 1280:
            raise ObservationFailure("Expected bounded BGR image")
        if frame.depth.shape != frame.rgb.shape[:2] or frame.depth.dtype.kind != "f":
            raise ObservationFailure("Depth must be aligned to RGB and expressed as floating-point metres")
        if frame.transform_accepted is not True or not isinstance(frame.transform_version, str) or not frame.transform_version:
            raise ObservationFailure("Camera-to-base calibration has not been accepted")
        if abs(frame.transform_stamp - frame.image_stamp) > .08:
            raise ObservationFailure("Camera transform does not describe the captured arm pose")
        transform = np.asarray(frame.camera_to_base, dtype=float)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
                or abs(np.linalg.det(transform[:3, :3]) - 1) > 1e-4):
            raise ObservationFailure("Invalid camera-to-base rigid transform")
        k, d = np.asarray(frame.k, dtype=float), np.asarray(frame.d, dtype=float).reshape(-1)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1], atol=1e-8)
                or not np.isfinite(d).all() or len(d) not in (4, 5, 8, 12, 14)):
            raise ObservationFailure("Camera intrinsics/distortion unavailable")
        if frame.pose_verified is not True or frame.pose.get("frame_id") != "map" or not isinstance(frame.map_epoch, str) or not frame.map_epoch:
            raise ObservationFailure("Verified localization in the current map is required")
        if not 0 <= now - frame.pose_stamp < self.limits.frame_max_age_s or abs(frame.pose_stamp - frame.image_stamp) > .08:
            raise ObservationFailure("Map pose is stale or not synchronized with RGB-D")
        for key in ("x", "y", "yaw"):
            value = frame.pose.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ObservationFailure("Invalid measured map pose")
        if live_epoch is not None and live_epoch != frame.map_epoch:
            raise ObservationFailure("Map epoch changed after capture")
        if live_transform_version is not None and live_transform_version != frame.transform_version:
            raise ObservationFailure("Camera calibration changed after capture")
        if live_pose is not None:
            try:
                drift = math.hypot(live_pose["x"] - frame.pose["x"], live_pose["y"] - frame.pose["y"])
                turn = abs(angle(live_pose["yaw"] - frame.pose["yaw"]))
            except (KeyError, TypeError, ValueError):
                raise ObservationFailure("Current map pose unavailable")
            if not math.isfinite(drift) or not math.isfinite(turn) or drift > .04 or turn > .05:
                raise ObservationFailure("Robot moved after the selected frame; observe again", "lost")

    def _evidence(self, frame, now):
        return {"frame_id": frame.frame_id, "image_stamp": frame.image_stamp, "depth_stamp": frame.depth_stamp,
                "camera_frame": frame.camera_frame, "transform_version": frame.transform_version,
                "map_epoch": frame.map_epoch, "pose_stamp": frame.pose_stamp, "pose_at_capture": dict(frame.pose),
                "valid_until": min(frame.image_stamp + self.limits.frame_max_age_s, now + self.limits.proposal_lifetime_s),
                "requires_nav2_clearance": True, "physical_corridor_verified": False,
                "corridor_half_width_m": self.limits.corridor_half_width_m}

    def _map_pose(self, frame, x, y, heading):
        yaw = frame.pose["yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        return {"frame_id": "map", "x": float(frame.pose["x"] + c * x - s * y),
                "y": float(frame.pose["y"] + s * x + c * y), "yaw": angle(yaw + heading)}

    def _proposal(self, frame, endpoint, tangent, evidence):
        distance = math.hypot(*endpoint)
        bearing = math.atan2(endpoint[1], endpoint[0])
        if abs(bearing) > self.limits.heading_tolerance_rad:
            turn = max(-self.limits.max_turn_rad, min(self.limits.max_turn_rad, bearing))
            poses = [self._map_pose(frame, 0, 0, turn)]
            return {"state": "proposal", "motion": "rotate", "stop_required": False,
                    "poses": poses, "advance_m": 0., "turn_rad": turn, "evidence": evidence}
        if distance < .05:
            return {"state": "reached", "stop_required": True, "poses": [], "evidence": evidence}
        fraction = min(1, self.limits.max_step_m / distance)
        x, y = endpoint[0] * fraction, endpoint[1] * fraction
        yaw = max(-self.limits.max_turn_rad, min(self.limits.max_turn_rad, tangent))
        poses = [self._map_pose(frame, x * step / 3, y * step / 3, yaw * step / 3) for step in (1, 2, 3)]
        return {"state": "proposal", "motion": "navigate", "stop_required": False, "poses": poses,
                "advance_m": math.hypot(x, y), "turn_rad": yaw, "evidence": evidence}

    def _failure(self, exc, frame):
        return {"state": getattr(exc, "phase", "blocked"), "reason": str(exc), "stop_required": True, "poses": [],
                "evidence": {"frame_id": getattr(frame, "frame_id", None), "physical_corridor_verified": False}}

    def target_step(self, frame, target, stand_off_m=.65, live_pose=None, live_epoch=None, live_transform_version=None, now=None):
        """One measured selected-ROI step; held objects and lost identity never cause pursuit."""
        checked_at = time.time() if now is None else now
        try:
            self._validate(frame, checked_at, live_pose, live_epoch, live_transform_version)
            if type(stand_off_m) not in (int, float) or not math.isfinite(stand_off_m) or not .4 <= stand_off_m <= 1.2:
                raise ObservationFailure("Stand-off must be .4..1.2 metres")
            if not isinstance(target, dict) or not target.get("target_id") or target.get("phase") != "tracking":
                raise ObservationFailure("Selected object association was lost", "lost")
            if target.get("held") or target.get("attached") or target.get("attachment"):
                raise ObservationFailure("Held/attached objects must not be pursued by the base")
            target_stamp = target.get("image_stamp")
            association = target.get("association_fraction", 0)
            if type(target_stamp) not in (int, float) or not math.isfinite(target_stamp) or abs(target_stamp - frame.image_stamp) > .04:
                raise ObservationFailure("Target does not belong to the aligned depth frame", "lost")
            if type(association) not in (int, float) or not math.isfinite(association) or not .65 <= association <= 1:
                raise ObservationFailure("Selected object association is ambiguous", "lost")
            if target.get("frame_id") is not None and target["frame_id"] != frame.frame_id:
                raise ObservationFailure("Selected target frame identifier differs from RGB-D", "lost")
            if target.get("image_size") is not None and target["image_size"] != [frame.rgb.shape[1], frame.rgb.shape[0]]:
                raise ObservationFailure("Selected ROI image dimensions differ from depth")
            box = np.asarray(target.get("bbox", []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or not 0 <= box[0] < box[2] <= frame.rgb.shape[1] or not 0 <= box[1] < box[3] <= frame.rgb.shape[0] or min(box[2:] - box[:2]) < 15:
                raise ObservationFailure("Selected object ROI is invalid or too small")
            point, depth, spread = selected_depth_point(frame.depth, frame.k, frame.d, *(box[:2] + box[2:]) / 2)
            base = frame.camera_to_base @ point
            distance = math.hypot(base[0], base[1])
            if not .2 <= distance <= self.limits.max_target_distance_m or not -.06 <= base[2] <= 1.4:
                raise ObservationFailure("Measured selected object is outside the permitted base workspace")
            evidence = dict(self._evidence(frame, checked_at), target_id=target["target_id"], target_bbox=box.tolist(),
                target_image_stamp=target_stamp, association_fraction=association, target_point_base_m=base[:3].tolist(),
                object_distance_m=distance, depth_m=depth, depth_spread_m=spread, stand_off_m=stand_off_m,
                semantic_identity_verified=False, object_motion_prediction_used=False)
            if distance <= stand_off_m + .07:
                return {"state": "reached", "stop_required": True, "poses": [], "evidence": evidence}
            advance = min(self.limits.max_step_m, distance - stand_off_m)
            endpoint = base[:2] / distance * advance
            return self._proposal(frame, endpoint, math.atan2(base[1], base[0]), evidence)
        except (ValueError, TypeError, KeyError) as exc:
            return self._failure(exc, frame)

    def _line_centres(self, image, color):
        if color not in HSV_BANDS:
            raise ObservationFailure("Choose red/yellow/green/blue/white/black line")
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros((h, w), np.uint8)
        for low, high in HSV_BANDS[color]:
            mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        mask[:round(h * .52)] = 0
        mask[round(h * .96):] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        ranked = sorted(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True)
        if not ranked or stats[ranked[0], cv2.CC_STAT_AREA] < 100:
            raise ObservationFailure("Line not visible or ended; completion is not proven", "lost")
        area = int(stats[ranked[0], cv2.CC_STAT_AREA])
        if area > h * w * .10 or (len(ranked) > 1 and stats[ranked[1], cv2.CC_STAT_AREA] > area * .35):
            raise ObservationFailure("Colored regions are cluttered or line identity is ambiguous", "lost")
        component = labels == ranked[0]
        centres = []
        for v in np.linspace(round(h * .55), round(h * .94), 16).astype(int):
            pixels = np.flatnonzero(component[v])
            if len(pixels) >= 3:
                gaps = np.flatnonzero(np.diff(pixels) > 2)
                if len(gaps) or len(pixels) > w * .18:
                    raise ObservationFailure("Line branch or broad colored region is ambiguous", "lost")
                centres.append([float(np.median(pixels)), float(v)])
        if len(centres) < 8 or centres[-1][1] - centres[0][1] < h * .22:
            raise ObservationFailure("Insufficient connected line coverage", "lost")
        centres = np.asarray(centres)
        if np.max(np.diff(centres[:, 1])) > h * .09:
            raise ObservationFailure("Line continuity was lost", "lost")
        curve = np.polyfit(centres[:, 1], centres[:, 0], 2)
        residual = np.abs(centres[:, 0] - np.polyval(curve, centres[:, 1]))
        if np.percentile(residual, 90) > 8:
            raise ObservationFailure("Image line fit is inconsistent", "lost")
        return centres, area, float(np.percentile(residual, 90))

    def line_step(self, frame, color="yellow", endpoint_map=None, live_pose=None, live_epoch=None, live_transform_version=None, now=None):
        """Fit a colored line only after several real depth patches establish local floor."""
        checked_at = time.time() if now is None else now
        try:
            self._validate(frame, checked_at, live_pose, live_epoch, live_transform_version)
            if endpoint_map is not None:
                if (not isinstance(endpoint_map, dict) or endpoint_map.get("frame_id") != "map"
                        or endpoint_map.get("map_epoch") != frame.map_epoch
                        or any(type(endpoint_map.get(key)) not in (int, float) or not math.isfinite(endpoint_map[key]) for key in ("x", "y"))):
                    raise ObservationFailure("Declared endpoint must be a finite point in the current map epoch")
                if math.hypot(endpoint_map["x"] - frame.pose["x"], endpoint_map["y"] - frame.pose["y"]) <= .12:
                    return {"state": "reached", "stop_required": True, "poses": [], "evidence": dict(self._evidence(frame, checked_at), completion="declared_map_endpoint")}
            centres, area, pixel_residual = self._line_centres(frame.rgb, color)
            points, floor_evidence = [], []
            for u, v in centres:
                patch = floor_goal(frame.depth, frame.k, frame.d, u, v, frame.camera_to_base, frame.pose)
                point, _, _ = selected_depth_point(frame.depth, frame.k, frame.d, u, v)
                base = frame.camera_to_base @ point
                points.append(base[:3])
                floor_evidence.append({"pixel": [u, v], "height_m": patch["floor_height_m"], "plane_p90_m": patch["floor_fit_p90_m"], "normal_base": patch["floor_normal_base"]})
            points = np.asarray(points)
            ordered = points[np.argsort(points[:, 0])]
            if ordered[-1, 0] - ordered[0, 0] < .16 or np.max(np.diff(ordered[:, 0])) > .25:
                raise ObservationFailure("Measured floor line has insufficient forward continuity", "lost")
            if np.min(np.linalg.norm(ordered[:, :2], axis=1)) > .8:
                raise ObservationFailure("Floor line is too far away for a bounded following step", "lost")
            coefficients = np.polyfit(ordered[:, 0], ordered[:, 1], 2)
            residual = np.abs(ordered[:, 1] - np.polyval(coefficients, ordered[:, 0]))
            if np.percentile(residual, 90) > .035:
                raise ObservationFailure("Metric floor line fit is ambiguous", "lost")
            # Aim inside observed floor coverage; never extrapolate a line through an unseen near field.
            lookahead_x = max(float(ordered[0, 0]), min(float(ordered[-1, 0]), self.limits.max_step_m))
            lookahead_y = float(np.polyval(coefficients, lookahead_x))
            tangent = math.atan2(float(2 * coefficients[0] * lookahead_x + coefficients[1]), 1.)
            evidence = dict(self._evidence(frame, checked_at), color=color, measured_line_points_base_m=points.tolist(),
                floor_patches=floor_evidence, floor_verified=True, pixel_line_fit_p90_px=pixel_residual,
                metric_line_fit_p90_m=float(np.percentile(residual, 90)), colored_area_px=area,
                actual_line_end_verified=False, traversability_source="nav2_costmap_and_lidar_required")
            return self._proposal(frame, [lookahead_x, lookahead_y], tangent, evidence)
        except (ValueError, TypeError, KeyError) as exc:
            return self._failure(exc, frame)
