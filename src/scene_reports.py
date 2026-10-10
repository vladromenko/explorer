"""Saved observational exports and passive two-frame zone comparison; no motion."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import time
import uuid

import cv2
import numpy as np


def _name(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value) is None:
        raise ValueError("Invalid artifact identifier")
    return value


def _safe_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Saved artifact is unavailable or outside its directory")
    return path


def _save(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def scene_composition(perception):
    """Counts are detector hypotheses in ONE timestamped frame, never inventory."""
    objects = perception.get("objects", [])
    if not isinstance(objects, list) or len(objects) > 500:
        raise ValueError("Invalid or excessive detection list")
    counts = {}
    rows = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("Invalid detection")
        confidence = item.get("confidence")
        label = item.get("label", "unknown")
        if confidence is not None and (type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Invalid detector confidence")
        if not isinstance(label, str) or len(label) > 200:
            raise ValueError("Invalid detector label")
        hypothesis = {"label_hypothesis": label, "confidence": confidence, "identity_verified": False}
        for source, destination in (("color", "color_hypothesis"), ("shape_2d", "shape_2d_hypothesis"),
                                    ("bbox", "bbox_px"), ("tag_id", "visual_tag_id"), ("family", "visual_tag_family")):
            if source in item:
                hypothesis[destination] = item[source]
        rows.append(hypothesis)
        counts[label] = counts.get(label, 0) + 1
    return {"image_stamp": perception.get("image_stamp"), "camera_frame": perception.get("frame"),
            "per_frame_hypothesis_counts": counts, "detections": rows,
            "physical_item_count": None, "deduplicated_inventory_verified": False,
            "identity_status": "unknown", "counts_scope": "single_detection_frame"}


class SceneReports:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.surveys = self.root / "data/surveys"
        self.exports = self.root / "data/scene-reports"

    def _directories(self):
        if any(not folder.resolve().is_relative_to(self.root) for folder in (self.surveys, self.exports)):
            raise ValueError("Report storage escapes the project directory")

    def export(self, mission):
        mission = _name(mission)
        self._directories()
        directory = self.surveys / mission
        if not directory.resolve().is_relative_to(self.surveys.resolve()) or not directory.is_dir():
            raise ValueError("Saved survey mission unavailable")
        paths = sorted(directory.glob("*.json"))
        if not paths or len(paths) > 2000:
            raise ValueError("Report requires 1..2000 saved observations")
        identifier = uuid.uuid4().hex
        destination = self.exports / identifier
        destination.mkdir(parents=True)
        observations, errors = [], []
        for path in paths:
            try:
                source = _safe_file(directory, path.name)
                source_bytes = source.read_bytes()
                if len(source_bytes) > 2_000_000:
                    raise ValueError("Saved observation is oversized")
                record = json.loads(source_bytes)
                if not isinstance(record, dict) or record.get("mission") != mission:
                    raise ValueError("Saved observation mission mismatch")
                composition = scene_composition(record.get("perception", {}))
                row = {key: record.get(key) for key in ("id", "at", "place", "robot_pose", "map_epoch", "battery_voltage_v", "summary")}
                row.update(composition=composition, source_record=str(source.relative_to(self.root)),
                           source_record_sha256=hashlib.sha256(source_bytes).hexdigest(),
                           image=None, context_image_same_frame_verified=record.get("context_image_same_frame_verified") is True)
                if record.get("context_image"):
                    image = _safe_file(directory, record["context_image"])
                    content = image.read_bytes()
                    if len(content) > 8_000_000 or cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) is None:
                        raise ValueError("Saved context image is invalid or oversized")
                    name = str(len(observations)) + ".jpg"
                    (destination / name).write_bytes(content)
                    row["image"] = {"path": name, "sha256": hashlib.sha256(content).hexdigest(),
                                    "source_path": str(image.relative_to(self.root)), "kind": "saved_actual_context"}
                observations.append(row)
            except (OSError, ValueError, TypeError, cv2.error) as exc:
                errors.append({"file": path.name, "error": str(exc)})
        report = {"id": identifier, "mission": mission, "generated_at": time.time(), "observations": observations,
                  "record_errors": errors, "physical_item_count": None, "deduplicated_inventory_verified": False,
                  "claim": "Saved observations and single-frame detector hypotheses; no verified inventory total"}
        _save(destination / "report.json", report)
        with (destination / "report.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["observation_id", "place", "image_stamp", "label_hypothesis", "confidence", "color_hypothesis", "shape_2d_hypothesis", "image", "same_frame_verified"])
            for row in observations:
                for detection in row["composition"]["detections"] or [{}]:
                    values = [row["id"], row["place"], row["composition"]["image_stamp"], detection.get("label_hypothesis"),
                              detection.get("confidence"), detection.get("color_hypothesis"), detection.get("shape_2d_hypothesis"),
                              row["image"]["path"] if row["image"] else "", row["context_image_same_frame_verified"]]
                    # CSV is safe to open in spreadsheet tools: labels cannot be formulas.
                    writer.writerow(["'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value for value in values])
        sections = []
        for row in observations:
            image = "<img width=640 alt='Сохранённый реальный кадр' src='" + row["image"]["path"] + "'>" if row["image"] else "<p>Изображение отсутствует</p>"
            sections.append("<section><h2>" + html.escape(str(row["place"])) + "</h2>" + image + "<p>Совпадение кадра с детекцией: " + str(row["context_image_same_frame_verified"]) + "</p><pre>" + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + "</pre></section>")
        (destination / "report.html").write_text("<!doctype html><meta charset=utf-8><title>Наблюдения Explorer</title><h1>Наблюдения Explorer</h1><p>Гипотезы детектора в отдельных кадрах. Общая инвентаризация не подтверждена.</p>" + "".join(sections))
        return {"id": identifier, "report": report, "artifacts": {format: str((destination / ("report." + format)).relative_to(self.root)) for format in ("json", "csv", "html")}}

    def artifact(self, identifier, filename):
        identifier = _name(identifier)
        self._directories()
        if not (self.exports / identifier).resolve().is_relative_to(self.exports.resolve()):
            raise ValueError("Report directory escapes artifact storage")
        if filename not in ("report.json", "report.csv", "report.html") and re.fullmatch(r"\d{1,4}\.jpg", str(filename)) is None:
            raise ValueError("Unsupported report artifact")
        return _safe_file(self.exports / identifier, filename)


class ZoneWatch:
    """Passive bounded two-frame comparison; rejects moving or unknown references."""
    def __init__(self):
        self.baseline = None
        self.latest = None

    @staticmethod
    def _frame(jpeg, metadata, reference, now):
        if not isinstance(jpeg, bytes) or len(jpeg) > 8_000_000:
            raise ValueError("Camera bytes unavailable or oversized")
        if metadata.get("jpeg_sha256") != hashlib.sha256(jpeg).hexdigest() or metadata.get("same_frame_hash_verified") is not True:
            raise ValueError("Camera frame provenance mismatch")
        for stamp in (metadata.get("image_stamp"), reference.get("at")):
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= now - stamp <= .8:
                raise ValueError("Fresh camera/reference unavailable")
        if abs(reference["at"] - metadata["image_stamp"]) > .4:
            raise ValueError("Camera/reference timestamps disagree")
        if reference.get("reference_valid") is not True or reference.get("stationary") is not True:
            raise ValueError("Stationary arm/base reference is not confirmed")
        q = reference.get("arm_deg")
        pose = reference.get("base_pose")
        if not isinstance(q, (list, tuple)) or len(q) != 6 or not isinstance(pose, (list, tuple)) or len(pose) != 3:
            raise ValueError("Arm/base reference unavailable")
        if any(type(item) not in (int, float) or not math.isfinite(item) for item in (*q, *pose)):
            raise ValueError("Invalid arm/base reference")
        if not reference.get("calibration_id") or not reference.get("map_epoch"):
            raise ValueError("Reference provenance unavailable")
        if not all(metadata.get(key) for key in ("frame_id", "camera_id", "camera_info_version", "frame")):
            raise ValueError("Camera identity/calibration unavailable")
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size > 2_100_000 or min(image.shape) < 40 or float(image.std()) < 3.:
            raise ValueError("Camera image unavailable, blank or oversized")
        frozen_reference = dict(reference, arm_deg=tuple(q), base_pose=tuple(pose))
        return {"image": image, "metadata": dict(metadata), "reference": frozen_reference}

    def begin(self, jpeg, metadata, reference, roi, now=None):
        frame = self._frame(jpeg, metadata, reference, time.time() if now is None else now)
        if not isinstance(roi, (list, tuple)) or len(roi) != 4 or any(type(item) is not int for item in roi):
            raise ValueError("Select an integer pixel ROI")
        x1, y1, x2, y2 = roi
        height, width = frame["image"].shape
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height or min(x2-x1, y2-y1) < 12:
            raise ValueError("Zone ROI outside camera or too small")
        frame["roi"] = tuple(roi)
        self.baseline, self.latest = frame, None
        return {"phase": "watching", "frame_id": metadata["frame_id"], "roi": list(roi), "motion_requested": False}

    def compare(self, jpeg, metadata, reference, now=None):
        if self.baseline is None:
            raise ValueError("Select zone on a verified fresh frame first")
        current = self._frame(jpeg, metadata, reference, time.time() if now is None else now)
        previous = self.baseline
        for key in ("camera_id", "camera_info_version", "frame"):
            if current["metadata"][key] != previous["metadata"][key]:
                raise ValueError("Camera identity/reference changed")
        delta = metadata["image_stamp"] - previous["metadata"]["image_stamp"]
        if not 0 < delta <= 60 or metadata["frame_id"] == previous["metadata"]["frame_id"]:
            raise ValueError("Comparison needs a new frame within sixty seconds")
        if self.latest is not None and (metadata["image_stamp"] <= self.latest["metadata"]["image_stamp"]
                                       or metadata["frame_id"] == self.latest["metadata"]["frame_id"]):
            raise ValueError("Comparison frame has already been evaluated")
        old, new = previous["reference"], current["reference"]
        yaw_delta = math.atan2(math.sin(old["base_pose"][2] - new["base_pose"][2]), math.cos(old["base_pose"][2] - new["base_pose"][2]))
        if (old["calibration_id"] != new["calibration_id"] or old["map_epoch"] != new["map_epoch"]
                or max(abs(a-b) for a, b in zip(old["arm_deg"], new["arm_deg"])) > .3
                or math.dist(old["base_pose"][:2], new["base_pose"][:2]) > .015 or abs(yaw_delta) > .02):
            raise ValueError("Arm/base moved; select a new fixed-camera baseline")
        if current["image"].shape != previous["image"].shape:
            raise ValueError("Camera image dimensions changed")
        x1, y1, x2, y2 = previous["roi"]
        before = cv2.GaussianBlur(previous["image"], (5, 5), 0).astype(np.float32)
        after = cv2.GaussianBlur(current["image"], (5, 5), 0).astype(np.float32)
        # Outside-zone image alignment independently checks unreported camera motion.
        mask = np.ones(before.shape, dtype=bool)
        mask[y1:y2, x1:x2] = False
        if mask.sum() < 400 or float(before[mask].std()) < 3.:
            raise ValueError("Keep textured context outside the watched zone")
        alignment_before, alignment_after = before.copy(), after.copy()
        alignment_before[~mask] = 0
        alignment_after[~mask] = 0
        shift, response = cv2.phaseCorrelate(alignment_before, alignment_after)
        if response < .15 or math.hypot(*shift) > 2.:
            raise ValueError("Camera view shifted or context association unavailable")
        lighting_offset = float(np.median((after-before)[mask]))
        difference = np.abs(after[y1:y2, x1:x2] - before[y1:y2, x1:x2] - lighting_offset)
        fraction = float((difference > 18.).mean())
        self.latest = current
        return {"phase": "changed" if fraction >= .04 else "unchanged", "changed_fraction": fraction,
                "pixel_threshold": 18., "fraction_threshold": .04, "roi": list(previous["roi"]),
                "baseline_frame_id": previous["metadata"]["frame_id"], "frame_id": metadata["frame_id"],
                "baseline_image_sha256": previous["metadata"]["jpeg_sha256"], "image_sha256": metadata["jpeg_sha256"],
                "image_stamp": metadata["image_stamp"], "camera_reference": {key: metadata[key] for key in ("camera_id", "camera_info_version", "frame")},
                "reference": dict(reference), "semantic_identity": "unknown", "motion_requested": False,
                "claim": "Visual zone change hypothesis; no verified object arrival/removal"}

    def cancel(self):
        self.baseline = self.latest = None
        return {"phase": "cancelled", "motion_requested": False}
