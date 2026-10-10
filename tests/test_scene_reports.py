import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scene_reports import SceneReports, ZoneWatch, scene_composition


class SceneReportTests(unittest.TestCase):
    def test_report_preserves_observations_without_cross_view_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            surveys = root / "data/surveys/task"
            surveys.mkdir(parents=True)
            image = cv2.imencode(".jpg", np.full((40, 40, 3), 80, np.uint8))[1].tobytes()
            (surveys / "actual.jpg").write_bytes(image)
            for index in range(2):
                record = {"id": str(index), "mission": "task", "at": 100 + index, "place": "<script>alert(1)</script>",
                          "context_image": "actual.jpg", "context_image_same_frame_verified": False,
                          "perception": {"image_stamp": 100 + index, "frame": "camera",
                                         "objects": [{"label": "sock", "confidence": .8, "color": "blue", "shape_2d": "irregular"}]}}
                (surveys / (str(index) + ".json")).write_text(json.dumps(record))
            reports = SceneReports(root)
            result = reports.export("task")
            self.assertEqual(len(result["report"]["observations"]), 2)
            self.assertIsNone(result["report"]["physical_item_count"])
            self.assertFalse(result["report"]["deduplicated_inventory_verified"])
            for row in result["report"]["observations"]:
                self.assertEqual(row["composition"]["per_frame_hypothesis_counts"], {"sock": 1})
                self.assertEqual(row["image"]["sha256"], hashlib.sha256(image).hexdigest())
                self.assertFalse(row["context_image_same_frame_verified"])
            exported = reports.artifact(result["id"], "report.html").read_text()
            self.assertNotIn("<script>", exported)
            self.assertIn("&lt;script&gt;", exported)
            csv_rows = list(csv.reader(io.StringIO(reports.artifact(result["id"], "report.csv").read_text())))
            self.assertEqual(len(csv_rows), 3)
            self.assertEqual(json.loads(reports.artifact(result["id"], "report.json").read_text())["mission"], "task")

    def test_missing_or_escaping_artifacts_report_error_and_no_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            surveys = root / "data/surveys/task"
            surveys.mkdir(parents=True)
            outside = root / "secret.jpg"
            outside.write_text("private")
            (surveys / "linked.jpg").symlink_to(outside)
            (surveys / "bad.json").write_text(json.dumps({"id": "bad", "mission": "task", "context_image": "linked.jpg", "perception": {}}))
            reports = SceneReports(root)
            result = reports.export("task")
            self.assertEqual(result["report"]["observations"], [])
            self.assertEqual(len(result["report"]["record_errors"]), 1)
            with self.assertRaises(ValueError):
                reports.export("../task")
            with self.assertRaises(ValueError):
                reports.artifact(result["id"], "../../secret.jpg")
            with self.assertRaises(ValueError):
                reports.export("missing")

    def test_csv_formula_escape_and_shape_is_hypothesis(self):
        result = scene_composition({"objects": [{"label": "=unknown", "confidence": .5, "shape_2d": "round", "tag_id": 7}]})
        self.assertEqual(result["identity_status"], "unknown")
        self.assertFalse(result["detections"][0]["identity_verified"])
        self.assertEqual(result["detections"][0]["visual_tag_id"], 7)
        with self.assertRaises(ValueError):
            scene_composition({"objects": [{"label": "sock", "confidence": float("nan")}]})


class ZoneWatchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.image = rng.integers(30, 150, (160, 200), dtype=np.uint8)
        self.roi = [65, 50, 135, 120]
        self.reference = {"at": 100., "arm_deg": [90.] * 6, "base_pose": [0., 0., 0.],
                          "stationary": True, "reference_valid": True, "calibration_id": "arm-v1", "map_epoch": "map-v1"}
        self.watch = ZoneWatch()

    def frame(self, image=None, stamp=100., identity="first"):
        jpeg = cv2.imencode(".jpg", self.image if image is None else image)[1].tobytes()
        metadata = {"frame_id": identity, "image_stamp": stamp, "camera_id": "onboard", "camera_info_version": "k-v1",
                    "frame": "color_optical", "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(), "same_frame_hash_verified": True}
        return jpeg, metadata

    def begin(self):
        return self.watch.begin(*self.frame(), self.reference, self.roi, now=100.)

    def compare(self, image=None, reference=None, stamp=100.5, identity="second"):
        current = dict(self.reference, at=stamp) if reference is None else reference
        return self.watch.compare(*self.frame(image, stamp, identity), current, now=stamp)

    def test_actual_new_frames_change_and_unchanged(self):
        self.begin()
        result = self.compare()
        self.assertEqual(result["phase"], "unchanged")
        changed = self.image.copy()
        changed[65:110, 80:120] = 245
        result = self.compare(changed, stamp=100.6, identity="third")
        self.assertEqual(result["phase"], "changed")
        self.assertGreater(result["changed_fraction"], .04)
        self.assertFalse(result["motion_requested"])
        self.assertEqual(result["semantic_identity"], "unknown")
        self.assertNotEqual(result["baseline_image_sha256"], result["image_sha256"])
        with self.assertRaises(ValueError):
            self.compare(changed, stamp=100.6, identity="third")

    def test_arm_base_motion_calibration_change_rejected(self):
        self.begin()
        for modification in ({"arm_deg": [91.] * 6}, {"base_pose": [.02, 0., 0.]}, {"base_pose": [0., 0., .03]},
                             {"stationary": False}, {"calibration_id": "v2"}, {"map_epoch": "v2"}):
            with self.assertRaises(ValueError):
                self.compare(reference=dict(self.reference, at=100.5, **modification))

    def test_camera_shift_rejected_even_with_unchanged_state(self):
        self.begin()
        matrix = np.float32([[1, 0, 8], [0, 1, 0]])
        shifted = cv2.warpAffine(self.image, matrix, (200, 160))
        with self.assertRaisesRegex(ValueError, "shifted|association"):
            self.compare(shifted)

    def test_blank_unavailable_stale_or_reused_image_rejected(self):
        for image in (np.zeros_like(self.image), np.full_like(self.image, 100)):
            with self.assertRaises(ValueError):
                self.watch.begin(*self.frame(image), self.reference, self.roi, now=100.)
        with self.assertRaises(ValueError):
            self.watch.begin(b"unavailable", self.frame()[1], self.reference, self.roi, now=100.)
        self.begin()
        with self.assertRaises(ValueError):
            self.watch.compare(*self.frame(), self.reference, now=100.5)
        with self.assertRaises(ValueError):
            self.watch.compare(*self.frame(stamp=100.5, identity="second"), dict(self.reference, at=100.5), now=102.)
        self.watch.cancel()
        with self.assertRaises(ValueError):
            self.compare()

    def test_global_brightness_is_not_zone_change_and_reference_is_frozen(self):
        self.begin()
        self.reference["arm_deg"][0] = 91.
        with self.assertRaises(ValueError):
            self.compare()
        self.reference["arm_deg"][0] = 90.
        result = self.compare(self.image + 20)
        self.assertEqual(result["phase"], "unchanged")


if __name__ == "__main__":
    unittest.main()
