from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_navigation import RGBDFrame, NavigationLimits, VisualNavigation


def snapshot(line=True, floor_height=0., yaw=0., side_px=0):
    h, w, fx = 480, 640, 500.
    k = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=float)
    pitch = math.radians(35)
    c, s = math.cos(pitch), math.sin(pitch)
    transform = np.array([[0, -s, c, 0], [-1, 0, 0, 0], [0, -c, -s, .6], [0, 0, 0, 1]], dtype=float)
    v, _ = np.mgrid[0:h, 0:w]
    depth = (.6 - floor_height) / (c * ((v - h / 2) / fx) + s)
    rgb = np.full((h, w, 3), 110, np.uint8)
    if line:
        cv2.line(rgb, (w // 2 + side_px, h - 1), (w // 2 + side_px, h // 2), (255, 0, 0), 13)
    return RGBDFrame(rgb, depth, k, np.zeros(5), 10., 10., "rgbd-frame-001", "camera_color_optical_frame", transform,
                     10., "accepted-handeye-1", True, {"frame_id": "map", "x": 10., "y": 20., "yaw": yaw}, 10., True, "map-epoch-1")


def object_frame(distance=1.5, yaw=0.):
    frame = snapshot()
    transform = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, .5], [0, 0, 0, 1]], dtype=float)
    return replace(frame, depth=np.full((480, 640), distance, dtype=float), camera_to_base=transform,
                   pose={"frame_id": "map", "x": 10., "y": 20., "yaw": yaw})


def target(frame, x=320, stamp=10.):
    return {"target_id": "selected-object", "phase": "tracking", "bbox": [x - 20, 220, x + 20, 260],
            "image_stamp": stamp, "frame_id": frame.frame_id, "image_size": [640, 480], "association_fraction": .9}


class VisualNavigationTests(unittest.TestCase):
    def setUp(self):
        self.nav = VisualNavigation()

    def test_line_uses_measured_floor_and_map_frame(self):
        result = self.nav.line_step(snapshot(yaw=math.pi / 2), "blue", now=10.1)
        self.assertEqual(result["state"], "proposal", result)
        self.assertEqual(result["motion"], "navigate")
        self.assertLessEqual(result["advance_m"], .4)
        self.assertGreaterEqual(len(result["evidence"]["floor_patches"]), 8)
        self.assertTrue(result["evidence"]["floor_verified"])
        self.assertTrue(result["evidence"]["requires_nav2_clearance"])
        self.assertFalse(result["evidence"]["physical_corridor_verified"])
        self.assertAlmostEqual(result["poses"][-1]["x"], 10.)
        self.assertGreater(result["poses"][-1]["y"], 20.)
        self.assertEqual(result["poses"][-1]["frame_id"], "map")

    def test_line_missing_is_lost_not_completion(self):
        result = self.nav.line_step(snapshot(line=False), "blue", now=10.1)
        self.assertEqual(result["state"], "lost")
        self.assertTrue(result["stop_required"])
        self.assertFalse(result["poses"])

    def test_line_above_floor_is_rejected(self):
        result = self.nav.line_step(snapshot(floor_height=.2), "blue", now=10.1)
        self.assertEqual(result["state"], "blocked", result)
        self.assertTrue(result["stop_required"])
        self.assertIn("пола", result["reason"])

    def test_two_equal_color_lines_are_ambiguous(self):
        frame = snapshot()
        image = frame.rgb.copy()
        cv2.line(image, (220, 479), (220, 240), (255, 0, 0), 13)
        result = self.nav.line_step(replace(frame, rgb=image), "blue", now=10.1)
        self.assertEqual(result["state"], "lost")
        self.assertIn("ambiguous", result["reason"])

    def test_broad_region_and_junction_do_not_become_a_line(self):
        frame = snapshot()
        image = frame.rgb.copy()
        image[260:470, 100:550] = (255, 0, 0)
        result = self.nav.line_step(replace(frame, rgb=image), "blue", now=10.1)
        self.assertEqual(result["state"], "lost")
        junction = frame.rgb.copy()
        cv2.line(junction, (200, 400), (400, 400), (255, 0, 0), 15)
        self.assertEqual(self.nav.line_step(replace(frame, rgb=junction), "blue", now=10.1)["state"], "lost")

    def test_depth_missing_cannot_be_replaced_by_color(self):
        result = self.nav.line_step(replace(snapshot(), depth=np.zeros((480, 640), dtype=float)), "blue", now=10.1)
        self.assertEqual(result["state"], "blocked")
        self.assertFalse(result["poses"])

    def test_stale_and_unsynchronized_rgbd_stop(self):
        frame = snapshot()
        self.assertEqual(self.nav.line_step(frame, "blue", now=10.6)["state"], "lost")
        result = self.nav.line_step(replace(frame, depth_stamp=9.95), "blue", now=10.1)
        self.assertEqual(result["state"], "blocked")
        self.assertIn("aligned", result["reason"])

    def test_accepted_transform_and_verified_map_pose_are_required(self):
        for frame in (replace(snapshot(), transform_accepted=False), replace(snapshot(), pose_verified=False),
                      replace(snapshot(), transform_stamp=9.8), replace(snapshot(), pose_stamp=9.8)):
            result = self.nav.line_step(frame, "blue", now=10.1)
            self.assertEqual(result["state"], "blocked")
            self.assertTrue(result["stop_required"])

    def test_changed_map_camera_or_robot_pose_invalidates_frozen_frame(self):
        frame = snapshot()
        args = ({"live_epoch": "different-map"}, {"live_transform_version": "different-handeye"},
                {"live_pose": {"x": 10.1, "y": 20., "yaw": 0.}})
        for options in args:
            result = self.nav.line_step(frame, "blue", now=10.1, **options)
            self.assertTrue(result["stop_required"])
            self.assertFalse(result["poses"])

    def test_reflection_is_not_a_valid_extrinsic(self):
        frame = snapshot()
        transform = frame.camera_to_base.copy()
        transform[:3, 0] *= -1
        result = self.nav.line_step(replace(frame, camera_to_base=transform), "blue", now=10.1)
        self.assertEqual(result["state"], "blocked")

    def test_frame_copies_arrays_and_pose_for_exact_provenance(self):
        frame = snapshot()
        self.assertFalse(frame.rgb.flags.writeable)
        with self.assertRaises(ValueError):
            frame.depth[0, 0] = 0
        with self.assertRaises(TypeError):
            frame.pose["x"] = 0

    def test_selected_object_uses_measured_standoff_and_world_yaw(self):
        frame = object_frame(yaw=math.pi / 2)
        result = self.nav.target_step(frame, target(frame), now=10.1)
        self.assertEqual(result["state"], "proposal", result)
        self.assertEqual(result["motion"], "navigate")
        self.assertLessEqual(result["advance_m"], .4)
        self.assertAlmostEqual(result["evidence"]["object_distance_m"], 1.5)
        self.assertAlmostEqual(result["poses"][-1]["x"], 10.)
        self.assertAlmostEqual(result["poses"][-1]["y"], 20.4)
        self.assertEqual(result["evidence"]["frame_id"], frame.frame_id)
        self.assertFalse(result["evidence"]["semantic_identity_verified"])

    def test_target_turns_toward_measured_object_before_advance(self):
        frame = object_frame()
        result = self.nav.target_step(frame, target(frame, x=100), now=10.1)
        self.assertEqual(result["motion"], "rotate", result)
        self.assertEqual(result["advance_m"], 0.)
        self.assertGreater(result["turn_rad"], 0)
        self.assertLessEqual(result["turn_rad"], .35)
        self.assertEqual(result["poses"][0]["x"], 10.)

    def test_object_reached_does_not_send_extra_goal(self):
        frame = object_frame(distance=.68)
        result = self.nav.target_step(frame, target(frame), now=10.1)
        self.assertEqual(result["state"], "reached")
        self.assertTrue(result["stop_required"])
        self.assertFalse(result["poses"])

    def test_lost_or_held_target_is_not_pursued(self):
        frame = object_frame()
        for changes in ({"phase": "lost"}, {"association_fraction": .5}, {"held": True}, {"attached": True}, {"phase": "tracking", "target_id": ""}):
            item = dict(target(frame), **changes)
            result = self.nav.target_step(frame, item, now=10.1)
            self.assertTrue(result["stop_required"])
            self.assertFalse(result["poses"])

    def test_target_roi_requires_exact_aligned_frame(self):
        frame = object_frame()
        for changes in ({"frame_id": "another-frame"}, {"image_stamp": 9.9}, {"image_size": [320, 240]}, {"bbox": [1, 1, 10, 10]}):
            result = self.nav.target_step(frame, dict(target(frame), **changes), now=10.1)
            self.assertTrue(result["stop_required"])
            self.assertFalse(result["poses"])

    def test_inconsistent_object_depth_stops(self):
        frame = object_frame()
        depth = frame.depth.copy()
        depth[233:248, 313:320] = 2.5
        result = self.nav.target_step(replace(frame, depth=depth), target(frame), now=10.1)
        self.assertEqual(result["state"], "blocked")

    def test_extreme_limits_and_standoff_are_rejected(self):
        with self.assertRaises(ValueError):
            VisualNavigation(NavigationLimits(max_step_m=10))
        frame = object_frame()
        self.assertEqual(self.nav.target_step(frame, target(frame), stand_off_m=.1, now=10.1)["state"], "blocked")
        self.assertEqual(self.nav.line_step(snapshot(), "infrared", now=10.1)["state"], "blocked")

    def test_declared_endpoint_belongs_to_current_map(self):
        frame = snapshot()
        point = {"frame_id": "map", "map_epoch": frame.map_epoch, "x": 10., "y": 20.}
        self.assertEqual(self.nav.line_step(frame, "blue", endpoint_map=point, now=10.1)["state"], "reached")
        self.assertEqual(self.nav.line_step(frame, "blue", endpoint_map=dict(point, map_epoch="old-map"), now=10.1)["state"], "blocked")
        self.assertEqual(self.nav.line_step(frame, "blue", endpoint_map=[], now=10.1)["state"], "blocked")

    def test_malformed_intrinsics_and_association_are_rejected(self):
        frame = object_frame()
        k = frame.k.copy()
        k[2, 2] = 0
        self.assertEqual(self.nav.target_step(replace(frame, k=k), target(frame), now=10.1)["state"], "blocked")
        self.assertEqual(self.nav.target_step(frame, dict(target(frame), association_fraction=2), now=10.1)["state"], "lost")


if __name__ == "__main__":
    unittest.main()
