from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_navigation_adapter import VisualNavigationAdapter, VisualInterruption
from visual_navigation import RGBDFrame
from test_visual_navigation import object_frame, snapshot, target


class FakeClock:
    def __init__(self):
        self.value = 10.1

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class FakeContext:
    id = "own-task"
    observing = True
    deadline = 50.

    def __init__(self):
        self.events = []
        self.cancelled = False

    def permit(self):
        if self.cancelled:
            raise InterruptedError("Manual takeover")

    def event(self, phase, details):
        self.permit()
        self.events.append((phase, details))


class FakePorts:
    def __init__(self):
        self.calls = []
        self.cancelled = []
        self.hook = None

    def navigation_skill(self, spec, context):
        self.calls.append(spec)
        try:
            if self.hook:
                self.hook(context)
            context.permit()
            return {"state": "succeeded", "evidence": {"pose_reached": True}}
        except Exception:
            self.cancel(context.id)
            raise

    def cancel(self, identifier):
        self.cancelled.append(identifier)


class VisualNavigationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data/handeye-validations/test").mkdir(parents=True)
        (self.root / "config").mkdir()
        record = b"physical-validation-record"
        (self.root / "data/handeye-validations/test/result.json").write_bytes(record)
        self.accepted = {"execution_authorized": True, "reference_mount": "arm4", "camera_frame": "camera_color_optical_frame",
            "physical_validation_record": "data/handeye-validations/test/result.json", "physical_validation_sha256": hashlib.sha256(record).hexdigest()}
        self.write_json("config/handeye-accepted.json", self.accepted)
        self.frame = object_frame()
        self.view = {"phase": "ready", "view": "forward", "servo_deg": [90, 90, 60, 15, 90, 100], "settled_at": 9.}
        self.arm = {"phase": "command_elapsed_observation_required", "servo_deg": self.view["servo_deg"], "updated_at": 9.8, "at": 9.8, "boot_id": "current-boot"}
        self.write_json("data/arm-state.json", self.arm)
        self.maps_pose = {"x": 10., "y": 20., "yaw": 0., "frame": "map", "age_s": .1, "localization_verified": True, "provisional": False}
        self.epoch = "map-epoch-1"
        self.tracker = target(self.frame)
        # Existing tracker has an image timestamp, not the NPZ's content hash ID.
        self.tracker.pop("frame_id")
        self.clock = FakeClock()
        self.preview_hook = None
        self.maps = SimpleNamespace(pose=lambda: dict(self.maps_pose), epoch=lambda: self.epoch, preview=self.preview)
        self.views = SimpleNamespace(status=lambda: dict(self.view), arm=SimpleNamespace(boot="current-boot"), geometry=lambda pose: {"camera_to_base_estimate": self.frame.camera_to_base.tolist(), "handeye_source": self.accepted["physical_validation_sha256"]})
        self.vision = SimpleNamespace(tick=lambda: dict(self.tracker))
        self.ports = FakePorts()
        self.context = FakeContext()
        self.adapter = VisualNavigationAdapter(self.root, self.vision, self.maps, object(), self.views, self.ports,
                                               clock=self.clock, monotonic=self.clock, sleeper=self.clock.sleep)
        self.write_frame(self.frame)

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def write_frame(self, frame, depth_stamp=True):
        values = {key: getattr(frame, key) for key in ("rgb", "depth", "k", "d")}
        values.update(stamp=frame.image_stamp, frame=frame.camera_frame)
        if depth_stamp:
            values["depth_stamp"] = frame.depth_stamp
        np.savez_compressed(self.root / "data/rgbd-snapshot.npz", **values)

    def preview(self, x, y):
        if self.preview_hook:
            self.preview_hook()
        return {"safe": True}

    def execute(self, **args):
        return self.adapter.follow_target({"target_id": self.tracker["target_id"], "duration_s": 1, **args}, self.context)

    def test_real_snapshot_adapter_freezes_accepted_stationary_geometry(self):
        frame = self.adapter.snapshot()
        self.assertIsInstance(frame, RGBDFrame)
        self.assertTrue(frame.pose_verified)
        self.assertTrue(frame.transform_accepted)
        self.assertEqual(frame.pose["frame_id"], "map")
        self.assertTrue(frame.frame_id.startswith("rgbd:"))
        self.assertFalse(frame.depth.flags.writeable)
        self.assertEqual(frame.depth_stamp, 10.)

    def test_provisional_localization_rejects_before_any_navigation(self):
        self.maps_pose.update(provisional=True, localization_verified=False)
        result = self.execute()
        self.assertEqual(result["state"], "failed")
        self.assertIn("provisional", result["evidence"]["reason"])
        self.assertFalse(self.ports.calls)

    def test_missing_depth_capture_time_is_not_fabricated(self):
        self.write_frame(self.frame, depth_stamp=False)
        result = self.execute()
        self.assertIn("depth_stamp", result["evidence"]["reason"])
        self.assertFalse(self.ports.calls)

    def test_tampered_handeye_acceptance_record_is_rejected(self):
        (self.root / self.accepted["physical_validation_record"]).write_text("altered")
        self.assertIn("изменилась", self.execute()["evidence"]["reason"])
        self.assertFalse(self.ports.calls)

    def test_arm_motion_wrong_boot_and_capture_before_settle_are_rejected(self):
        for change in ({"phase": "moving"}, {"boot_id": "old-boot"}, {"updated_at": 10.2}):
            self.write_json("data/arm-state.json", {**self.arm, **change})
            self.assertEqual(self.execute()["state"], "failed")
        self.write_json("data/arm-state.json", self.arm)
        self.view["settled_at"] = 10.2
        self.assertEqual(self.execute()["state"], "failed")
        self.assertFalse(self.ports.calls)

    def test_measured_reached_target_does_not_start_nav2(self):
        self.frame = object_frame(.68)
        self.write_frame(self.frame)
        result = self.execute()
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["evidence"]["completion"], "measured_target_distance")
        self.assertFalse(self.ports.calls)

    def test_preview_expiry_cannot_dispatch_stale_goal(self):
        self.preview_hook = lambda: self.clock.sleep(.6)
        result = self.execute()
        self.assertEqual(result["state"], "failed")
        self.assertIn("устарело", result["evidence"]["reason"])
        self.assertFalse(self.ports.calls)

    def test_tracker_loss_during_active_navigation_cancels_only_own_task(self):
        self.ports.hook = lambda context: self.tracker.update(phase="lost")
        result = self.execute()
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["evidence"]["visual_state"], "lost")
        self.assertEqual(set(self.ports.cancelled), {self.context.id})
        self.assertEqual(len(self.ports.calls), 1)

    def test_changed_map_epoch_cancels_active_goal(self):
        def changed(context):
            self.epoch = "map-epoch-2"
        self.ports.hook = changed
        result = self.execute()
        self.assertIn("Карта", result["evidence"]["reason"])
        self.assertEqual(set(self.ports.cancelled), {self.context.id})

    def test_target_world_movement_invalidates_old_goal(self):
        def moved(context):
            self.frame = object_frame(1.8)
            self.write_frame(self.frame)
        self.ports.hook = moved
        result = self.execute()
        self.assertEqual(result["evidence"]["visual_state"], "lost")
        self.assertIn("переместился", result["evidence"]["reason"])

    def test_sensor_staleness_during_navigation_stops(self):
        self.ports.hook = lambda context: self.clock.sleep(.55)
        result = self.execute()
        self.assertEqual(result["evidence"]["visual_state"], "lost")
        self.assertTrue(self.ports.cancelled)

    def test_bounded_duration_interrupts_long_navigation(self):
        self.ports.hook = lambda context: self.clock.sleep(1.1)
        result = self.execute()
        self.assertEqual(result["state"], "waiting")
        self.assertEqual(result["evidence"]["visual_state"], "budget_exhausted")
        self.assertFalse(result["evidence"]["completed"])
        self.assertTrue(self.ports.cancelled)

    def test_cancel_and_deadline_exceptions_are_not_swallowed(self):
        def manual(context):
            self.context.cancelled = True
        self.ports.hook = manual
        with self.assertRaises(InterruptedError):
            self.execute()
        self.assertTrue(self.ports.cancelled)

    def test_duplicate_snapshot_is_not_replayed(self):
        result = self.execute()
        self.assertEqual(len(self.ports.calls), 1)
        self.assertEqual(result["state"], "failed")
        self.assertIn("stale", result["evidence"]["reason"])

    def test_nav2_failure_is_not_claimed_as_arrival(self):
        self.ports.navigation_skill = lambda spec, context: {"state": "failed", "evidence": {"reason": "Obstacle"}}
        result = self.execute()
        self.assertEqual(result["state"], "failed")
        self.assertIn("navigation", result["evidence"])

    def test_catalog_uses_common_physical_registry_and_bounded_schema(self):
        ports = self.adapter.skill_ports()
        self.assertEqual([port.name for port in ports], ["follow_line", "follow_selected_target"])
        for port in ports:
            self.assertTrue(port.physical)
            self.assertIn("base", port.resources)
            self.assertEqual(port.frame, "map")
            self.assertFalse(port.schema["additionalProperties"])
            self.assertEqual(port.schema["properties"]["duration_s"]["maximum"], 30)
            self.assertEqual(port.cancel, self.ports.cancel)

    def test_line_adapter_uses_actual_metric_floor_and_same_navigation_owner(self):
        self.frame = snapshot()
        self.write_frame(self.frame)
        result = self.adapter.follow_line({"color": "blue", "duration_s": 1}, self.context)
        self.assertEqual(len(self.ports.calls), 1)
        self.assertEqual(self.ports.calls[0]["kind"], "navigate_current")
        self.assertLessEqual(self.ports.calls[0]["x"] - self.maps_pose["x"], .400001)
        self.assertEqual(result["state"], "failed")
        self.assertFalse(result["evidence"]["completed"])

    def test_invalid_duration_rejected_without_navigation(self):
        with self.assertRaises(ValueError):
            self.execute(duration_s=31)
        self.assertFalse(self.ports.calls)


if __name__ == "__main__":
    unittest.main()
