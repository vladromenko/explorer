"""A lost view must not become a lift or an automatic gripper release."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from local_grasp_experiment import LocalGraspExperiment


class LocalGraspSafetyTests(unittest.TestCase):
    def test_camera_can_wait_only_during_non_actuating_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.experiment()
            experiment.root = Path(folder)
            (experiment.root / "data").mkdir()
            (experiment.root / "data/status.json").write_text(json.dumps({
                "stop_latched": True, "velocity": [0, 0, 0]}))
            experiment.cancelled = SimpleNamespace(is_set=lambda: False)
            experiment.readiness = SimpleNamespace(status=lambda: {"capabilities": [{
                "id": "LEARN_GRASP_LOCAL", "runtime_missing": ["camera"],
                "evidence_missing": [], "permission_missing": []}]})
            experiment.trajectory.status.return_value = {"busy": False}
            experiment.state = {"phase": "detecting", "physical_execution": False}
            permit = LocalGraspExperiment.permit.__get__(experiment)
            permit(waiting_for_inference=True)
            with self.assertRaisesRegex(ValueError, "camera"):
                permit()
            experiment.state["physical_execution"] = True
            with self.assertRaisesRegex(ValueError, "camera"):
                permit(waiting_for_inference=True)

    def test_own_finite_motion_does_not_invalidate_its_initial_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.experiment()
            experiment.root = Path(folder)
            (experiment.root / "data").mkdir()
            state = {"at": time.time(), "phase": "command_in_progress",
                "source": "timed_factory_command_estimate", "boot_id": "current"}
            path = experiment.root / "data/arm-state.json"
            path.write_text(json.dumps(state))
            experiment.trajectory.status.return_value = {"busy": True, "execution_mode": "local_mission"}
            experiment.trajectory.local_permit = experiment.motion_permit
            self.assertTrue(experiment.owns_moving_arm({"boot_id": "current"}))
            self.assertFalse(experiment.owns_moving_arm({"boot_id": "other"}))
            fault = experiment.root / "data/arm-telemetry-fault.json"
            fault.write_text(json.dumps({"at": time.time()}))
            self.assertFalse(experiment.owns_moving_arm({"boot_id": "current"}))
            fault.unlink()
            experiment.trajectory.local_permit = lambda: None
            self.assertFalse(experiment.owns_moving_arm({"boot_id": "current"}))

    def experiment(self):
        experiment = LocalGraspExperiment.__new__(LocalGraspExperiment)
        experiment.permit = Mock()
        experiment.vision = SimpleNamespace(tracking_status=Mock(), observe=Mock(),
            preview_visibility=Mock(return_value={"observable": True}))
        experiment.arm = SimpleNamespace(reference=Mock(return_value={"servo_deg": [90] * 6}))
        experiment.model = Mock(return_value=SimpleNamespace(grasp_orientation=lambda point, quaternion: list(quaternion),
            ik=Mock(return_value={
            "solved": True, "collision": False, "servo_deg": [95, 90, 90, 90, 90]})))
        experiment.trajectory = SimpleNamespace(plan=Mock(return_value={"plan_id": "planned"}),
            start_local=Mock(), status=Mock(), stop=Mock())
        experiment.state = {}
        return experiment

    def test_lost_tracking_after_planning_prevents_execution(self):
        experiment = self.experiment()
        experiment.vision.tracking_status.side_effect = [None, ValueError("Object association lost")]
        with self.assertRaisesRegex(ValueError, "association lost"):
            experiment.move([0.3, 0, 0.08], 90, {"gripper_linkage_rad": -0.8,
                "grasp_quaternion_xyzw": [0, 0, 0, 1]})
        experiment.trajectory.plan.assert_called_once()
        experiment.trajectory.start_local.assert_not_called()

    def test_unreachable_contact_is_rejected_before_any_approach(self):
        experiment = self.experiment()
        model = experiment.model()
        model.ik.side_effect = [
            {"solved": True, "collision": False, "servo_deg": [90] * 5},
            {"solved": False, "collision": False, "servo_deg": [90] * 5}] * 3
        with self.assertRaisesRegex(ValueError, "approach/contact/lift"):
            experiment.candidates({"object_xyz": [.25, 0, .025], "object_position_uncertainty_m": .005},
                {"grasp_tcp_offset_m": 0, "approach_height_m": .055, "lift_height_m": .065,
                    "gripper_linkage_rad": -.8, "grasp_quaternion_xyzw": [0,0,0,1]}, 1)
        self.assertEqual(model.ik.call_count, 6)
        experiment.trajectory.plan.assert_not_called()
        self.assertEqual(len(experiment.state["target_preflight"]["checks"]), 3)

    def test_executor_callback_requires_fresh_image_association(self):
        experiment = self.experiment()
        experiment.vision.tracking_status.side_effect = ValueError("No fresh image-space object association")
        with self.assertRaisesRegex(ValueError, "No fresh"):
            experiment.motion_permit()
        experiment.permit.assert_called_once()

    def test_reachable_but_unobservable_contact_is_rejected_before_motion(self):
        experiment = self.experiment()
        experiment.vision.preview_visibility.side_effect = [
            {"observable": True}, {"observable": False}, {"observable": True}] * 3
        with self.assertRaisesRegex(ValueError, "view_unavailable"):
            experiment.candidates({"object_xyz": [.25, 0, .025], "object_position_uncertainty_m": .005},
                {"grasp_tcp_offset_m": 0, "approach_height_m": .055, "lift_height_m": .065,
                    "gripper_linkage_rad": -.8, "grasp_quaternion_xyzw": [0,0,0,1]}, 1)
        experiment.trajectory.plan.assert_not_called()
        self.assertEqual(len(experiment.state["target_preflight"]["checks"][0]["visibility"]), 3)
        action = experiment.state["next_action"]
        self.assertFalse(action["motion_required"])
        self.assertFalse(action["execution_permission_granted"])
        self.assertEqual(action["code"], "check_contact_camera_view")

    def test_guarded_stop_never_becomes_a_hold_and_lift(self):
        experiment = self.experiment()
        experiment.move = Mock()
        experiment.vision.observe.return_value = [{}, {}]
        with patch("local_grasp_experiment.guarded_closure", return_value={
                "action": "stop", "reason": "object_tracking_lost"}):
            with self.assertRaisesRegex(ValueError, "object_tracking_lost"):
                experiment.close_guarded([0.3, 0, 0.04], {"open_deg": 30, "close_deg": 160})
        experiment.move.assert_called_once()

    def trial(self, outcome):
        experiment = self.experiment()
        settings = {"open_deg": 30, "approach_height_m": 0.055, "lift_height_m": 0.065,
            "floor_plane_base": [0, 0, 1, 0]}
        experiment.settings = Mock(return_value=settings)
        experiment.detect = Mock(return_value=({"object_id": "sock", "at": 100}, settings))
        experiment.candidates = Mock(return_value=({"policy_version": "geometry-baseline"}, [0.3, 0, 0.04]))
        experiment.move = Mock()
        experiment.close_guarded = Mock(return_value={"final_deg": 120})
        experiment.vision.observe.return_value = [{"object_id": "sock"}] * 3
        experiment.workflows = SimpleNamespace(begin_autonomous_episode=Mock(return_value={"id": "episode"}),
            episodes=SimpleNamespace(append=Mock()), finish_autonomous_episode=Mock(return_value={"state": "complete"}),
            needs_reset=Mock(return_value={"state": "needs_reset"}))
        experiment.learn = Mock()
        with patch("local_grasp_experiment.verify_lift", return_value={"outcome": outcome,
                "evidence": {"reason": "visual result"}}):
            result = experiment.trial({"id": "workflow", "target": "sock", "autonomous_done": 0})
        return experiment, result

    def test_unknown_lift_preserves_grip_and_does_not_repeat(self):
        experiment, result = self.trial("unknown")
        self.assertEqual(experiment.move.call_count, 3)  # Approach, contact, initial lift only.
        self.assertEqual(result["state"], "needs_reset")
        evidence = experiment.workflows.finish_autonomous_episode.call_args.args[2]
        self.assertEqual(evidence["reset"]["state"], "unknown")
        experiment.workflows.needs_reset.assert_called_once()

    def test_observed_failure_keeps_existing_withdrawal_behavior(self):
        experiment, result = self.trial("failure")
        self.assertEqual(experiment.move.call_count, 4)
        self.assertEqual(experiment.move.call_args.args[1], 30)
        experiment.workflows.needs_reset.assert_not_called()
        self.assertEqual(result["state"], "complete")


if __name__ == "__main__":
    unittest.main()
