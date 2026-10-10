import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from delivery_robot import DeliveryRobot
from factory_delivery import ARTIFACTS, SCOPES, factory_reference, load_factory_settings, next_exercises, places_digest


class FactoryDeliveryTests(unittest.TestCase):
    def fixture(self, directory):
        """Synthetic contract fixtures, never installed as physical acceptance."""
        root = Path(directory)
        (root / "config").mkdir()
        (root / "data").mkdir()
        write = lambda name, value: (root / name).write_text(json.dumps(value))
        digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        write("data/camera-validation.json", {"outcome": "passed"})
        write("data/gripper-validation.json", {"outcome": "passed"})
        config = {"transport_deg": [90]*6, "search_deg": [90]*6, "search_places": ["sock"], "destination": "basket",
                  "map_epoch": "m", "place_poses": {"sock": {"x": 0, "y": 0, "yaw": 0}, "basket": {"x": 1, "y": 0, "yaw": 0}},
                  "floor_plane_base": [0,0,1,0], "grasp_quaternion_xyzw": [0,0,0,1], "approach_height_m": .04,
                  "lift_height_m": .08, "grasp_tcp_offset_m": 0, "gripper_linkage_rad": -.5, "guarded_closure_enabled": True,
                  "drop_zone": {"center_xyz": [.3,0,.02], "radius_m": .1, "support_tolerance_m": .01}}
        handeye = {"execution_authorized": True, "joint_state_source": "command_estimate", "reference_mount": "arm4",
                   "camera_to_mount_reference": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                   "physical_validation_record": "data/camera-validation.json", "physical_validation_sha256": digest("data/camera-validation.json")}
        gripper = {"execution_authorized": True, "aperture_mm_calibrated": True, "open_deg": 30, "sock_close_deg": 160,
                   "physical_validation_record": "data/gripper-validation.json", "evidence_sha256": digest("data/gripper-validation.json")}
        for name in ARTIFACTS:
            write(name, config if name.endswith("delivery.json") else handeye if "handeye" in name else gripper if "gripper" in name else {"synthetic_contract_fixture": True})
        artifacts = {name: digest(name) for name in ARTIFACTS}
        places_sha = places_digest(config["place_poses"])
        write("data/observed.json", {"synthetic_contract_fixture": True, "actual_frame_id": "f1"})
        records = {}
        for kind in SCOPES:
            name = "data/" + kind + ".json"
            write(name, {"at": time.time()-.1, "kind": kind, "hardware_executed": True, "simulation": False,
                         "operator_observed": True, "outcome": "passed", "joint_state_source": "command_estimate",
                         "controller_protocol": "factory_micro_ros", "map_epoch": "m", "places_sha256": places_sha,
                         "artifacts": artifacts, "observation_artifacts": {"data/observed.json": digest("data/observed.json")}})
            records[name] = digest(name)
        evidence = {"scope": "factory_delivery_prerequisites", "joint_state_source": "command_estimate",
                    "controller_protocol": "factory_micro_ros", "map_epoch": "m", "places_sha256": places_sha,
                    "artifacts": artifacts, "physical_test_records": records}
        write("config/factory-delivery-acceptance.json", evidence)
        return root, evidence

    def test_factory_scoped_contract_needs_no_native_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            result = load_factory_settings(root)
            self.assertEqual(result["destination"], "basket")
            self.assertEqual(result["joint_state_source"], "command_estimate")
            self.assertFalse(result["physical_delivery_verified"])
            self.assertFalse((root/"config/controller-profile.json").exists())

    def test_changed_calibration_places_or_physical_logs_are_rejected(self):
        for name in ("config/explorer.urdf", "config/arm-motion.json", "config/delivery.json", "data/factory_arm.json", "data/observed.json"):
            with tempfile.TemporaryDirectory() as directory:
                root, _ = self.fixture(directory)
                (root/name).write_text("changed")
                with self.assertRaises(ValueError):
                    load_factory_settings(root)

    def test_force_accept_flags_do_not_replace_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root, evidence = self.fixture(directory)
            evidence.update(accepted=True, hardware_accepted=True, physical_test_records={})
            (root/"config/factory-delivery-acceptance.json").write_text(json.dumps(evidence))
            with self.assertRaises(ValueError):
                load_factory_settings(root)
            self.assertFalse(next_exercises(root)["physical_delivery_verified"])

    def test_simulated_or_unobserved_record_does_not_admit(self):
        for change in ({"simulation": True}, {"operator_observed": False}, {"artifacts": {}}, {"observation_artifacts": {}}):
            with tempfile.TemporaryDirectory() as directory:
                root, evidence = self.fixture(directory)
                path = root/"data/factory_arm.json"
                row = json.loads(path.read_text());row.update(change);path.write_text(json.dumps(row))
                evidence["physical_test_records"]["data/factory_arm.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
                (root/"config/factory-delivery-acceptance.json").write_text(json.dumps(evidence))
                with self.assertRaises(ValueError):
                    load_factory_settings(root)

    def test_scope_records_do_not_require_irrelevant_controller_source_or_map(self):
        with tempfile.TemporaryDirectory() as directory:
            from factory_delivery import SCOPE_ARTIFACTS
            root,evidence=self.fixture(directory)
            for name in evidence["physical_test_records"]:
                path=root/name;record=json.loads(path.read_text());kind=record["kind"]
                record["artifacts"]={key:record["artifacts"][key] for key in SCOPE_ARTIFACTS[kind]}
                if kind not in ("localization","camera_geometry"):
                    record.pop("map_epoch");record.pop("places_sha256")
                path.write_text(json.dumps(record))
                evidence["physical_test_records"][name]=hashlib.sha256(path.read_bytes()).hexdigest()
            (root/"config/factory-delivery-acceptance.json").write_text(json.dumps(evidence))
            self.assertEqual(load_factory_settings(root)["map_epoch"],"m")

    def arm_record(self, root, pose=None, phase="command_elapsed_observation_required"):
        row = {"servo_deg": pose or [90]*6, "boot_id": "boot", "phase": phase, "measured": False,
               "at": time.time()-1, "runtime_ms": 100, "ends_monotonic": time.monotonic()-1, "command_completed": True,
               "source": "timed_factory_command_estimate"}
        (root/"data/arm-state.json").write_text(json.dumps(row))
        return row

    def test_factory_permit_while_base_moves_and_same_goal_no_native_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            self.arm_record(root)
            arm = SimpleNamespace(factory_timed=True, boot="boot", status=lambda: {"busy": False},
                                  reference=lambda: json.loads((root/"data/arm-state.json").read_text()))
            permits = []
            missions = SimpleNamespace(survey_permit=lambda mid: permits.append(mid))
            trajectory = SimpleNamespace(status=lambda: {"busy": False})
            robot = DeliveryRobot(root, missions, arm, trajectory, None, None, None)
            robot.mid = "m";robot.boot = "boot";robot.hold = lambda: None
            robot.permit("m")
            result = robot._move([90.]*6)
            self.assertFalse(result["measured"]);self.assertFalse(result["attained"])
            self.assertTrue(result["command_completed"])
            self.assertTrue(permits)
            with self.assertRaises(ValueError):
                factory_reference(root, arm, "old-boot")

    def test_factory_finite_path_is_integer_and_command_estimated(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            self.arm_record(root)
            arm = SimpleNamespace(factory_timed=True, boot="boot", status=lambda: {"busy": False},
                                  reference=lambda: json.loads((root/"data/arm-state.json").read_text()))
            planned = []
            def plan(goal):
                planned.append(goal)
                return {"plan_id": "path"}
            def start(*args):
                self.arm_record(root, planned[-1])
                return {"session": "s"}
            trajectory = SimpleNamespace(plan=plan, start_local=start,
                status=lambda: {"busy": False, "phase": "command_completed", "command_completed": True, "session": "s"})
            robot = DeliveryRobot(root, None, arm, trajectory, None, None, None)
            robot.mid="m";robot.boot="boot";robot.permit=lambda mid: None;robot.hold=lambda: None
            result = robot._move([91.2]*6)
            self.assertEqual(planned, [[91]*6])
            self.assertFalse(result["attained"]);self.assertFalse(result["measured"])
            self.assertEqual(result["q_estimated_deg"], [91]*6)

    def test_guarded_final_angle_survives_lift_transport_and_support(self):
        robot = DeliveryRobot(".", None, None, None, None, SimpleNamespace(observe=lambda *args, **kwargs: []), None)
        robot.settings={"open_deg": 30, "close_deg": 160, "guarded_closure_enabled": True, "lift_height_m": .08,
                        "transport_deg": [90]*6, "drop_zone": {"center_xyz": [.3,0,.02]}, "object_kind": "soft"}
        robot.grasp_xyz=[.3,0,.02];robot.mid="m";robot.permit=lambda mid: None
        commands=[];robot._xyz=lambda point, angle: commands.append((point, angle));robot._move=lambda goal: commands.append(goal)
        with patch("delivery_robot.guarded_closure", side_effect=[{"action": "close", "step_deg": 1}, {"action": "hold"}]):
            result=robot.grasp(None)
        self.assertEqual(result["final_deg"], 31)
        robot.lift();robot.transport()
        self.assertEqual(commands[-2][1], 31);self.assertEqual(commands[-1][-1], 31)
        robot.destination_pose={"x": 0,"y": 0,"yaw": 0};robot.hold=lambda: None
        robot.missions=SimpleNamespace(maps=SimpleNamespace(pose=lambda: {"x": 0,"y": 0,"yaw": 0}))
        robot.support();self.assertEqual(commands[-1][1],31)
        with patch("delivery_robot.guarded_closure", return_value={"action": "stop", "reason": "deformation"}):
            with self.assertRaises(ValueError):robot.grasp(None)
        with self.assertRaises(ValueError):robot.lift()

    def test_final_visual_alignment_uses_same_tcp_offset_as_corrections(self):
        positions=iter([[.3,.01,.03],[.31,.01,.03],[.32,.01,.03],[.32,.01,.03]])
        robot=DeliveryRobot(".",None,None,None,None,SimpleNamespace(latest=lambda: {"object_xyz":next(positions)}),None)
        robot.settings={"grasp_tcp_offset_m":.02,"approach_height_m":.04,"open_deg":30}
        robot.mid="m";robot.permit=lambda mid:None;robot._xyz=lambda *args:{"command_completed":True}
        robot.grasp_xyz=[.3,0,.05]
        result=robot.align(None)
        self.assertTrue(result["aligned"])
        self.assertAlmostEqual(result["error_m"],0.)


if __name__ == "__main__":
    unittest.main()
