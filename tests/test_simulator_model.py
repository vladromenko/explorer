"""Offline CAD simulator validation; dynamics tests use the Mac sim runtime."""
import json
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.explorer_model import (ARM_JOINTS, WHEEL_JOINTS, build_mjcf,
    joint_to_servo, load_model, model_config, resolve_mesh_dir, servo_to_joint, wheel_targets)

try:
    import mujoco
    import numpy as np
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "Install simulator runtime numpy and mujoco")
class SimulatorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mesh_dir = resolve_mesh_dir()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.model, cls.data = load_model(mesh_dir=cls.mesh_dir, scene="sock")

    def reset_data(self):
        data = mujoco.MjData(self.model)
        for i in range(6):
            data.ctrl[self.model.actuator(f"servo{i+1}").id] = 0
        mujoco.mj_forward(self.model, data)
        return data

    def test_endpoint_mapping_preserves_base_sign_and_gripper_direction(self):
        command = [40, 65, 120, 80, 200, 160]
        q = servo_to_joint(command)
        self.assertAlmostEqual(q[0], math.radians(50))
        self.assertAlmostEqual(q[1], math.radians(-25))
        self.assertLess(q[5], 0)
        np.testing.assert_allclose(joint_to_servo(q), command)
        for invalid in ([90]*5, [90]*5+[171], [float("nan")]*6):
            with self.assertRaises(ValueError):
                servo_to_joint(invalid)

    def test_cad_joint_origins_axes_and_mimic_constraints(self):
        source = ET.parse(ROOT / "config/explorer.urdf").getroot()
        for joint in source.findall("joint"):
            child = joint.find("child").get("link")
            if child != "base_link":
                origin = [float(x) for x in joint.find("origin").get("xyz").split()]
                np.testing.assert_allclose(self.model.body(child).pos, origin, atol=1e-10)
            if joint.get("type") == "revolute":
                expected_axis = np.array([float(x) for x in joint.find("axis").get("xyz").split()])
                np.testing.assert_allclose(self.model.joint(joint.get("name")).axis,
                    expected_axis / np.linalg.norm(expected_axis), atol=1e-9)
        self.assertGreaterEqual(self.model.neq, 6)  # five mimic joints plus cloth edges
        self.assertEqual(self.model.nu, 10)
        for name in ARM_JOINTS + WHEEL_JOINTS:
            self.assertEqual(self.model.joint(name).type, mujoco.mjtJoint.mjJNT_HINGE)

    def test_camera_follows_accepted_wrist_optical_transform(self):
        accepted = json.loads((ROOT / "config/handeye-accepted.json").read_text())
        transform = np.asarray(accepted["camera_to_mount_reference"])
        camera = self.model.camera("wrist")
        self.assertEqual(camera.bodyid, self.model.body("arm4").id)
        np.testing.assert_allclose(camera.pos, transform[:3,3], atol=1e-10)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, camera.quat)
        np.testing.assert_allclose(matrix.reshape(3,3), transform[:3,:3] @ np.diag([1,-1,-1]), atol=1e-10)
        cfg = model_config()["camera"]
        self.assertAlmostEqual(float(camera.fovy[0]), math.degrees(2*math.atan(cfg["height"]/(2*cfg["fy"]))), places=3)

    def test_two_laser_sites_match_navigation_accepted_extrinsics(self):
        cfg = json.loads((ROOT / "config/lidar_geometry.json").read_text())
        for name in ("laser0_frame", "laser1_frame"):
            np.testing.assert_allclose(self.model.body(name).pos, cfg["frames"][name], atol=1e-10)
            self.assertEqual(self.model.site(name).bodyid, self.model.body(name).id)

    def test_visual_meshes_have_no_contact_and_finite_actuator_torque(self):
        xml = ET.fromstring(build_mjcf(mesh_dir=self.mesh_dir))
        for geom in xml.findall(".//geom"):
            if geom.get("name", "").startswith("visual_"):
                self.assertEqual(geom.get("contype"), "0")
                self.assertEqual(geom.get("conaffinity"), "0")
        self.assertEqual(len(xml.findall(".//freejoint")), 1)
        self.assertFalse(xml.findall(".//weld"))
        self.assertFalse(xml.findall(".//body[@mocap='true']"))
        self.assertEqual(len([j for j in xml.findall(".//joint") if "roller" in j.get("name", "")]), 48)
        self.assertTrue(np.all(np.isfinite(self.model.actuator_forcerange)))
        self.assertTrue(np.all(self.model.actuator_forcerange[:,1] > 0))

    def test_deformable_sock_falls_and_makes_physical_floor_contacts(self):
        model, data = load_model(mesh_dir=self.mesh_dir, sock_position=(.33,0,.08))
        initial = data.flexvert_xpos.copy()
        for _ in range(350):
            mujoco.mj_step(model, data)
        self.assertEqual(model.nflex, 1)
        self.assertEqual(model.flex_dim[0], 2)
        self.assertGreater(model.nflexvert, 40)
        self.assertLess(data.flexvert_xpos[:,2].mean(), initial[:,2].mean()-.03)
        self.assertGreater(data.ncon, 0)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertFalse(np.any(data.warning.number))
        # A shell can fold; it is not one rigid mesh or a mocap attachment.
        self.assertGreater(np.ptp(data.flexvert_xpos[:,2]), .001)

    def test_driven_rolls_forward_and_strafes_via_contact(self):
        model, data = load_model(mesh_dir=self.mesh_dir, scene="empty")
        for _ in range(100):
            mujoco.mj_step(model, data)
        initial = data.qpos[:3].copy()
        data.ctrl[:4] = wheel_targets(.12, 0, 0)
        for _ in range(400):
            mujoco.mj_step(model, data)
        self.assertGreater(data.qpos[0]-initial[0], .015)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertFalse(np.any(data.warning.number))
        model, data = load_model(mesh_dir=self.mesh_dir, scene="empty")
        for _ in range(100):
            mujoco.mj_step(model, data)
        initial = data.qpos[:3].copy()
        data.ctrl[:4] = wheel_targets(0, .12, 0)
        for _ in range(400):
            mujoco.mj_step(model, data)
        self.assertGreater(data.qpos[1]-initial[1], .015)

    def test_six_servo_endpoints_move_through_finite_force_dynamics(self):
        model, data = load_model(mesh_dir=self.mesh_dir, scene="empty")
        target = servo_to_joint([60, 65, 110, 85, 130, 160])
        before = np.asarray([data.qpos[model.joint(n).qposadr[0]] for n in ARM_JOINTS])
        for i, q in enumerate(target):
            data.ctrl[model.actuator(f"servo{i+1}").id] = q
        # Setting an endpoint leaves the actual state unchanged until physics runs.
        np.testing.assert_allclose([data.qpos[model.joint(n).qposadr[0]] for n in ARM_JOINTS], before)
        for _ in range(1200):
            mujoco.mj_step(model, data)
        actual = np.asarray([data.qpos[model.joint(n).qposadr[0]] for n in ARM_JOINTS])
        np.testing.assert_allclose(actual, target, atol=math.radians(2))
        self.assertTrue(np.all(np.abs(data.actuator_force) <= model.actuator_forcerange[:,1] + 1e-8))
        self.assertFalse(np.any(data.warning.number))

    def test_wheel_drive_does_not_force_airborne_base_translation(self):
        model, data = load_model(mesh_dir=self.mesh_dir, scene="empty")
        data.qpos[2] = 1
        data.ctrl[:4] = wheel_targets(.15, .15, 0)
        mujoco.mj_forward(model, data)
        for _ in range(80):
            mujoco.mj_step(model, data)
        self.assertLess(np.linalg.norm(data.qpos[:2]), .002)
        self.assertLess(data.qpos[2], 1)


if __name__ == "__main__":
    unittest.main()
