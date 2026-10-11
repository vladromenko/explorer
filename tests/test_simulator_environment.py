"""Observation boundaries and actual simulated time, independent of hardware."""
import unittest
from unittest.mock import MagicMock

try:
    import mujoco
    import numpy as np
    from simulator.environment import ExplorerEnv
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Install the separate Mac simulator runtime") from exc


class SimulatorEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = ExplorerEnv(render=False)

    def tearDown(self):
        self.env.close()

    def test_policy_inputs_exclude_object_truth_and_use_command_estimate(self):
        observation, state = self.env.reset(seed=4)
        self.assertEqual(set(observation), {"state", "wrist"})
        self.assertEqual(observation["state"].shape, (9,))
        self.assertEqual(observation["wrist"].shape, (240, 320, 3))
        np.testing.assert_allclose(observation["state"], state["command"])
        self.assertEqual(state["joint_state_source"], "command_estimate")
        self.assertFalse(state["provenance"]["real_hardware_accepted"])

    def test_episode_success_requires_lift_then_release_in_basket(self):
        self.env.bin_position[:2] = self.env._sock_points()[:, :2].mean(axis=0)
        self.env._evaluate_contact(10.)
        self.assertFalse(self.env.ever_lifted)
        self.assertFalse(self.env.task_success)

    def test_variation_is_reproducible_without_changing_contact_parameters(self):
        self.env.reset(seed=23, options={"difficulty": "varied"})
        first = self.env.sock_position.copy()
        color = self.env.model.flex_rgba.copy()
        self.env.reset(seed=23, options={"difficulty": "varied"})
        np.testing.assert_allclose(first, self.env.sock_position)
        np.testing.assert_allclose(color, self.env.model.flex_rgba)
        self.env.reset(seed=24, options={"difficulty": "varied"})
        self.assertFalse(np.allclose(first, self.env.sock_position))

    def test_recorded_clock_advances_by_physics_period_and_action_is_bounded(self):
        before = self.env.capture_sample()
        goal = self.env.command.copy()
        goal[0] += 30.
        self.env.step(goal)
        after = self.env.capture_sample()
        self.assertAlmostEqual(after["simulation_time"]-before["simulation_time"], .05)
        self.assertLess(self.env.command[0], goal[0])
        self.assertTrue(np.isfinite(self.env.data.qpos).all())
        with self.assertRaises(ValueError):
            self.env.step([float("nan")]*9)

    def test_cloth_is_rendered_as_faces_and_two_lidars_have_separate_origins(self):
        self.assertTrue(self.env.render_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE])
        self.assertFalse(self.env.render_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE])
        scans = [self.env.lidar_scan(i, rays=16) for i in (0, 1)]
        self.assertEqual(len(scans[0]["ranges"]), 16)
        self.assertNotEqual(scans[0]["frame"], scans[1]["frame"])

    def test_issued_endpoint_stream_is_not_filtered_twice(self):
        teacher = ExplorerEnv(render=False)
        try:
            goal = teacher.command.copy()
            goal[0] += 12.
            for _ in range(24):
                teacher.step(goal)
                self.env.step(teacher.command.copy(), applied=True)
                np.testing.assert_allclose(self.env.command, teacher.command, atol=1e-9)
            # A proposed jump still cannot exceed the finite endpoint speed.
            previous = self.env.command.copy()
            goal[0] = 180.
            self.env.step(goal, applied=True)
            self.assertLessEqual(self.env.command[0]-previous[0], self.env.velocity_limit[0]*.05+1e-9)
            self.assertTrue(np.isfinite(self.env.data.qpos).all())
        finally:
            teacher.close()

    def test_rgb_reuse_never_carries_a_frame_across_physics_or_reset(self):
        renderer = MagicMock()
        renderer.render.return_value = np.zeros((240, 320, 3), np.uint8)
        self.env.renderer = renderer
        first = self.env.render_rgb()
        first[:] = 255
        self.assertEqual(int(self.env.render_rgb().max()), 0)
        self.assertEqual(renderer.render.call_count, 1)
        self.env.step(self.env.command.copy())
        self.env.render_rgb()
        self.assertEqual(renderer.render.call_count, 2)
        self.env.reset(seed=5)
        renderer.render.return_value[:] = 123
        self.env.renderer = renderer
        self.assertEqual(int(self.env.render_rgb().min()), 123)
        self.assertEqual(renderer.render.call_count, 3)


if __name__ == "__main__":
    unittest.main()
