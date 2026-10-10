import unittest
from types import SimpleNamespace
import numpy as np
from arm_velocity import VelocityGenerator, cartesian_joint_velocity


class VelocityGeneratorTests(unittest.TestCase):
    def test_held_multiaxis_keeps_velocity_without_restarting_at_rest(self):
        generator = VelocityGenerator([90] * 6)
        samples = [generator.step([24, -12, 8, 0, 0, 0]) for _ in range(40)]
        self.assertTrue(all(sample["velocity_deg_s"][0] > 23.99 for sample in samples[3:]))
        self.assertGreater(samples[-1]["q_estimated_deg"][0], 120)
        self.assertLess(samples[-1]["q_estimated_deg"][1], 75)
        self.assertTrue(all(sample["pose"][5] == 90 for sample in samples))
        self.assertTrue(all(sample["runtime_ms"] == 80 for sample in samples))

    def test_reversal_and_release_obey_acceleration_and_jerk(self):
        generator = VelocityGenerator([90] * 6)
        samples = []
        for velocity in ([24] * 6, [-24] * 6, [0] * 6):
            samples.extend(generator.step(velocity) for _ in range(15))
        velocities = np.asarray([row["velocity_deg_s"] for row in samples])
        accelerations = np.asarray([row["acceleration_deg_s2"] for row in samples])
        self.assertTrue(np.all(np.abs(velocities) <= generator.maximum + 1e-7))
        self.assertTrue(np.all(np.abs(accelerations) <= generator.acceleration + 1e-7))
        self.assertTrue(np.all(np.abs(np.diff(accelerations, axis=0)) <= generator.jerk * generator.period + 1e-6))
        self.assertTrue(samples[-1]["settled"])
        self.assertLess(np.max(np.abs(velocities[-1])), 1e-6)
        self.assertFalse(samples[-1]["measured"])

    def test_fractional_small_input_survives_until_integer_transport(self):
        generator = VelocityGenerator([90] * 6)
        samples = [generator.step([0.2, 0, 0, 0, 0, 0]) for _ in range(100)]
        self.assertGreater(samples[-1]["q_estimated_deg"][0], 90.7)
        self.assertEqual(samples[-1]["pose"][0], 91)
        self.assertTrue(all(type(row["pose"][0]) is int for row in samples))

    def test_limits_brake_without_expanding_ranges(self):
        generator = VelocityGenerator([176, 3, 90, 90, 265, 175])
        samples = [generator.step([48, -48, 0, 0, 60, 72]) for _ in range(100)]
        positions = np.asarray([row["q_estimated_deg"] for row in samples])
        self.assertTrue(np.all(positions >= generator.lower))
        self.assertTrue(np.all(positions <= generator.upper))
        self.assertTrue(samples[-1]["settled"])
        self.assertLessEqual(samples[-1]["pose"][5], 180)

    def test_invalid_input_does_not_advance_state(self):
        generator = VelocityGenerator([90] * 6)
        with self.assertRaises(ValueError):
            generator.step([float("nan")] * 6)
        self.assertEqual(generator.steps, 0)

    def test_cartesian_differential_motion_and_direct_gripper(self):
        def fk(q, shape):
            return dict(xyz=[q[0] * .001 + q[3] * .0002, q[1] * .001, q[2] * .001])
        model = SimpleNamespace(fk=fk)
        velocity = cartesian_joint_velocity(model, [90] * 6, [.01, .02, -.01], [0, 0, 0, 2, 3, 5], [48] * 6)
        self.assertGreater(velocity[0], 0)
        self.assertGreater(velocity[1], 0)
        self.assertLess(velocity[2], 0)
        self.assertEqual(velocity[3:], [2, 3, 5])
        self.assertTrue(all(abs(v) <= 48 for v in velocity))

    def test_singular_direction_rejected_without_fake_ik(self):
        model = SimpleNamespace(fk=lambda q, shape: dict(xyz=[0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "singular"):
            cartesian_joint_velocity(model, [90] * 6, [.01, 0, 0], [0] * 6, [48] * 6)


if __name__ == "__main__":
    unittest.main()
