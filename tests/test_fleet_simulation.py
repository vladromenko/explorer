import ast
import math
from pathlib import Path
import unittest

from fleet_simulation import FleetSimulation


class FleetSimulationTests(unittest.TestCase):
    def fleet(self):
        return FleetSimulation(robots=[{"robot_id": "alpha", "pose": (0., 0., 0.)},
                                       {"robot_id": "beta", "pose": (2., 0., 0.)}])

    def command(self, fleet, robot="alpha", velocity=(.5, 0., 0.), sequence=1, duration=.5):
        return fleet.submit(robot, velocity, sequence, fleet.time + duration, fleet.session,
                            fleet.robots[robot]["namespace"])

    def test_ids_namespaces_and_tf_are_disjoint(self):
        fleet = self.fleet()
        for key in ("namespace", "cmd_topic", "tf_topic", "map_frame", "base_frame"):
            self.assertNotEqual(fleet.robots["alpha"][key], fleet.robots["beta"][key])
        with self.assertRaises(ValueError):
            fleet.register("alpha", (5., 0., 0.))
        for robot_id in ("../robotio", "/explorer", "A", "a/b"):
            with self.assertRaises(ValueError):
                fleet.register(robot_id, (5., 0., 0.))
        with self.assertRaises(ValueError):
            fleet.register("gamma", (5., 0., 0.), namespace="/robotio")

    def test_cannot_submit_physical_namespace_or_inject_publisher(self):
        fleet = self.fleet()
        for namespace in ("/robotio", "/explorer", "/cmd_vel", "/simulation/other"):
            with self.assertRaises(ValueError):
                fleet.submit("alpha", (0., 0., 0.), 1, .5, fleet.session, namespace)
        with self.assertRaises(TypeError):
            FleetSimulation(publisher=lambda value: value)
        tree = ast.parse((Path(__file__).resolve().parents[1] / "src/fleet_simulation.py").read_text())
        modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertEqual(set(modules), {"copy", "math", "re", "secrets"})
        self.assertFalse(any(isinstance(node, ast.ImportFrom) for node in ast.walk(tree)))
        self.assertEqual(fleet.snapshot()["physical_publishers"], 0)

    def test_deadline_stops_and_never_replays(self):
        fleet = self.fleet()
        self.command(fleet, duration=.251)
        fleet.advance(.8)
        self.assertAlmostEqual(fleet.robots["alpha"]["pose"][0], .1255, places=9)
        self.assertFalse(fleet.commands)
        fleet.advance(.7)
        self.assertAlmostEqual(fleet.robots["alpha"]["pose"][0], .1255, places=9)
        with self.assertRaises(ValueError):
            self.command(fleet)
        self.command(fleet, sequence=2)
        fleet.cancel("alpha")
        with self.assertRaises(ValueError):
            self.command(fleet, sequence=2)

    def test_old_session_and_snapshot_cannot_restore_command(self):
        fleet = self.fleet()
        self.command(fleet)
        saved = fleet.snapshot()
        fleet.reset_session()
        with self.assertRaises(ValueError):
            fleet.submit("alpha", (.5, 0., 0.), 2, .5, saved["session"], fleet.robots["alpha"]["namespace"])
        fleet.advance(.5)
        self.assertEqual(fleet.robots["alpha"]["pose"], (0., 0., 0.))
        saved["robots"]["alpha"]["pose"] = (99., 0., 0.)
        self.assertEqual(fleet.robots["alpha"]["pose"], (0., 0., 0.))

    def test_reject_nan_excessive_speed_and_expiration(self):
        fleet = self.fleet()
        for velocity in ((float("nan"), 0., 0.), (.8, .8, 0.), (0., 0., 2.)):
            with self.assertRaises(ValueError):
                self.command(fleet, velocity=velocity)
        for duration in (0., -1., 1.01):
            with self.assertRaises(ValueError):
                self.command(fleet, duration=duration)

    def test_body_velocity_rotates_into_own_world(self):
        fleet = FleetSimulation(robots=[{"robot_id": "alpha", "pose": (0., 0., math.pi / 2.)}])
        self.command(fleet)
        fleet.advance(.5)
        self.assertAlmostEqual(fleet.robots["alpha"]["pose"][0], 0.)
        self.assertAlmostEqual(fleet.robots["alpha"]["pose"][1], .25)

    def test_swept_collision_stops_all_virtual_robots(self):
        fleet = FleetSimulation(robots=[{"robot_id": "alpha", "pose": (0., 0., 0.), "radius": .01},
                                       {"robot_id": "beta", "pose": (1., 0., 0.), "radius": .01}],
                                obstacles=[(.3, 0., .001)])
        self.command(fleet, velocity=(.8, 0., 0.), duration=1.)
        self.command(fleet, robot="beta", velocity=(-.8, 0., 0.), duration=1.)
        fleet.advance(1.)
        self.assertLess(fleet.robots["alpha"]["pose"][0], .289)
        self.assertFalse(fleet.commands)
        self.assertEqual(fleet.events[0]["reason"], "simulated_collision")

    def test_pairwise_collision_checks_relative_swept_path(self):
        fleet = FleetSimulation(robots=[{"robot_id": "alpha", "pose": (-.1, 0., 0.), "radius": .01},
                                       {"robot_id": "beta", "pose": (.1, 0., 0.), "radius": .01}])
        self.command(fleet, velocity=(.8, 0., 0.))
        self.command(fleet, robot="beta", velocity=(-.8, 0., 0.))
        fleet.advance(.5)
        self.assertGreater(fleet.robots["beta"]["pose"][0] - fleet.robots["alpha"]["pose"][0], .02)
        self.assertEqual(fleet.events[0]["robots"], ["alpha", "beta"])

    def test_exact_assignment_not_greedy_and_no_commands(self):
        fleet = self.fleet()
        result = fleet.assign({"near_beta": (1.1, 0., 0.), "near_alpha": (1., 0., 0.)})
        self.assertEqual(result["assignments"]["alpha"]["target_id"], "near_alpha")
        self.assertEqual(result["assignments"]["beta"]["target_id"], "near_beta")
        self.assertAlmostEqual(result["total_distance_m"], 1.9)
        self.assertFalse(result["route_verified"])
        self.assertFalse(fleet.commands)

    def test_formation_rotation_separation_and_obstacle_rejection(self):
        fleet = self.fleet()
        for shape in ("line", "grid", "v"):
            result = fleet.formation(shape, (4., 4., math.pi / 2.))
            poses = list(result["targets"].values())
            self.assertGreater(math.dist(poses[0][:2], poses[1][:2]), .4)
            self.assertFalse(result["execution_allowed"])
        line = fleet.formation("line", (4., 4., math.pi / 2.))["targets"]
        self.assertAlmostEqual(line["slot_0"][0], 4.4)
        fleet.obstacles.append((4.4, 4., .1))
        with self.assertRaises(ValueError):
            fleet.formation("line", (4., 4., math.pi / 2.))
        with self.assertRaises(ValueError):
            fleet.formation("line", (4., 4., 0.), spacing=.4)


if __name__ == "__main__":
    unittest.main()
