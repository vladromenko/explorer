import unittest
import numpy as np
from timed_trajectory import Execution, Limits, TimedPath


class TimedTrajectoryTests(unittest.TestCase):
    def path(self, checker=lambda a, b: True):
        return TimedPath(['j1'], [0, 1, 2], [[0], [.2], [.4]], [[0], [.2], [0]], [[0], [0], [0]],
                         Limits(np.array([-1]), np.array([1]), np.array([.3]), np.array([.5]), np.array([1.])), checker)

    def test_waypoint_has_nonzero_velocity_and_fractional_positions(self):
        p = self.path()
        middle = p.sample(p.times[1])
        self.assertAlmostEqual(middle['position'][0], .2)
        self.assertGreater(middle['velocity'][0], 0)
        self.assertGreater(p.duration, 2)
        for t in np.linspace(0, p.duration, 501):
            s = p.sample(t)
            self.assertLessEqual(abs(s['velocity'][0]), .3 + 1e-8)
            self.assertLessEqual(abs(s['acceleration'][0]), .5 + 1e-8)
            self.assertLessEqual(abs(s['jerk'][0]), 1 + 1e-8)

    def test_complete_path_collision_check_not_goal_shortcut(self):
        visited = []
        def check(a, b):
            visited.append(b[0])
            return not .19 < b[0] < .21
        with self.assertRaises(ValueError):
            self.path(check)
        self.assertGreater(len(visited), 5)

    def test_ack_without_feedback_cannot_reach(self):
        path = self.path()
        clock = [0.]
        sent, cancelled = [], []
        def feedback():
            return dict(position=[0], acquired_monotonic=[clock[0]], valid=[True])
        execution = Execution(path, lambda *a: sent.append(a), lambda: cancelled.append(True), feedback, period_s=.02)
        for t in np.arange(0, path.duration + 3.1, .02):
            clock[0] = float(t)
            execution.tick(clock[0])
        self.assertEqual(execution.state, 'fault')
        self.assertTrue(cancelled)

    def test_fresh_measured_settle_reaches_and_cancel_invalidates(self):
        path = self.path()
        clock = [0.]
        def feedback():
            return dict(position=path.sample(clock[0])['position'], acquired_monotonic=[clock[0]], valid=[True])
        sent = []
        execution = Execution(path, lambda *a: sent.append(a), lambda: None, feedback, period_s=.02)
        for t in np.arange(0, path.duration + 1, .02):
            clock[0] = float(t)
            execution.tick(clock[0])
        self.assertEqual(execution.state, 'reached')
        execution.cancel()
        count = len(sent)
        execution.tick(clock[0] + 1)
        self.assertEqual(execution.state, 'cancelled')
        self.assertEqual(count, len(sent))

    def test_stale_feedback_and_delayed_tick_stop(self):
        path = self.path()
        execution = Execution(path, lambda *a: None, lambda: None,
                              lambda: dict(position=[0], acquired_monotonic=[0], valid=[True]), period_s=.02)
        self.assertEqual(execution.tick(0), 'sent')
        self.assertEqual(execution.tick(.3), 'fault')


if __name__ == '__main__':
    unittest.main()
