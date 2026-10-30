import unittest
from safety import SafetyGate


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.g = SafetyGate(dict(base_commissioned=True, mcu_watchdog_verified=True,
                                 max_velocity=[.3, .3, .7], max_acceleration=[1, 1, 1]))

    def test_boot_is_stopped(self):
        self.g.submit([.3, 0, 0], 'manual', 1)
        self.assertEqual(self.g.tick(1, .1, True)[0], [0, 0, 0])

    def test_only_manual_commands_are_accepted(self):
        for source in ('unknown', 'remote', 'automatic'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.g.submit([.2, 0, 0], source, 1)

    def test_deadman_and_fault_require_a_new_command(self):
        self.g.estop = False
        self.g.submit([.3, 0, 0], 'manual', 1)
        self.assertGreater(self.g.tick(1, .1, True)[0][0], 0)
        self.assertEqual(self.g.tick(1.1, .1, False)[0], [0, 0, 0])
        self.assertEqual(self.g.tick(1.2, .1, True)[0], [0, 0, 0])
        self.g.submit([.3, 0, 0], 'manual', 2)
        self.assertEqual(self.g.tick(2.251, .1, True)[0], [0, 0, 0])

    def test_link_loss_latches_stop_and_recovery_does_not_rearm(self):
        self.g.estop = False
        self.g.submit([.2, 0, 0], 'manual', 1)
        self.g.observe_controller_link(False)
        self.assertTrue(self.g.estop)
        self.assertEqual(self.g.output, [0, 0, 0])
        self.g.observe_controller_link(True)
        self.g.submit([.2, 0, 0], 'manual', 1.1)
        self.assertEqual(self.g.tick(1.1, .1, True)[0], [0, 0, 0])

    def test_limits_and_input_validation(self):
        with self.assertRaises(ValueError):
            self.g.submit([float('nan'), 0, 0], 'manual', 0)
        self.g.estop = False
        self.g.submit([3, -3, 7], 'manual', 0)
        self.assertEqual(self.g.command, [.3, -.3, .7])


if __name__ == '__main__':
    unittest.main()
