import unittest
from safety import SafetyGate


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.g = SafetyGate(dict(base_commissioned=True, mcu_watchdog_verified=True,
                                 max_velocity=[.3,.3,.7], max_acceleration=[1,1,1]))

    def test_boot_latched(self):
        self.g.submit([.3,0,0], 'manual', 1)
        self.assertEqual(self.g.tick(1,.1,True)[0], [0,0,0])

    def test_deadman_and_no_resume_after_obstacle(self):
        self.g.estop = False
        self.g.submit([.3,0,0], 'manual', 1)
        self.assertGreater(self.g.tick(1,.1,True)[0][0], 0)
        self.assertEqual(self.g.tick(1.1,.1,True,True)[0], [0,0,0])
        self.assertEqual(self.g.tick(1.2,.1,True)[0], [0,0,0])
        self.g.submit([.3,0,0], 'manual', 2)
        self.assertEqual(self.g.tick(2.251,.1,True)[0], [0,0,0])

    def test_manual_preempts_autonomy(self):
        self.g.mode = 'AUTONOMOUS'
        self.g.submit([.2,0,0], 'autonomy', 1)
        self.g.submit([0,.2,0], 'manual', 1.01)
        self.assertFalse(self.g.submit([.3,0,0], 'autonomy', 1.02))
        self.assertEqual(self.g.command, [0,.2,0])

    def test_invalid_and_uncommissioned(self):
        with self.assertRaises(ValueError):
            self.g.submit([float('nan'),0,0], 'manual', 0)
        self.g.estop=False
        self.g.config['base_commissioned']=False
        self.g.submit([.3,0,0], 'manual', 0)
        self.assertEqual(self.g.tick(0,.1,True)[0], [0,0,0])

    def test_stop_revokes_autonomous_mode(self):
        self.g.estop=False;self.g.mode='AUTONOMOUS'
        self.g.submit([.2,0,0],'autonomy',1)
        self.g.stop();self.g.estop=False
        self.assertFalse(self.g.submit([.2,0,0],'autonomy',1.01))
        self.assertEqual(self.g.tick(1.02,.02,True)[0],[0,0,0])

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):self.g.submit([.2,0,0],'untrusted',1)

    def test_sensor_loss_stops_immediately(self):
        self.g.estop=False
        self.g.submit([.3,0,0], 'manual', 0)
        self.g.tick(0,.1,True)
        self.assertEqual(self.g.tick(.1,.1,False)[0], [0,0,0])

    def test_controller_link_recovery_requires_explicit_rearming(self):
        self.g.estop=False;self.g.mode='AUTONOMOUS'
        self.g.submit([.2,0,0],'autonomy',1)
        self.assertGreater(self.g.tick(1,.1,True)[0][0],0)
        self.g.observe_controller_link(False)
        self.assertTrue(self.g.estop)
        self.assertEqual(self.g.mode,'MANUAL')
        self.assertEqual(self.g.output,[0,0,0])
        self.g.observe_controller_link(True)
        self.assertFalse(self.g.submit([.2,0,0],'autonomy',1.1))
        self.g.submit([.2,0,0],'manual',1.1)
        self.assertEqual(self.g.tick(1.1,.1,True)[0],[0,0,0])

    def test_each_source_expires_without_new_commands(self):
        for source in ('manual','autonomy'):
            with self.subTest(source=source):
                self.g.estop=False;self.g.mode='AUTONOMOUS'
                self.g.submit([.1,0,0],source,1)
                self.assertGreater(self.g.tick(1,.1,True)[0][0],0)
                self.assertEqual(self.g.tick(1.251,.1,True),([0,0,0],'COMMAND EXPIRED'))

    def test_autonomy_does_not_need_manual_input(self):
        self.g.estop=False;self.g.mode='AUTONOMOUS'
        for now in (1,1.1,1.2,1.3):
            self.g.observe_controller_link(True)
            self.assertTrue(self.g.submit([.1,0,0],'autonomy',now))
            self.assertGreater(self.g.tick(now,.1,True)[0][0],0)

if __name__ == '__main__':
    unittest.main()
