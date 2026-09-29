import unittest
from factory_zero_cadence import StationaryZeroCadence

class CadenceTests(unittest.TestCase):
    def test_stop_burst_then_quiet_windows_for_battery(self):
        gate=StationaryZeroCadence();sent=[]
        for tick in range(151):
            now=tick*.02
            if gate.publish_due([0,0,0],0.,now):sent.append(now)
        self.assertEqual(sent[:20],[i*.02 for i in range(20)])
        self.assertGreaterEqual(sent[20]-sent[19],1.)
    def test_moving_commands_are_never_decimated(self):
        gate=StationaryZeroCadence()
        for tick in range(100):self.assertTrue(gate.publish_due([.04,0,0],0.,tick*.02))
    def test_missing_or_moving_odometry_keeps_stop_publications(self):
        gate=StationaryZeroCadence()
        for tick in range(100):self.assertTrue(gate.publish_due([0,0,0],None,tick*.02))
    def test_transition_back_to_motion_is_immediate(self):
        gate=StationaryZeroCadence();self.assertTrue(gate.publish_due([0,0,0],0.,2.))
        self.assertFalse(gate.publish_due([0,0,0],0.,2.02))
        self.assertTrue(gate.publish_due([0,.04,0],None,2.04))

if __name__=='__main__':unittest.main()
