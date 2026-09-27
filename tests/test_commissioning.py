import unittest
from commissioning import Pulse

class ProbeTests(unittest.TestCase):
    def test_small_single_axis_only(self):
        for v,d in [([.051,0,0],.5),([.01,.01,0],.5),([float('nan'),0,0],.5),([.01,0,0],1)]:
            with self.assertRaises(ValueError):Pulse(v,d,1)
    def test_estop_sensor_collision_and_lease_stop(self):
        for estop,sensors,obstacle,now in [(True,True,False,1.02),(False,False,False,1.02),(False,True,True,1.02),(False,True,False,1.16)]:
            p=Pulse([.04,0,0],.5,1)
            p.tick(1.01,.02,False,True,False)
            self.assertEqual(p.tick(now,.02,estop,sensors,obstacle)[0],[0,0,0])
            p.last_lease=now
            self.assertEqual(p.tick(now+.01,.02,False,True,False)[0],[0,0,0])
    def test_deadline_and_acceleration(self):
        p=Pulse([.04,0,0],.5,1)
        self.assertAlmostEqual(p.tick(1.02,.02,False,True,False)[0][0],.005)
        p.last_lease=1.49
        self.assertEqual(p.tick(1.50,.02,False,True,False)[0],[0,0,0])

    def test_gap_does_not_disable_deadline_or_fault_stop(self):
        for reason in ('deadline','lease','sensor','estop','collision'):
            p=Pulse([.04,0.,0.],.6,1.,.8)
            p.last_lease=1.65
            self.assertIsNone(p.tick(1.65,.02,False,True,False)[0])
            now=2.41 if reason=='deadline' else (1.81 if reason=='lease' else 1.67)
            if reason=='deadline':p.last_lease=now
            self.assertEqual(p.tick(now,.02,reason=='estop',reason!='sensor',reason=='collision')[0],[0,0,0])

    def test_gap_velocity_and_duration_bounds(self):
        for v,q in [([.05,0.,0.],.8),([0.,.04,0.],.8),([.04,0.,0.],.81)]:
            with self.assertRaises(ValueError):Pulse(v,.6,1.,q)

if __name__=='__main__':unittest.main()
