import unittest
from pathlib import Path
import yaml
from power_policy import PowerPolicy
from battery_gauge import battery_summary


class PowerTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((Path(__file__).parents[1]/'config/power.yaml').read_text())
        self.p=PowerPolicy(self.c)

    def test_stale_data_blocks_motion(self):
        self.p.sample(12.,0)
        self.assertTrue(self.p.evaluate(0)['motion_allowed'])
        self.assertFalse(self.p.evaluate(3.01)['motion_allowed'])

    def test_critical_sustained_and_latched(self):
        self.p.sample(10.7,0)
        self.assertFalse(self.p.evaluate(0)['shutdown_required'])
        self.p.sample(10.7,2.1)
        self.assertTrue(self.p.evaluate(2.1)['shutdown_required'])
        self.p.sample(12.6,3)
        self.assertFalse(self.p.evaluate(3)['motion_allowed'])

    def test_short_sag_does_not_request_shutdown(self):
        for t,v in [(0,12.),(1,10.5),(1.5,12.)]:self.p.sample(v,t)
        self.assertFalse(self.p.evaluate(1.5)['shutdown_required'])

    def test_hysteresis_and_charging(self):
        for i in range(10):self.p.sample(11.,i)
        self.assertEqual(self.p.evaluate(9)['state'],'LOW_POWER')
        for i in range(10,30):self.p.sample(11.2,i)
        self.assertEqual(self.p.evaluate(29)['state'],'LOW_POWER')
        for i in range(30,50):self.p.sample(11.5,i)
        self.assertEqual(self.p.evaluate(49)['state'],'NORMAL')
        self.assertFalse(self.p.evaluate(49,charging=True)['motion_allowed'])
        self.assertIsNone(battery_summary(12.5,0)['state_of_charge_percent'])
