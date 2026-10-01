import json
from pathlib import Path
import tempfile
import time
import unittest
from power_endurance import PowerEndurance


class PowerEnduranceTests(unittest.TestCase):
    def test_runtime_estimate_requires_real_negative_history(self):
        now=time.time();rows=[{'at':now-700+i*6,'battery_voltage_v':12.2-i*.002,'charging':False,'state':'NORMAL'} for i in range(121)]
        result=PowerEndurance.trend(rows)
        self.assertTrue(result['available']);self.assertEqual(result['method'],'least_squares_voltage_trend_not_SOC')
        self.assertGreater(result['estimated_minutes_to_10_8v'],0)
        self.assertFalse(PowerEndurance.trend(rows[:20])['available'])

    def test_profile_is_persistent_and_does_not_claim_nvpmodel_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir();(root/'data/status.json').write_text('{}')
            manager=PowerEndurance(root);result=manager.select('ENDURANCE')
            self.assertEqual(result['selected_profile'],'ENDURANCE')
            self.assertFalse(result['claims']['nvpmodel_changed'])


if __name__=='__main__':unittest.main()
