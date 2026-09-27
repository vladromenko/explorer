import unittest
from battery_gauge import battery_percent, battery_summary


class BatteryGaugeTests(unittest.TestCase):
    def test_invalid_or_stale_never_looks_full(self):
        for value in (None, True, float('nan'), float('inf'), 0, 24):
            self.assertIsNone(battery_percent(value))
        for age in (3, -1, float('nan'), None):
            self.assertIsNone(battery_summary(12.6, age)['percent'])

    def test_operating_endpoints_and_no_charger_inference(self):
        self.assertEqual(battery_percent(10.8), 0)
        self.assertEqual(battery_percent(12.6), 100)
        self.assertEqual(battery_percent(11.7), 50)
        self.assertTrue(battery_summary(12.6, .1)['approximate'])
        self.assertIsNone(battery_summary(12.6, .1)['charging'])
