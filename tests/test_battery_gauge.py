import unittest
from battery_gauge import battery_percent, battery_summary


class BatteryGaugeTests(unittest.TestCase):
    def test_invalid_or_stale_never_looks_full(self):
        for value in (None, True, float('nan'), float('inf'), 0, 24):
            self.assertIsNone(battery_percent(value))
        for age in (3, -1, float('nan'), None):
            self.assertIsNone(battery_summary(12.6, age)['percent'])

    def test_operating_endpoints_and_no_charger_inference(self):
        for voltage in (10.8, 11.7, 12.6):self.assertIsNone(battery_percent(voltage))
        self.assertFalse(battery_summary(12.6, .1)['approximate'])
        self.assertIsNone(battery_summary(12.6, .1)['charging'])
