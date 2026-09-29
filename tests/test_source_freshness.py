import unittest
from source_freshness import SourceFreshness

class SourceFreshnessTests(unittest.TestCase):
    def test_old_duplicate_future_are_not_refreshed(self):
        f=SourceFreshness()
        self.assertTrue(f.accept('odom',100,100.01,.5))
        for stamp,now in [(100,100.1),(99.9,100.1),(100.1,101),(102,101),(float('nan'),101)]:
            self.assertFalse(f.accept('odom',stamp,now,.5))
        self.assertEqual(f.last['odom'],100)
        self.assertTrue(f.accept('odom',101,101.01,.5))
        self.assertTrue(f.accept('scan',101.2,101,.5))
    def test_independent_sources_and_checked_clock_recovery(self):
        f=SourceFreshness();f.accept('odom',100,100,.5)
        self.assertTrue(f.accept('scan',99.9,100,.6))
        self.assertFalse(f.accept('odom',101.5,101,.5))
        self.assertEqual(f.diagnostics['odom']['reason'],'SOURCE_CLOCK_OR_AGE_FAULT')
        self.assertTrue(f.accept('odom',101.01,101.02,.5))
