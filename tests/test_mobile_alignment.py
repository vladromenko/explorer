import math,unittest
from mobile_alignment import correction

class AlignmentTests(unittest.TestCase):
    def test_side_object_rotates_then_adjusts_range_and_requires_reobservation(self):
        p=correction([.18,-.08,.01])
        self.assertEqual(p['actions'][0]['direction'],'cw');self.assertTrue(p['reobserve_required'])
        self.assertLess(p['bearing_rad'],0)
    def test_front_workspace_needs_no_motion(self):
        p=correction([.24,.01,.02]);self.assertTrue(p['aligned']);self.assertFalse(p['actions'])
    def test_too_close_moves_back(self):
        p=correction([.14,0,.01]);self.assertEqual(p['actions'][0]['direction'],'backward')
