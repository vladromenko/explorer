import math,unittest
from mobile_alignment import correction, reachable_correction

class AlignmentTests(unittest.TestCase):
    def test_side_object_rotates_then_adjusts_range_and_requires_reobservation(self):
        p=correction([.18,-.08,.01])
        self.assertEqual(p['actions'][0]['direction'],'cw');self.assertTrue(p['reobserve_required'])
        self.assertLess(p['bearing_rad'],0)
    def test_front_workspace_needs_no_motion(self):
        p=correction([.24,.01,.02]);self.assertTrue(p['aligned']);self.assertFalse(p['actions'])
    def test_too_close_moves_back(self):
        p=correction([.14,0,.01]);self.assertEqual(p['actions'][0]['direction'],'backward')

    def test_range_is_chosen_from_ik_and_preserves_object_height(self):
        checked = []
        def reachable(point):
            checked.append(point)
            return abs(point[0] - .26) < .0001 and point[1] == 0.
        plan = reachable_correction([.33, -.07, .035], reachable)
        self.assertAlmostEqual(plan["desired_distance_m"], .26)
        self.assertEqual(plan["actions"][0]["kind"], "rotate")
        self.assertEqual(plan["actions"][1]["direction"], "forward")
        self.assertTrue(plan["reobserve_required"])
        self.assertTrue(all(point[2] == .035 for point in checked))

    def test_small_but_required_shift_is_not_lost_to_range_tolerance(self):
        plan = reachable_correction([.23, 0, .02], lambda point: abs(point[0] - .24) < .0001)
        self.assertFalse(plan["aligned"])
        self.assertEqual(plan["actions"][0]["direction"], "backward")
        self.assertAlmostEqual(plan["actions"][0]["distance_m"], -.01)

    def test_reachable_pose_does_not_move_base_and_impossible_pose_fails(self):
        self.assertTrue(reachable_correction([.29, 0, .02], lambda point: True)["aligned"])
        with self.assertRaisesRegex(ValueError, "Нет позиции"):
            reachable_correction([.3, .1, .02], lambda point: False)
