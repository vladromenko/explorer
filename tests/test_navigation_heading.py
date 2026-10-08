import math
import unittest
from navigation_heading import departure_heading, reverse_without_turn


class HeadingTests(unittest.TestCase):
    def test_follows_first_segment_instead_of_final_goal_bearing(self):
        pose={"x":0.,"y":0.,"yaw":0.}
        self.assertAlmostEqual(departure_heading([[0.,0.],[.01,0.],[0.,.2],[2.,.2]],pose),math.pi/2)
        self.assertAlmostEqual(departure_heading([[0.,0.],[-.3,0.]],pose),math.pi)

    def test_empty_invalid_and_rotation_only(self):
        pose={"x":1.,"y":2.,"yaw":.8}
        self.assertEqual(departure_heading([[1.,2.]],pose),.8)
        for points in ([],[[float("nan"),0.]],[[1.]]):
            with self.assertRaises(ValueError):departure_heading(points,pose)

    def test_short_reverse_does_not_require_turning_near_furniture(self):
        self.assertTrue(reverse_without_turn(.8,math.pi))
        self.assertFalse(reverse_without_turn(2.,math.pi))
        self.assertFalse(reverse_without_turn(.8,math.pi/2))
