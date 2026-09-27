import unittest
from policy_preview import describe_prediction

class PreviewTests(unittest.TestCase):
    def test_never_claims_execution_or_collision_validation(self):
        s=[90,125,3,0,90,30];a=[92,125,3,0,90,30]
        r=describe_prediction(s,a)
        self.assertTrue(r['within_two_degree_envelope'])
        self.assertFalse(r['executed']);self.assertFalse(r['collision_checked'])
        self.assertFalse(r['automatic_execution_allowed'])

    def test_large_and_out_of_range_outputs_remain_visible_not_clipped(self):
        r=describe_prediction([90,125,3,0,90,30],[200,125,3,0,90,30])
        self.assertFalse(r['within_two_degree_envelope']);self.assertFalse(r['inside_hardware_limits'])
        self.assertEqual(r['proposed_deg'][0],200)

    def test_nonfinite_and_wrong_shape_rejected(self):
        for a in ([float('nan')]*6,[90]*5):
            with self.assertRaises(ValueError):describe_prediction([90]*6,a)

if __name__=='__main__':unittest.main()
