import unittest
from arm_model import ArmModel
class ReferenceArmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=ArmModel()
    def test_fk_ik_round_trip_without_execution(self):
        p=self.model.fk([90,75,110,100,90],-.7)
        q=self.model.ik(p['xyz'],[88,77,108,102,91],-.7)
        self.assertTrue(q['solved']);self.assertLess(q['position_error_m'],.002)
        self.assertFalse(q['execution_allowed']);self.assertFalse(q['executed'])
    def test_unreachable_target_not_reported_solved(self):
        q=self.model.ik([0,0,.9],[90]*5,-.7)
        self.assertFalse(q['solved'])
    def test_invalid_and_colliding_reference_rejected(self):
        for q in ([90,181,90,90,90],[90,90,90,90,float('nan')]):
            with self.assertRaises(ValueError):self.model.fk(q,-.7)
        p=self.model.path([90]*5,[95,90,90,90,90],-1.)
        self.assertFalse(p['valid']);self.assertFalse(p['executed'])
    def test_real_joint_adjacency_and_fixed_wrist_roll(self):
        self.assertFalse(self.model.fk([90,125,3,0,90],-.5)['collision'])
        p=self.model.fk([90,75,110,100,90],-.5)
        q=self.model.ik(p['xyz'],[88,77,108,102,100],-.5)
        self.assertEqual(q['servo_deg'][4],100.)
