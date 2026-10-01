import unittest
import numpy as np
from arm_model import ArmModel
from cartesian_jog import propose

class CartesianJogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=ArmModel()

    def test_integer_command_reaches_small_forward_target_within_bound(self):
        start=[90,125,3,0,90,30]
        p=propose(self.model,start,'x',1)
        self.assertTrue(all(type(v) is int for v in p['goal_deg']))
        self.assertLessEqual(max(abs(a-b) for a,b in zip(start,p['goal_deg'])),10)
        self.assertEqual(p['goal_deg'][4:],start[4:])
        self.assertGreaterEqual(p['predicted_delta_m'][0],.002)
        self.assertLessEqual(p['quantization_error_m'],.003)
        self.assertFalse(p['executed'])

    def test_invalid_direction_and_impossible_local_target_rejected(self):
        for a,d in [('x',0),('y',True),('z',2),('q',1)]:
            with self.assertRaises(ValueError):propose(self.model,[90,125,3,0,90,30],a,d)
        self.assertEqual(propose(self.model,[90,125,3,0,90,30],'x',-1)['goal_deg'][4:], [90,30])

    def test_local_ik_never_uses_distant_redundant_solution(self):
        start=[90,125,3,0,90]
        p=np.array(self.model.fk(start,-.3)['xyz']);p[0]+=.005
        q=self.model.ik(p.tolist(),start,-.3,max_step_deg=10)
        self.assertLessEqual(np.max(np.abs(np.array(q['servo_deg'])-start)),10.00001)

    def test_cartesian_keeps_manual_wrist_roll_and_uses_pitch_as_local_seed(self):
        start=[90,125,3,20,110,30]
        p=propose(self.model,start,'x',1)
        self.assertEqual(p['goal_deg'][4],110)
        self.assertLessEqual(abs(p['goal_deg'][3]-20),10)
        self.assertNotEqual(p['goal_deg'][3],90)
