import copy
import math
import unittest
from capture_geometry import stationary_exposure

class CaptureTests(unittest.TestCase):
    def samples(self):
        a=dict(measured=True,boot_id=1,session=2,position_rad=[0.]*6,
               servo_deg=[90.]*6,acquired_monotonic=[9.9]*6)
        b=copy.deepcopy(a);b['acquired_monotonic']=[10.1]*6
        return a,b
    def test_static_measurements_bracket_camera_and_keep_signed_joints(self):
        a,b=self.samples();a['position_rad'][2]=b['position_rad'][2]=-2.
        self.assertEqual(stationary_exposure(a,b,10.)['position_rad'][2],-2.)
    def test_reject_moving_stale_or_rebooted_capture(self):
        for field,value in [('session',3),('measured',False),('acquired_monotonic',[11.]*6),('position_rad',[math.radians(1)]*6)]:
            a,b=self.samples();b[field]=value
            with self.assertRaises(ValueError):stationary_exposure(a,b,10.)
