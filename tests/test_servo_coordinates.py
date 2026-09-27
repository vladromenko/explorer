import unittest
import numpy as np
from servo_coordinates import to_radians,to_servo

class ServoCoordinatesTests(unittest.TestCase):
    def test_vendor_base_inversion_and_other_axes_preserved(self):
        np.testing.assert_allclose(np.degrees(to_radians([100,100,100,100,100])),[-10,10,10,10,10])
    def test_roundtrip_at_hardware_endpoints(self):
        for q in ([0,0,0,0,0],[180,180,180,180,270],[92,125,3,0,90]):
            np.testing.assert_allclose(to_servo(to_radians(q)),q,atol=1e-12)

if __name__=='__main__':unittest.main()
