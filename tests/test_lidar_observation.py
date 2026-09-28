import math
import unittest
from types import SimpleNamespace
from lidar_observation import summarize

class LidarObservationTests(unittest.TestCase):
    def test_single_close_return_remains_an_obstacle_with_bearing(self):
        s=SimpleNamespace(ranges=[math.nan,math.inf,.07,.9],range_min=.05,
            range_max=8.,angle_min=0.,angle_increment=.1,header=SimpleNamespace(frame_id='laser0_frame'))
        r=summarize(s)
        self.assertEqual(r['nearest'],.07)
        self.assertAlmostEqual(r['nearest_bearing_rad'],.2)
        self.assertEqual(r['close_count'],1)
        self.assertFalse(r['mask_applied'])
        s.ranges=[math.nan,math.inf,0.]
        self.assertIsNone(summarize(s)['nearest'])
