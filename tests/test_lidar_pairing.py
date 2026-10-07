import unittest
from lidar_pairing import pair_indices

class LidarPairingTests(unittest.TestCase):
    def test_new_scans_wait_for_pose_instead_of_overwriting_usable_pair(self):
        self.assertEqual(pair_indices([900_000_000,1_040_000_000],[940_000_000,1_080_000_000],
            1_010_000_000,1_100_000_000),(0,0))
        self.assertEqual(pair_indices([900_000_000,1_040_000_000],[940_000_000,1_080_000_000],
            1_100_000_000,1_100_000_000),(1,1))
    def test_stale_unpaired_and_future_data_never_form_a_scan(self):
        self.assertIsNone(pair_indices([0],[0],1_000_000_000,1_000_000_000))
        self.assertIsNone(pair_indices([800_000_000],[1_000_000_000],1_000_000_000,1_000_000_000))
        self.assertIsNone(pair_indices([1_100_000_000],[1_100_000_000],1_000_000_000,1_000_000_000))
