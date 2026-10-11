import unittest
import numpy as np
from surface_foreground import locate, interior_surface_point

class ForegroundTests(unittest.TestCase):
    def sample(self):return dict(depth=np.full((120,120),.5),k=np.array([[100.,0,60],[0,100,60],[0,0,1]]),d=np.zeros(5),frame='optical')
    def test_disconnected_equal_candidates_are_unknown(self):
        s=self.sample();s['depth'][40:53,40:53]=.47;s['depth'][68:81,68:81]=.47
        self.assertEqual(locate(s,[30,30,90,90])['reason'],'AMBIGUOUS_COMPONENTS')
    def test_unresolved_thin_object_is_unknown(self):
        s=self.sample();s['depth'][40:80,40:80]=.497
        self.assertEqual(locate(s,[30,30,90,90])['reason'],'NO_RESOLVED_FOREGROUND')
    def test_sloped_background_plane(self):
        s=self.sample();v,u=np.mgrid[:120,:120]
        s['depth']=.5/(1+.25*(u-60)/100)
        s['depth'][40:80,40:80]-=.025
        r=locate(s,[30,30,90,90]);self.assertEqual(r['outcome'],'candidate')
        self.assertFalse(r['evidence']['support_plane_semantics_verified'])

    def test_folded_foreground_median_in_empty_gap_is_not_a_grasp_point(self):
        s = self.sample()
        s["depth"][35:85, 35:45] = .47
        s["depth"][35:85, 75:85] = .47
        s["depth"][75:85, 35:85] = .47
        result = locate(s, [30, 30, 90, 90])
        self.assertEqual(result["outcome"], "candidate")
        centroid = result["position"]
        x = round(100 * centroid["x"] / centroid["z"] + 60)
        y = round(100 * centroid["y"] / centroid["z"] + 60)
        self.assertEqual(s["depth"][y, x], .5)
        grasp = result["grasp_surface"]
        x, y = grasp["pixel"]
        self.assertEqual(s["depth"][y, x], .47)
        self.assertGreaterEqual(grasp["clearance_px"], 4.)
        self.assertFalse(grasp["finger_clearance_measured"])

    def test_selection_rejects_invalid_depth_and_counts_image_border(self):
        xyz = np.ones((8, 8, 3))
        xyz[3:5, 3:5] = np.nan
        mask = np.ones((8, 8), dtype=bool)
        grasp = interior_surface_point(xyz, mask)
        x, y = grasp["pixel"]
        self.assertTrue(np.isfinite(xyz[y, x]).all())
        self.assertLess(grasp["clearance_px"], 5.)
        with self.assertRaises(ValueError):
            interior_surface_point(xyz, np.zeros_like(mask))
