import unittest
import numpy as np
from surface_foreground import locate

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
