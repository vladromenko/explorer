import unittest
import numpy as np
from object_finder import depth_position,english_label

class ObjectFinderTests(unittest.TestCase):
    def sample(self):return dict(depth=np.full((100,100),.5),k=np.array([[100.,0,50],[0,100,50],[0,0,1]]),d=np.zeros(5),frame='optical')
    def test_surface_is_not_mistaken_for_object(self):
        self.assertIsNone(depth_position(self.sample(),[30,30,70,70]))
    def test_foreground_not_box_center_supplies_depth(self):
        s=self.sample();s['depth'][35:48,35:48]=.48
        r=depth_position(s,[30,30,70,70])
        self.assertEqual(r['z'],.48);self.assertLess(r['x'],0)
        self.assertFalse(r['depth_is_object_verified'])
        self.assertTrue(r['foreground_separated'])
    def test_holes_and_edges_rejected(self):
        s=self.sample();s['depth'][:]=0
        self.assertIsNone(depth_position(s,[30,30,70,70]))
        s=self.sample();s['depth'][50:]=1
        self.assertIsNone(depth_position(s,[30,30,70,70]))
    def test_nonfinite_box_rejected(self):
        self.assertIsNone(depth_position(self.sample(),[0,0,float('nan'),50]))
    def test_aliases_are_not_shell_or_model_code(self):
        self.assertEqual(english_label('Носок'),'sock')
        self.assertEqual(english_label('red bottle'),'red bottle')
        with self.assertRaises(ValueError):english_label('$(poweroff)')

if __name__=='__main__':unittest.main()
