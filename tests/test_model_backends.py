import tempfile
import unittest
import numpy as np
from model_backends import ExplorerPolicyAdapter,SmolVLAAdapter,ActionChunk


class BackendTests(unittest.TestCase):
    def test_command_semantics_and_bounds_are_shared(self):
        adapter=ExplorerPolicyAdapter();image=np.zeros((10,12,3),np.uint8)
        obs=adapter.observation(image,[90]*6+[0,0,0],'pick sock',10)
        self.assertEqual(obs['state_source'],'command_estimate_not_proprioception')
        action=adapter.action([100]*6+[1,1,1],[90]*6+[0,0,0])
        self.assertTrue(action['clipped']);self.assertEqual(action['executed_request'][0],92)

    def test_action_chunk_invalidates_on_time_scene_or_policy(self):
        chunk=ActionChunk(1,[[0]*9],'p1','s1')
        self.assertEqual(chunk.next(1.1,'s1','p1'),[0]*9)
        chunk=ActionChunk(1,[[0]*9],'p1','s1')
        with self.assertRaises(ValueError):chunk.next(2,'s1','p1')
        with tempfile.TemporaryDirectory() as folder:self.assertFalse(SmolVLAAdapter(folder).status()['ready'])


if __name__=='__main__':unittest.main()
