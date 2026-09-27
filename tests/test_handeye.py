import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from handeye import fit


class HandEyeTests(unittest.TestCase):
    def test_frame_convention_and_held_out_consistency(self):
        rng=np.random.default_rng(91)
        camera=np.eye(4);camera[:3,:3]=Rotation.from_euler('xyz',[.4,-.3,.2]).as_matrix();camera[:3,3]=[-.1,0,.05]
        tools=[]
        for _ in range(12):
            tool=np.eye(4);tool[:3,:3]=Rotation.from_rotvec(rng.normal(0,.4,3)).as_matrix()
            tool[:3,3]=rng.uniform(-.1,.1,3);tools.append(tool)
        views=[np.linalg.inv(t@camera)@tools[0]@camera for t in tools]
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i,t in enumerate(tools):
                path=Path(directory)/f'{i}.npz';np.savez(path,base_tool=t,base_pose=[0,0,0]);paths.append(path)
            with patch('handeye.visual_pose',side_effect=[(v,{}) for v in views[1:]]):result=fit(paths)
            self.assertTrue(result['consistent'])
            self.assertTrue(np.allclose(result['camera_to_mount_reference'],camera,atol=1e-6))
            self.assertEqual(result['held_out_indices'],[10,11])
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['measured_joint_positions'])

    def test_insufficient_samples_cannot_authorize(self):
        with self.assertRaises(ValueError):fit([])
