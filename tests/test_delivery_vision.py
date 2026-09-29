import unittest
import cv2
import numpy as np
from delivery_vision import FeatureObject,JointHistory

class VisionTests(unittest.TestCase):
    def sample(self,t=1.,dx=0):
        rng=np.random.default_rng(33)
        rgb=np.zeros((120,160,3),np.uint8)
        rgb[40:80,60+dx:100+dx]=rng.integers(0,256,(40,40,3),np.uint8)
        depth=np.ones((120,160));depth[40:80,60+dx:100+dx]=.98
        return dict(rgb=rgb,depth=depth,k=np.array([[200.,0,80],[0,200,60],[0,0,1.]]),
                    d=np.zeros(5),stamp=t,frame='camera')
    def test_real_feature_continuity_not_detector_label_id(self):
        track=FeatureObject(self.sample(),[59,39,101,81])
        result=track.update(self.sample(1.1,2))
        self.assertEqual(result['object_id'],track.object_id)
        self.assertGreater(result['association_fraction'],.85)
        self.assertAlmostEqual(result['point_camera'][2],.98)
        with self.assertRaises(ValueError):track.update(self.sample(2,2))
        with self.assertRaises(ValueError):track.update(self.sample(2.1,2))
    def test_occlusion_never_inherits_old_identity(self):
        track=FeatureObject(self.sample(),[59,39,101,81])
        sample=self.sample(1.1);sample['rgb'][:]=0
        with self.assertRaises(ValueError):track.update(sample)
    def test_camera_pose_interpolation_never_extrapolates_or_crosses_boot(self):
        history=JointHistory()
        for index in range(1,7):
            for t,q in [(1000000000,-10.),(1020000000,-9.)]:
                history.add(dict(boot_id=1,joint=index,position_valid=True,error=0,device_error=0,
                                 acquired_monotonic_ns=t,physical_deg=q))
        np.testing.assert_allclose(history.at(1.01,1),[-9.5]*6)
        with self.assertRaises(ValueError):history.at(1.1,1)
        with self.assertRaises(ValueError):history.at(1.01,2)
        history.add(dict(boot_id=1,joint=3,position_valid=False,error=14,device_error=0))
        with self.assertRaises(ValueError):history.at(1.01,1)

    def test_reacquisition_uses_true_timestamp_and_new_identity(self):
        from delivery_vision import reacquire_candidate
        old=self.sample(1);fresh=self.sample(20)
        candidate=reacquire_candidate(old,fresh,[59,39,101,81])
        self.assertEqual(candidate['image_stamp'],20)
        self.assertFalse(candidate['continuous_from_previous_image'])
        self.assertFalse(candidate['semantic_identity_verified'])
        a=FeatureObject(old,candidate['bbox']);b=FeatureObject(fresh,candidate['bbox'])
        self.assertNotEqual(a.object_id,b.object_id)
        changed=self.sample(20,8)
        with self.assertRaises(ValueError):reacquire_candidate(old,changed,candidate['bbox'])
