import tempfile
import unittest
import numpy as np
from grasp_learning import GraspMemory, features, FEATURES


class LearningTests(unittest.TestCase):
    def test_unknown_and_calibration_are_not_training_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            memory=GraspMemory(directory);image=np.zeros((20,20,3),np.uint8)
            vector=features(image,[0,0,20,20],[.2,0,.1],.04,90)
            self.assertEqual(len(vector),FEATURES)
            attempt=memory.begin('scene1','observed_calibration',vector,{},image)
            with self.assertRaises(ValueError):memory.finish(attempt,'success',{'servo_command_complete':True},image)
            self.assertEqual(memory.status()['training_eligible'],0)
            with self.assertRaises(ValueError):memory.train()
            self.assertFalse(memory.rank([{'features':vector}])['ranked'])

    def test_training_and_persistence_on_synthetic_fixture(self):
        # Synthetic data are confined to a temporary database, never robot memory.
        with tempfile.TemporaryDirectory() as directory:
            memory=GraspMemory(directory);image=np.zeros((20,20,3),np.uint8)
            evidence=dict(verifier='rgbd_lift_v1',calibration_validated=True,object_identity_verified=True)
            for scene in range(12):
                for sample in range(8):
                    positive=sample%2==0
                    vector=[1. if positive else -1.]*FEATURES
                    attempt=memory.begin(str(scene),'robot',vector,{},image)
                    memory.finish(attempt,'success' if positive else 'failure',evidence,image)
            report=memory.train()
            self.assertLess(report['validation_loss'],report['baseline_loss'])
            ranked=GraspMemory(directory).rank([{'id':'bad','features':[-1.]*FEATURES},
                                               {'id':'good','features':[1.]*FEATURES}])
            self.assertEqual(ranked['candidates'][0]['id'],'good')
            self.assertFalse(ranked['execution_permission'])

    def test_invalid_features_and_missing_scene_diversity(self):
        with self.assertRaises(ValueError):features(np.zeros((20,20,3)),[-1,0,20,20],[0,0,0],.03,0)
        with self.assertRaises(ValueError):features(np.zeros((20,20,3)),[0,0,20,20],[float('nan'),0,0],.03,0)


if __name__=='__main__':unittest.main()
