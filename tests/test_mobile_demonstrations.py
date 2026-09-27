import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mobile_demonstrations import MobileDemonstrations,sample

class MobileTests(unittest.TestCase):
    def setup_files(self,root):
        now=time.time();(root/'data').mkdir();state=dict(at=now,mode='MANUAL',velocity=[0,0,0],power={'state':'IDLE'},sensor_age=dict(odom=0,battery=0,scan0=0,scan1=0),arm_command_state=dict(at=now,phase='command_elapsed_observation_required',servo_deg=[90,125,3,0,90,30]))
        (root/'data/status.json').write_text(json.dumps(state))
        np.savez(root/'data/rgbd-snapshot.npz',stamp=now,rgb=np.zeros((3,3,3),dtype=np.uint8))
        return state,now
    def test_samples_are_nine_commands_not_measured_positions(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);state,now=self.setup_files(root);r,image=sample(root,now)
            self.assertEqual(len(r['command']),9);self.assertFalse(r['measured_arm_angles'])
            self.assertEqual(r['velocity_source'],'commanded_body_velocity')
    def test_unsynchronized_or_faulted_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);state,now=self.setup_files(root)
            np.savez(root/'data/rgbd-snapshot.npz',stamp=now-.8,rgb=np.zeros((3,3,3),dtype=np.uint8))
            with self.assertRaises(ValueError):sample(root,now)
            np.savez(root/'data/rgbd-snapshot.npz',stamp=now,rgb=np.zeros((3,3,3),dtype=np.uint8))
            (root/'data/arm-telemetry-fault.json').write_text(json.dumps({'at':now+1}))
            with self.assertRaises(ValueError):sample(root,now)
    def test_restart_cannot_resume_demo_and_partial_task_not_success(self):
        with tempfile.TemporaryDirectory() as path,patch('mobile_demonstrations.threading.Thread'):
            root=Path(path);self.setup_files(root);m=MobileDemonstrations(root);m.start('deliver',True)
            with self.assertRaises(ValueError):m.finish('success')
            r=MobileDemonstrations(root);self.assertIsNone(r.active)
            saved=json.loads(next(r.folder.glob('*/episode.json')).read_text());self.assertEqual(saved['state'],'interrupted')

    def test_invalid_sensor_age_and_unknown_power_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);state,now=self.setup_files(root)
            for age in (float('nan'),float('inf'),-1):
                state['sensor_age']['odom']=age
                (root/'data/status.json').write_text(json.dumps(state))
                with self.assertRaises(ValueError):sample(root,now)
            state['sensor_age']['odom']=0;state['power']={}
            (root/'data/status.json').write_text(json.dumps(state))
            with self.assertRaises(ValueError):sample(root,now)
