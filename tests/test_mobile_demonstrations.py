import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mobile_demonstrations import MobileDemonstrations,sample

class MobileTests(unittest.TestCase):
    def test_damaged_samples_finish_episode_without_stuck_recorder(self):
        with tempfile.TemporaryDirectory() as path,patch("mobile_demonstrations.threading.Thread"):
            root=Path(path);self.setup_files(root)
            recorder=MobileDemonstrations(root);recorder.start("grasp sock",True)
            (recorder.folder/recorder.active["id"]/"samples.jsonl").write_text("{broken")
            status=recorder.finish("unknown")
            self.assertIsNone(status["active"])
            self.assertFalse(status["last"]["quality"]["usable"])
            self.assertIn("Повреждена запись",status["last"]["quality"]["reason"])
            recorder.start("new grasp sock",True)
            self.assertIsNotNone(recorder.active)

    def test_nonfinite_corrupt_times_do_not_prevent_episode_finalization(self):
        with tempfile.TemporaryDirectory() as path,patch("mobile_demonstrations.threading.Thread"):
            root=Path(path);self.setup_files(root)
            recorder=MobileDemonstrations(root);recorder.start("grasp sock",True)
            rows=[{"image_stamp":stamp,"command":[90]*6+[0]*3}
                  for stamp in (1,float("nan"))]
            (recorder.folder/recorder.active["id"]/"samples.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows))
            status=recorder.finish("unknown")
            self.assertIsNone(status["active"])
            self.assertFalse(status["last"]["quality"]["usable"])
            json.dumps(status["last"],allow_nan=False)

    def test_recognition_metadata_requires_the_recorded_camera_exposure(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);state,now=self.setup_files(root)
            target={"at":now,"image_stamp":now,"objects":[{"label":"sock","identity_verified":False}]}
            (root/"data/grounded-targets.json").write_text(json.dumps(target))
            row,_=sample(root,now)
            self.assertEqual(row["recognized_targets"]["objects"][0]["label"],"sock")
            self.assertFalse(row["recognized_targets"]["identity_verified"])
            target["image_stamp"]=now-.1
            (root/"data/grounded-targets.json").write_text(json.dumps(target))
            row,_=sample(root,now)
            self.assertIsNone(row["recognized_targets"])
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
            self.assertEqual(r['camera_state_offset_s'],0)
    def test_unsynchronized_or_faulted_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);state,now=self.setup_files(root)
            np.savez(root/'data/rgbd-snapshot.npz',stamp=now-2.5,rgb=np.zeros((3,3,3),dtype=np.uint8))
            with self.assertRaises(ValueError):sample(root,now)
            np.savez(root/'data/rgbd-snapshot.npz',stamp=now,rgb=np.zeros((3,3,3),dtype=np.uint8))
            (root/'data/arm-telemetry-fault.json').write_text(json.dumps({'at':now+1}))
            with self.assertRaises(ValueError):sample(root,now)
    def test_restart_cannot_resume_demo_and_partial_task_not_success(self):
        with tempfile.TemporaryDirectory() as path,patch('mobile_demonstrations.threading.Thread'):
            root=Path(path);self.setup_files(root);m=MobileDemonstrations(root);m.start('deliver',True)
            result=m.finish("success")
            self.assertFalse(result["last"]["quality"]["usable"])
            self.assertEqual(result["successful"],0)
            m.start("deliver",True)
            r=MobileDemonstrations(root);self.assertIsNone(r.active)
            saved=[json.loads(file.read_text()) for file in r.folder.glob("*/episode.json")]
            interrupted=[row for row in saved if row["state"]=="interrupted"]
            self.assertEqual(len(interrupted),1)
            self.assertEqual(interrupted[0]["outcome"],"unknown")

    def test_local_grasp_demo_uses_real_samples_without_stage_buttons(self):
        with tempfile.TemporaryDirectory() as path,patch('mobile_demonstrations.threading.Thread'):
            root=Path(path);self.setup_files(root);m=MobileDemonstrations(root);m.start('grasp sock',True)
            folder=m.folder/m.active["id"]
            rows=[dict(image_stamp=float(index)/2,command=[90+index,90,90,90,90,30+index,0,0,0]) for index in range(20)]
            (folder/"samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            m.active["samples"]=len(rows)
            result=m.finish("success")
            self.assertEqual(result['last']['outcome'],'success')
            self.assertEqual(result['successful'],1)

    def test_status_exposes_latest_episode_and_storage_location(self):
        with tempfile.TemporaryDirectory() as path,patch('mobile_demonstrations.threading.Thread'):
            root=Path(path);self.setup_files(root);m=MobileDemonstrations(root);m.start('grasp sock',True)
            identifier=m.active['id'];m.stage('grasp');m.active['samples']=20;m.finish('success')
            restored=MobileDemonstrations(root).status()
            self.assertEqual(restored['last']['id'],identifier)
            self.assertEqual(restored['recent'][0]['id'],identifier)
            self.assertEqual(restored['storage_root'],str(root/'data/mobile-demonstrations'))

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
