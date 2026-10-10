import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from lerobot_bridge import LearningJobs,save_job
from mobile_demonstrations import sample
from mobile_policy_contract import SAMPLE_CONTRACT,CONTRACT_SHA256
import numpy as np


class LearningQueueTests(unittest.TestCase):
    def setup(self,root):
        (root/"data").mkdir()
        (root/"data/learning-backend.json").write_text(json.dumps({"ready":True}))
        return LearningJobs(root)

    def episode(self,root,identifier):
        folder=root/"data/mobile-demonstrations"/identifier
        folder.mkdir(parents=True)
        record={"id":identifier,"name":"Носок","skill_id":"a"*32,"state":"complete",
                "outcome":"success","label_source":"operator","quality":{"usable":True}}
        (folder/"episode.json").write_text(json.dumps(record))
        (folder/"samples.jsonl").write_text(json.dumps({"image_stamp":1,"command":[90]*6+[0]*3}))

    def test_waiting_queue_survives_restart_budget_and_coalesces_new_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=self.setup(root);self.episode(root,"b"*32)
            with patch("lerobot_bridge.training_budget",side_effect=ValueError("needs power")),patch(
                "lerobot_bridge.subprocess.run",return_value=Mock(stdout="inactive")) as service:
                first=jobs.start_mobile(1000,"Носок",skill_id="a"*32)
                self.assertEqual(first["state"],"queued")
                self.assertEqual(service.call_count,0)
                jobs.dispatch()
                self.assertEqual(jobs.status()["jobs"][0]["state"],"deferred")
                restored=LearningJobs(root)
                self.assertEqual(restored.start_mobile(1000,"Носок",skill_id="a"*32)["id"],first["id"])
                self.episode(root,"c"*32)
                second=restored.start_mobile(1000,"Носок",skill_id="a"*32)
                self.assertNotEqual(second["id"],first["id"])
                restored.dispatch()
                saved={record["id"]:record for record in restored.status()["jobs"]}
                self.assertEqual(saved[first["id"]]["state"],"superseded")
                self.assertEqual(saved[second["id"]]["state"],"deferred")
            with patch("lerobot_bridge.training_budget"),patch("lerobot_bridge.subprocess.run",
                return_value=Mock(stdout="inactive")) as service:
                restored.dispatch()
                self.assertEqual(service.call_count,2)
                self.assertEqual(json.loads((root/"data/learning-request.json").read_text())["job"],second["id"])

    def test_explicit_cancel_stays_cancelled_after_reboot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=self.setup(root);self.episode(root,"b"*32)
            first=jobs.start_mobile(1000,"Носок",skill_id="a"*32)
            with patch("lerobot_bridge.subprocess.run",return_value=Mock(stdout="inactive")):
                jobs.stop()
                self.assertEqual(LearningJobs(root).dispatch()["state"],"idle")
            self.assertEqual(jobs.status()["jobs"][0]["state"],"cancelled")
            self.assertEqual(jobs.start_mobile(1000,"Носок",skill_id="a"*32)["id"],first["id"])

    def test_progress_writer_cannot_erase_concurrent_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=self.setup(root);self.episode(root,"b"*32)
            record=jobs.start_mobile(1000,"Носок",skill_id="a"*32)
            path=jobs.folder/record["id"]/"job.json"
            with patch("lerobot_bridge.subprocess.run",return_value=Mock(stdout="inactive")):
                jobs.stop()
            with self.assertRaises(InterruptedError):save_job(path,{**record,"state":"training"})
            save_job(path,{**record,"state":"deferred"},terminal=True)
            self.assertTrue(json.loads(path.read_text())["cancel_requested"])
            self.assertEqual(json.loads(path.read_text())["state"],"cancelled")

    def test_dead_training_retains_job_and_defers_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=self.setup(root);self.episode(root,"b"*32)
            record=jobs.start_mobile(1000,"Носок",skill_id="a"*32)
            path=jobs.folder/record["id"]/"job.json"
            record.update(state="training",at=time.time()-30)
            path.write_text(json.dumps(record))
            with patch("lerobot_bridge.training_budget",side_effect=ValueError("busy")),patch(
                "lerobot_bridge.subprocess.run",return_value=Mock(stdout="inactive")):
                jobs.dispatch()
            saved=json.loads(path.read_text())
            self.assertEqual(saved["state"],"deferred")
            self.assertEqual(saved["dataset_fingerprint"],record["dataset_fingerprint"])

    def test_sample_separates_estimated_observation_applied_command_and_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"data").mkdir();now=time.time()
            arm={"at":now,"phase":"command_in_progress","servo_deg":[92]*6,
                 "q_estimated_deg":[90.5]*6,"velocity_deg_s":[4]*6,"command_generation":4}
            status={"at":now,"mode":"MANUAL","velocity":[.1,0,0],"power":{"state":"NORMAL"},
                    "sensor_age":dict(odom=0,battery=0,scan0=0,scan1=0),"arm_command_state":arm}
            (root/"data/status.json").write_text(json.dumps(status))
            np.savez(root/"data/rgbd-snapshot.npz",stamp=now,rgb=np.zeros((3,3,3),dtype=np.uint8))
            item,_=sample(root,now)
            self.assertEqual(item["applied_action"][:6],[92]*6)
            self.assertEqual(item["observation_state"][:6],[90.5]*6)
            self.assertEqual(item["sample_contract_sha256"],CONTRACT_SHA256)
            self.assertFalse(item["measured_arm_angles"])

    def test_resume_requires_weights_and_optimizer_not_a_folder_name(self):
        spec=importlib.util.spec_from_file_location("queue_learning_run",Path(__file__).resolve().parents[1]/"bin/learning-run.py")
        module=importlib.util.module_from_spec(spec)
        with patch("learning_environment.configure",return_value={}):spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);checkpoint=folder/"model/checkpoints/000025"
            pretrained=checkpoint/"pretrained_model";state=checkpoint/"training_state"
            pretrained.mkdir(parents=True);state.mkdir()
            (pretrained/"train_config.json").write_text("{}")
            (pretrained/"model.safetensors").write_bytes(b"weights")
            (state/"training_step.json").write_text(json.dumps({"step":25}))
            self.assertIsNone(module.resumable_checkpoint(folder))
            for name in ("optimizer_state.safetensors","rng_state.safetensors"):
                (state/name).write_bytes(b"saved")
            self.assertEqual(module.resumable_checkpoint(folder)[0],25)


if __name__=="__main__":unittest.main()
