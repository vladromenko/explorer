import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from mobile_demonstrations import episode_quality
from mobile_policy_contract import bound_action,read_bundle,ACTION_ORDER,UNITS
from arm_commissioning import coordinated_policy_status
from skill_learning import SkillLearning
from mobile_policy_execution import MobilePolicyExecution


class FakeJobs:
    def __init__(self):
        self.records=[]

    def status(self):
        return {"jobs":self.records}

    def start_mobile(self,steps,task,skill_id=None):
        record={"id":"job-"+str(len(self.records)+1),"state":"queued","skill_id":skill_id,"task":task}
        self.records.append(record)
        return record


class AutomaticLearningTests(unittest.TestCase):
    def test_record_quality_distinguishes_stationary_video_from_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            rows=[]
            for index in range(24):
                rows.append({"image_stamp":index/3,"command":[90,90,90,90,90,90,0,0,0]})
            (folder/"samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            quality=episode_quality(folder)
            self.assertFalse(quality["usable"])
            self.assertIn("почти нет разных команд",quality["reason"])
            for index,row in enumerate(rows):
                row["command"][0]=90+index%5
            (folder/"samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            self.assertTrue(episode_quality(folder)["usable"])
            self.assertFalse(episode_quality(folder,full_task=True)["usable"])
            for index,row in enumerate(rows):
                row["command"][5]=90+index%3
                row["command"][6]=.1 if index%3 else 0
            (folder/"samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            self.assertTrue(episode_quality(folder,full_task=True)["usable"])

    def test_outcome_is_idempotent_and_rename_keeps_episode_links(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs=FakeJobs()
            manager=SkillLearning(directory,None,jobs,interval=0)
            skill=manager.create("Собрать носок")
            quality={"usable":True,"samples":24,"observed_fps":3}
            record={"id":"a"*32,"skill_id":skill["id"],"state":"complete",
                    "outcome":"failure","quality":quality,"ended":100}
            manager.ingest(record,Path(directory)/"unused",schedule=True)
            self.assertEqual(len(jobs.records),0)
            record={**record,"id":"b"*32,"outcome":"success"}
            manager.ingest(record,Path(directory)/"unused",schedule=True)
            manager.ingest(record,Path(directory)/"unused",schedule=True)
            self.assertEqual(len(jobs.records),1)
            updated=manager.rename(skill["id"],"Носок в корзину")
            self.assertEqual(updated["successful_usable"],1)
            self.assertEqual(updated["name"],"Носок в корзину")
            correction={**record,"id":"c"*32,"outcome":"unknown","kind":"human_intervention"}
            manager.ingest(correction,Path(directory)/"unused",schedule=True)
            self.assertEqual(manager.get(skill["id"])["successful_usable"],1)
            jobs.records[0]["state"]="validated_offline"
            manager.ingest({**correction,"outcome":"success"},Path(directory)/"unused",schedule=True)
            self.assertEqual(manager.get(skill["id"])["successful_usable"],2)
            self.assertEqual(len(jobs.records),2)

    def test_policy_command_has_six_joints_and_three_body_speeds(self):
        current=[90,90,90,90,90,90,0,0,0]
        issued=bound_action(current,[95,90,90,90,90,90,.5,0,0])
        self.assertEqual(issued["issued_joint_goal_deg"][0],92)
        self.assertEqual(issued["issued_body_velocity"][0],.12)
        self.assertTrue(issued["arm_limited"])
        self.assertTrue(issued["base_limited"])

    def test_corrupt_or_wrong_unit_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint=root/"model";checkpoint.mkdir()
            (checkpoint/"model.safetensors").write_bytes(b"weights")
            (checkpoint/"config.json").write_text(json.dumps({"type":"act","n_action_steps":1,
                "input_features":{"observation.state":{"shape":[9]},
                                  "observation.images.wrist":{"shape":[3,240,320]}},
                "output_features":{"action":{"shape":[9]}}}))
            digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
            bundle={"format":"explorer_mobile_act_bundle_v1","dataset_kind":"mobile_manipulation_9dof",
                    "action_order":list(ACTION_ORDER),"observation_order":list(ACTION_ORDER),"units":list(UNITS),
                    "joint_state_source":"command_estimate","checkpoint_relative":"model",
                    "checkpoint_sha256":digests}
            (root/"bundle.json").write_text(json.dumps(bundle))
            self.assertEqual(read_bundle(root)[1],checkpoint)
            (checkpoint/"model.safetensors").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError,"Контрольная сумма"):
                read_bundle(root)
            (checkpoint/"model.safetensors").write_bytes(b"weights")
            bundle["units"][-1]="deg/s"
            (root/"bundle.json").write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError,"единицы"):
                read_bundle(root)

    def test_policy_arm_status_requires_real_localization_and_live_mission(self):
        state={"at":10,"mode":"AUTONOMOUS","stop_latched":False,"mission":"run",
               "commissioning":{key:True for key in ("base_commissioned","arm_commissioned",
                   "lidar_tf_validated","mcu_watchdog_verified","localization_verified")},
               "sensor_age":{"odom":.1,"battery":.1,"scan0":.1,"scan1":.1},
               "power":{"state":"NORMAL"},"velocity":[0,0,0],"odom_velocity":[0,0,0]}
        coordinated_policy_status(state,10.1)
        state["commissioning"]["localization_verified"]=False
        with self.assertRaisesRegex(ValueError,"Калибровка"):
            coordinated_policy_status(state,10.1)

    def test_correction_requires_completion_and_honest_operator_label(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/"data/mobile-demonstrations"/("d"*32)
            folder.mkdir(parents=True)
            path=folder/"episode.json"
            path.write_text(json.dumps({"id":"d"*32,"kind":"human_intervention",
                                        "state":"recording","outcome":"unknown"}))
            executor=MobilePolicyExecution.__new__(MobilePolicyExecution)
            executor.root=Path(directory)
            executor.skills=None
            with self.assertRaisesRegex(ValueError,"Сначала закончите"):
                executor.label_intervention("d"*32,"success")
            path.write_text(json.dumps({"id":"d"*32,"kind":"human_intervention",
                                        "state":"complete","outcome":"unknown"}))
            self.assertEqual(executor.label_intervention("d"*32,"failure")["outcome"],"failure")
            with self.assertRaisesRegex(ValueError,"уже оценён"):
                executor.label_intervention("d"*32,"success")


if __name__=="__main__":unittest.main()
