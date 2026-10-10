"""Isolated regressions from the 2026-10-08 audit; no hardware IO."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from autonomy_runtime import AutonomySupervisor
from delivery_robot import validate_delivery_goal
from delivery_task import DeliveryTask


class AuditRegressions(unittest.TestCase):
    def supervisor(self, root, gateway):
        world=lambda: {"scene_version":"test", "predicates":{}}
        readiness=lambda: {"hardware":{"physical_execution_ready":True}, "permissions":{}}
        with patch("autonomy_runtime.threading.Thread"):
            supervisor=AutonomySupervisor(root,world,readiness,gateway)
        supervisor.planner.plan=Mock(return_value={"steps":[],"blocked":False})
        return supervisor

    def test_terminal_outcome_and_cancellation_are_truthful(self):
        for outcome in ("success", "failure", "unknown", "infrastructure_error", "cancelled", "manual_cancel"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/"data").mkdir()
                def gateway(*args):
                    if outcome=="manual_cancel":
                        supervisor.cancel(job["id"])
                        return {"state":"success"}
                    return {"state":outcome}
                supervisor=self.supervisor(root,gateway)
                job=supervisor.submit("audit-request-1234",{"goal":"audit", "attempts":1,"time_budget_s":60})
                supervisor.process(job)
                result=supervisor.get(job["id"])
                expected="succeeded" if outcome=="success" else "cancelled" if outcome in ("cancelled","manual_cancel") else "failed"
                self.assertEqual(result["state"],expected)
                if outcome=="infrastructure_error":self.assertEqual(result["result"]["counts"]["infrastructure_errors"],1)
                if expected=="failed":self.assertFalse(result["result"]["goal_completed"])
                supervisor.shutdown()

    def test_twentieth_verified_grasp_registers_candidate_without_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"data").mkdir()
            supervisor=self.supervisor(root,Mock())
            checkpoint={"id":"audit-candidate","feature_version":"audit-v1","train_episodes":["train"],
                        "validation_episodes":["validation"],"metrics":{}}
            supervisor.scorer.train=Mock(return_value=checkpoint)
            outcome={"events":[{"stage":"approach","result":{"grasp_selection":{
                "rows":[{"candidate":{"approach_x":0.1}}],"selected":0,"context":{}}}},
                {"stage":"verify_hold","result":{"outcome":"success","evidence":{"verifier":"rgbd_lift_v1"}}}]}
            for index in range(20):supervisor.record_scorer_sample("episode-"+str(index),outcome)
            supervisor.scorer.train.assert_called_once()
            policies=supervisor.policies.all()
            self.assertEqual(len(policies),1)
            self.assertEqual(policies[0]["state"],"candidate")
            supervisor.shutdown()

    def test_delivery_goal_cannot_silently_change_object_or_destination(self):
        settings={"destination":"basket"}
        for label in ("sock","socks","носок","носки"):
            validate_delivery_goal({"object_query":label,"destination":{"name":"basket"}},settings)
        for goal in ({"object_query":"cookies","destination":{"name":"basket"}},
                     {"object_query":"sock","destination":{"name":"kitchen"}},{}):
            with self.assertRaises(ValueError):validate_delivery_goal(goal,settings)

    def test_delivery_worker_keeps_copy_of_requested_goal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"data").mkdir()
            robot=Mock();robot.blockers.return_value=[]
            # End before any skills: only inspect the arguments passed to begin.
            robot.begin.side_effect=ValueError("end fixture before motion")
            task=DeliveryTask(root,robot)
            goal={"object_query":"sock","destination":{"name":"basket"}}
            with patch("delivery_task.threading.Thread"):
                task.start(goal=goal)
            goal["destination"]["name"]="changed"
            task.run(task.active["id"],task.generation)
            self.assertEqual(robot.begin.call_args.kwargs["goal"]["destination"]["name"],"basket")
            robot.find.assert_not_called()

    def test_corrected_training_data_can_be_retried_without_duplicate_live_job(self):
        import hashlib
        import cv2
        import numpy as np
        from lerobot_bridge import LearningJobs
        from mobile_demonstrations import episode_quality
        from mobile_policy_contract import CONTRACT_SHA256
        for previous in ("failed","interrupted","cancelled","rejected","queued","complete"):
            with self.subTest(previous=previous),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=root/"data/mobile-demonstrations/demo";source.mkdir(parents=True)
                samples=[]
                for index in range(13):
                    image=f"{index:06d}.jpg"
                    self.assertTrue(cv2.imwrite(str(source/image),np.full((8,8,3),index*10,dtype=np.uint8)))
                    samples.append({"image":image,"image_stamp":1+index*.5,"state_stamp":1+index*.5,
                        "command":[90+index,90,90,90,90,90+index,.1,0,0],
                        "executed_action_source":"operator_manual_control",
                        "sample_contract_sha256":CONTRACT_SHA256})
                (source/"samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in samples))
                quality=episode_quality(source,full_task=True)
                self.assertTrue(quality["usable"],quality)
                (source/"episode.json").write_text(json.dumps({"id":"demo","name":"sock","state":"complete",
                    "outcome":"success","label_source":"operator","quality":quality}))
                inputs={"demo":{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                                for name in ("episode.json","samples.jsonl")}}
                fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
                jobs=LearningJobs(root);old=jobs.folder/"old";old.mkdir()
                old_record={"id":"old","at":9999999999,"state":previous,"skill_id":None,
                    "dataset_fingerprint":fingerprint,"input_sha256":inputs}
                (old/"job.json").write_text(json.dumps(old_record))
                (root/"data/learning-backend.json").write_text(json.dumps({"ready":True}))
                with patch("lerobot_bridge.training_budget"),patch("lerobot_bridge.subprocess.run",return_value=Mock(stdout="inactive")) as run:
                    result=jobs.start_mobile(1000,"sock")
                    # A restart must preserve terminal outcomes and explicit cancellation.
                    self.assertEqual(result["id"],"old")
                    self.assertEqual(result["state"],previous)
                    # A corrected demonstration has new source provenance and can queue once.
                    samples[-1]["command"][0]=101
                    (source/"samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in samples))
                    corrected_episode=json.loads((source/"episode.json").read_text())
                    corrected_episode["quality"]=episode_quality(source,full_task=True)
                    self.assertTrue(corrected_episode["quality"]["usable"])
                    (source/"episode.json").write_text(json.dumps(corrected_episode))
                    corrected=jobs.start_mobile(1000,"sock")
                    duplicate=LearningJobs(root).start_mobile(1000,"sock")
                    self.assertNotEqual(corrected["id"],"old")
                    self.assertEqual(corrected["state"],"queued")
                    self.assertEqual(duplicate["id"],corrected["id"])
                    self.assertNotEqual(corrected["input_sha256"]["demo"]["samples.jsonl"],
                                        inputs["demo"]["samples.jsonl"])
                    self.assertEqual(json.loads((old/"job.json").read_text()),old_record)
                    run.assert_not_called()

    def test_help_labels_are_literal_text_and_click_preserves_choice(self):
        import shutil
        import subprocess
        node=shutil.which("node")
        if not node:self.skipTest("node unavailable")
        html=(Path(__file__).resolve().parents[1]/"src/index.html").read_text()
        function=html.split("function renderAutonomyHelp(items){",1)[1].split("async function loadAutonomy",1)[0]
        harness='''
const assert=require("node:assert/strict");
const element=()=>({children:[],append(...values){this.children.push(...values)},replaceChildren(){this.children=[]},set innerHTML(value){throw Error("HTML injection path")}});
const target=element(),document={createElement:element},$=()=>target;
let answered;const answerAutonomy=(...args)=>{answered=args};
'''+"function renderAutonomyHelp(items){"+function+'''
const label="<img src=x onerror=alert(1)> ' quoted",id="untrusted'identifier";
renderAutonomyHelp([{id,request:{question:label,choices:[label]}}]);
assert.equal(target.children[0].children[0].textContent,label);
const button=target.children[0].children[1].children[0];assert.equal(button.textContent,label);
button.onclick();assert.deepEqual(answered,[id,label]);
renderAutonomyHelp([]);assert.equal(target.children.length,0);
'''
        result=subprocess.run([node,"-e",harness],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_service_diagnostics_handles_timeout_and_mixed_states(self):
        import subprocess
        from service_diagnostics import inspect_services
        units={"mcu":"explorer-mcu.service","web":"explorer-web.service"}
        runner=Mock(return_value=Mock(stdout="active\ninactive\n",returncode=3))
        result=inspect_services(units,runner)
        self.assertEqual(result["services"],{"mcu":"active","web":"inactive"})
        self.assertIsNone(result["services_error"])
        runner.assert_called_once_with(["systemctl","--user","is-active",*units.values()],
                                      capture_output=True,text=True,timeout=3)
        for error in (subprocess.TimeoutExpired("systemctl",3),OSError("unavailable")):
            result=inspect_services(units,Mock(side_effect=error))
            self.assertEqual(result["services"],{"mcu":"unknown","web":"unknown"})
            self.assertTrue(result["services_error"])
        partial=inspect_services(units,Mock(return_value=Mock(stdout="active\n",returncode=0)))
        self.assertEqual(partial["services"]["web"],"unknown")
        self.assertTrue(partial["services_error"])

    def test_mapping_probe_timeout_does_not_restart_a_healthy_map(self):
        import subprocess
        import time
        from mapping_services import MappingServices
        runner=Mock(side_effect=subprocess.TimeoutExpired("systemctl",3))
        services=MappingServices(runner)
        self.assertEqual(set(services.status().values()),{"unknown"})
        self.assertTrue(services.probe_error)
        with self.assertRaisesRegex(ValueError,"перезапуск карты не выполнялся"):
            services.recover({"at":time.time(),"stop_latched":True,"velocity":[0,0,0]})
        for call in runner.call_args_list:self.assertIn("is-active",call.args[0])
