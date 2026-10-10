from pathlib import Path
import tempfile
import threading
import time
import unittest
from onboard_agent import OnboardAgent, SkillPort, validate
from agent_intent import prepare_intent
from agent_ports import ExplorerPorts, object_schema


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.cancelled = []
        self.port = SkillPort("read", "read", object_schema(), (), self.read)
        self.runtime = OnboardAgent(self.root, [self.port], autostart=False)

    def read(self, args, context):
        context.permit()
        self.calls.append(context.id)
        return {"state": "succeeded", "evidence": {"stamp": time.time()}}

    def submit(self, skill="read", args=None, observing=False, request_id="stable_request_0001"):
        return self.runtime.submit(request_id, {"steps": [{"skill": skill, "args": args or {}}]}, observing)

    def test_idempotency_does_not_repeat_action(self):
        first = self.submit()
        self.runtime.execute(first)
        second = self.submit()
        self.runtime.execute(second)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second["state"], "succeeded")

    def test_rejects_unknown_skill_and_unexpected_args(self):
        for name, args in (("shell", {}), ("read", {"command": "motor"})):
            with self.assertRaises(ValueError):
                self.submit(name, args)

    def test_bool_nan_are_not_valid_numbers(self):
        for value in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate({"type": "number"}, value)

    def test_motion_needs_observer_and_cannot_retry(self):
        self.runtime.ports["move"] = SkillPort("move", "move", object_schema(), ("arm",), self.read, physical=True)
        with self.assertRaises(ValueError):
            self.submit("move")
        with self.assertRaises(ValueError):
            self.runtime.submit("request_with_retries", {"steps": [{"skill": "move", "attempts": 2}]}, True)
        self.assertEqual(self.submit("move", observing=True)["state"], "accepted")

    def test_late_result_is_discarded_after_cancel(self):
        started = threading.Event()
        release = threading.Event()
        def operation(args, context):
            started.set()
            release.wait(2)
            return {"state": "succeeded", "evidence": {"late": True}}
        self.runtime.ports["slow"] = SkillPort("slow", "slow", object_schema(), ("arm",), operation,
                                              lambda identifier: self.cancelled.append(identifier), physical=True)
        task = self.submit("slow", observing=True)
        thread = threading.Thread(target=self.runtime.execute, args=(task,))
        thread.start()
        self.assertTrue(started.wait(1))
        self.runtime.cancel(task["id"])
        release.set()
        thread.join(2)
        result = self.runtime.get(task["id"])
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(self.cancelled, [task["id"]])
        self.assertFalse(any(event["phase"] == "skill_finished" for event in result["events"]))

    def test_restart_never_resumes_queued_movement(self):
        task = self.submit()
        recovered = OnboardAgent(self.root, [self.port], autostart=False)
        self.assertEqual(recovered.get(task["id"])["state"], "interrupted")
        self.assertEqual(self.calls, [])

    def test_manual_base_takeover_preserves_read_and_arm_only_tasks(self):
        self.runtime.ports["base"] = SkillPort("base", "base", object_schema(), ("base", "navigation"), self.read, physical=True)
        self.runtime.ports["gaze"] = SkillPort("gaze", "gaze", object_schema(), ("arm", "camera"), self.read, physical=True)
        read = self.submit(request_id="readonly_request_001")
        base = self.submit("base", observing=True, request_id="base_request_000001")
        gaze = self.submit("gaze", observing=True, request_id="gaze_request_000001")
        result = self.runtime.cancel_resources(("base", "navigation"))
        self.assertEqual(result["cancelled"], [base["id"]])
        self.assertEqual(self.runtime.get(read["id"])["state"], "accepted")
        self.assertEqual(self.runtime.get(gaze["id"])["state"], "accepted")
        self.assertEqual(self.runtime.cancel_resources(("arm",))["cancelled"], [gaze["id"]])

    def test_command_ack_is_not_a_result(self):
        self.runtime.ports["ack"] = SkillPort("ack", "ack", object_schema(), (), lambda args, ctx: {"accepted": True, "completed": False})
        task = self.submit("ack")
        self.runtime.execute(task)
        result = self.runtime.get(task["id"])
        self.assertEqual(result["state"], "waiting")
        self.assertIn("evidence-bearing", result["result"]["reason"])

    def test_unknown_stops_remaining_steps(self):
        self.runtime.ports["unknown"] = SkillPort("unknown", "unknown", object_schema(), (), lambda args, ctx: {"state": "unknown", "evidence": {"pose_source": "command_estimate"}})
        task = self.runtime.submit("unknown_request_001", {"steps": [{"skill": "unknown"}, {"skill": "read"}]})
        self.runtime.execute(task)
        result = self.runtime.get(task["id"])
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(self.calls, [])
        self.assertFalse(result["result"]["all_requested_steps_completed"])

    def test_expired_task_does_not_invoke_port(self):
        task = self.submit()
        with self.runtime.db() as db:
            db.execute("UPDATE agent_tasks SET expires=? WHERE id=?", (time.time() - 1, task["id"]))
        self.runtime.execute(self.runtime.get(task["id"]))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.runtime.get(task["id"])["state"], "failed")

    def test_stop_error_does_not_kill_worker_or_claim_success(self):
        def operation(args, context):
            raise TimeoutError("motion deadline")
        def cancel(identifier):
            raise OSError("transport unavailable")
        self.runtime.ports["timeout"] = SkillPort("timeout", "timeout", object_schema(), ("arm",), operation, cancel, physical=True)
        task = self.submit("timeout", observing=True)
        self.runtime.execute(task)
        result = self.runtime.get(task["id"])
        self.assertEqual(result["state"], "failed")
        self.assertTrue(any(event["phase"] == "cancel_error" for event in result["events"]))
        following = self.submit(request_id="following_read_0001")
        self.runtime.execute(following)
        self.assertEqual(self.runtime.get(following["id"])["state"], "succeeded")

    def test_templates_keep_old_version_and_task_snapshot(self):
        plan = {"steps": [{"skill": "read"}]}
        first = self.runtime.save_template("Осмотр", plan)
        task = self.runtime.submit("snapshot_request_001", first["plan"])
        second = self.runtime.save_template("Осмотр", {"steps": [{"skill": "read"}, {"skill": "read"}]})
        self.assertEqual(second["version"], 2)
        self.assertEqual(len(self.runtime.templates()), 2)
        self.assertEqual(len(self.runtime.get(task["id"])["plan"]["steps"]), 1)

    def test_prompt_model_cannot_add_authority_or_shell(self):
        for response in ({"steps": [{"skill": "shell", "args": {}}]}, {"steps": [{"skill": "read"}], "observing": True}):
            with self.assertRaises(ValueError):
                prepare_intent("необычная команда", self.runtime, model=lambda text, catalog, places: response)

    def test_condition_uses_observed_found_and_does_not_eval(self):
        self.runtime.ports["detect"] = SkillPort("detect", "detect", object_schema(), (), lambda args, ctx: {"state": "succeeded", "evidence": {"found": False}})
        task = self.runtime.submit("conditional_request", {"steps": [{"skill": "detect"}, {"skill": "read", "if_result": {"step": 0, "field": "found", "equals": True}}]})
        self.runtime.execute(task)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.runtime.get(task["id"])["result"]["steps"][1]["state"], "skipped")


class PortTests(unittest.TestCase):
    def test_cancel_is_scoped_to_actual_child_owner(self):
        class Navigation:
            def status(self):
                return {"active": {"id": "another_operator"}}
            def cancel(self):
                raise AssertionError("Must not cancel another owner")
        ports = ExplorerPorts("/unused", lambda: {}, None, Navigation(), None, None, None, None, None, None, None)
        ports.owned["parent"] = ("navigation", "our_child")
        ports.cancel("parent")

    def test_gripper_does_not_reset_other_joints(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config").mkdir()
            (root / "config/local-grasp-profile.json").write_text(json.dumps({"experimental_execution_authorized": True, "open_command_deg": 30, "hold_command_deg": 160}))
            class Arm:
                def reference(self):
                    return {"servo_deg": [80, 75, 65, 45, 120, 90]}
            ports = ExplorerPorts(root, lambda: {}, None, None, None, None, None, Arm(), None, None, None)
            calls = []
            def move(args, context):
                calls.append(args)
                return {"state": "unknown", "evidence": {}}
            ports.move = move
            result = ports.gripper({}, None, True)
            self.assertEqual(calls[0]["goal_deg"], [80, 75, 65, 45, 120, 30])
            self.assertFalse(result["evidence"]["contact_verified"])


if __name__ == "__main__":
    unittest.main()
