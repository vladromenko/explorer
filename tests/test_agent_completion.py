"""Regressions for offline composition, real outcomes and map-bound return."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

from agent_intent import prepare_intent
from agent_ports import ExplorerPorts, object_schema
from onboard_agent import OnboardAgent, SkillPort
from semantic_world import SemanticWorld


class OfflineIntentTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(validate_plan=lambda value: value, ports={name: SimpleNamespace(physical=name not in {"inventory_report", "query_task_memory"}) for name in (
            "visit_and_return", "navigate_to", "recover_object_view", "explore_area", "follow_route", "inventory_report", "query_task_memory", "deliver_object")})
        self.places = [{"name": "Дверь", "compatible_map": True}, {"name": "Кресло", "compatible_map": True}, {"name": "Старая карта", "compatible_map": False}]

    def test_visit_return_is_one_bounded_skill_without_llm(self):
        result = prepare_intent("Поезжай к месту «Дверь» и вернись обратно", self.runtime, self.places)
        self.assertEqual(result["plan"]["steps"], [{"skill": "visit_and_return", "args": {"place": "Дверь"}}])
        self.assertEqual(result["source"], "deterministic_template")
        self.assertTrue(result["physical"])
        self.assertIsNone(prepare_intent("Поезжай к кухня и вернись", self.runtime, self.places)["plan"])

    def test_search_recovery_and_report_commands_are_offline(self):
        result = prepare_intent("Восстанови обзор носка", self.runtime)
        self.assertEqual(result["plan"]["steps"][0], {"skill": "recover_object_view", "args": {"label": "носок"}})
        result = prepare_intent("Обойди сохранённые места и покажи отчёт", self.runtime, self.places)
        self.assertEqual(result["plan"]["steps"], [{"skill": "follow_route", "args": {"places": ["Дверь", "Кресло"]}}, {"skill": "inventory_report", "args": {}}])
        self.assertIsNone(prepare_intent("Обойди все места и покажи отчёт", self.runtime)["plan"])
        self.assertEqual(prepare_intent("история задач", self.runtime)["plan"]["steps"][0]["skill"], "query_task_memory")

    def test_sock_delivery_preserves_exact_named_destination_and_no_acceptance(self):
        result = prepare_intent("Возьми носок и положи в место «Кресло»", self.runtime, self.places)
        self.assertEqual(result["plan"]["steps"], [{"skill": "deliver_object", "args": {"label": "носок", "destination": "Кресло"}}])
        self.assertFalse(result["executed"])
        self.assertEqual(result["source"], "deterministic_template")
        self.assertIsNone(prepare_intent("Возьми носок и положи в корзину", self.runtime, self.places)["plan"])
        self.assertIsNone(prepare_intent("Возьми синий кубик и положи в Кресло", self.runtime, self.places)["plan"])


class ContinualAgentMemoryTests(unittest.TestCase):
    def test_ready_camera_view_supports_next_inspection_without_measured_joints(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            calls = []
            def move(view, permit, progress=None):
                permit()
                settled = time.time() - .01
                stamp = time.time()
                (root / "data/perception.json").write_text(json.dumps({"image_stamp": stamp,
                    "objects": [{"label": "sock", "confidence": .9}]}))
                calls.append(view)
                return {"phase": "ready", "view": view, "settled_at": settled, "image_stamp": stamp,
                    "pose_source": "command_estimate", "measured_joints": False}
            views = SimpleNamespace(move=move)
            ports = ExplorerPorts(root, lambda: {}, None, None, None, None, views, None, None, None, None)
            selected = [port for port in ports.catalog() if port.name in {"look_direction", "observe_scene"}]
            agent = OnboardAgent(root, selected, autostart=False)
            task = agent.submit("camera_then_inspect", {"steps": [
                {"skill": "look_direction", "args": {"view": "down"}}, {"skill": "observe_scene"}]}, True)
            agent.execute(task)
            result = agent.get(task["id"])
            self.assertEqual(result["state"], "succeeded")
            self.assertTrue(result["result"]["all_requested_steps_completed"])
            self.assertEqual(calls, ["down"])
            self.assertFalse(result["result"]["steps"][0]["evidence"]["measured_joints"])
            self.assertEqual(result["result"]["steps"][1]["evidence"]["objects"][0]["label"], "sock")

    def test_action_outcome_is_logged_without_reusing_old_after_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            world = SemanticWorld(Path(folder))
            world.ingest({"image_stamp": time.time() - 5, "objects": []})
            def operation(args, context):
                context.permit()
                return {"state": "unknown", "evidence": {"pose_source": "command_estimate"}}
            port = SkillPort("move", "move", object_schema(), ("arm",), operation, physical=True)
            agent = OnboardAgent(Path(folder), [port], autostart=False, world=world)
            task = agent.submit("physical_unknown_001", {"steps": [{"skill": "move"}]}, True)
            agent.execute(task)
            rows = world.action_history()
            self.assertEqual([row["state"] for row in rows], ["unknown", "started"])
            self.assertEqual(rows[0]["task"], task["id"])
            self.assertIsNotNone(rows[0]["before_episode"])
            self.assertIsNone(rows[0]["after_episode"])
            self.assertEqual(agent.get(task["id"])["state"], "unknown")

    def test_new_observation_is_attached_to_actual_skill_outcome(self):
        with tempfile.TemporaryDirectory() as folder:
            world = SemanticWorld(Path(folder))
            before = world.ingest({"image_stamp": time.time() - 5, "objects": []})
            def operation(args, context):
                world.ingest({"image_stamp": time.time(), "objects": [{"label": "sock", "confidence": .9}]})
                return {"state": "succeeded", "evidence": {"found": True, "identity_verified": False}}
            agent = OnboardAgent(Path(folder), [SkillPort("detect", "detect", object_schema(), (), operation)], autostart=False, world=world)
            task = agent.submit("fresh_memory_request", {"steps": [{"skill": "detect"}]})
            agent.execute(task)
            row = world.action_history()[0]
            self.assertEqual(row["before_episode"], before["episode"])
            self.assertEqual(row["after_episode"], world.latest_episode()["id"])
            self.assertEqual(row["result"]["evidence"]["found"], True)

    def test_memory_failure_does_not_change_real_skill_result(self):
        with tempfile.TemporaryDirectory() as folder:
            def fail():
                raise OSError("memory disk unavailable")
            world = SimpleNamespace(latest_episode=fail)
            port = SkillPort("read", "read", object_schema(), (), lambda args, ctx: {"state": "succeeded", "evidence": {"stamp": time.time()}})
            agent = OnboardAgent(Path(folder), [port], autostart=False, world=world)
            task = agent.submit("memory_failure_read", {"steps": [{"skill": "read"}]})
            agent.execute(task)
            result = agent.get(task["id"])
            self.assertEqual(result["state"], "succeeded")
            self.assertTrue(any(event["phase"] == "memory_error" for event in result["events"]))


class ReturnPortTests(unittest.TestCase):
    def make_ports(self):
        maps = SimpleNamespace(pose=lambda: {"x": 1., "y": 2., "yaw": .3, "provisional": True}, epoch=lambda: "map-a")
        missions = SimpleNamespace(maps=maps, places=lambda: [{"name": "Door", "x": 3., "y": 4., "yaw": .5, "compatible_map": True}])
        ports = ExplorerPorts("/unused", lambda: {}, missions, None, None, None, None, None, None, None, None)
        events = []
        context = SimpleNamespace(permit=lambda: None, event=lambda phase, details: events.append((phase, details)))
        calls = []
        ports.navigation_skill = lambda spec, ctx: calls.append(spec) or {"state": "succeeded", "evidence": {"map_accuracy": "provisional"}}
        return ports, context, calls

    def test_return_uses_observed_origin_and_same_executor(self):
        ports, context, calls = self.make_ports()
        result = ports.visit_and_return({"place": "Door"}, context)
        self.assertEqual(calls, [{"kind": "navigate_current", "x": 3., "y": 4., "yaw": .5}, {"kind": "navigate_current", "x": 1., "y": 2., "yaw": .3}])
        self.assertTrue(result["evidence"]["returned_to_start"])
        self.assertFalse(result["evidence"]["physical_accuracy_measured"])

    def test_failed_outbound_does_not_blindly_return(self):
        ports, context, calls = self.make_ports()
        ports.navigate = lambda args, ctx: {"state": "failed", "evidence": {"reason": "blocked"}}
        result = ports.visit_and_return({"place": "Door"}, context)
        self.assertEqual(calls, [])
        self.assertEqual(result["state"], "failed")
        self.assertFalse(result["evidence"]["returned_to_start"])

    def test_changed_map_invalidates_saved_return(self):
        ports, context, calls = self.make_ports()
        def changed(args, ctx):
            ports.missions.maps.epoch = lambda: "map-b"
            return {"state": "succeeded", "evidence": {}}
        ports.navigate = changed
        with self.assertRaisesRegex(ValueError, "Map frame changed"):
            ports.visit_and_return({"place": "Door"}, context)
        self.assertEqual(calls, [])
