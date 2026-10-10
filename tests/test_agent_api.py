from pathlib import Path
import tempfile
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from onboard_agent import OnboardAgent, SkillPort
from agent_ports import object_schema
from agent_api import install_agent_api


class AgentApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []
        def read(args, context):
            self.calls.append(context.id)
            return {"state": "succeeded", "evidence": {"value": 7}}
        ports = [SkillPort("read", "read", object_schema(), (), read),
                 SkillPort("move", "move", object_schema(), ("arm",), read, physical=True)]
        self.runtime = OnboardAgent(Path(self.temp.name), ports, autostart=False)
        app = FastAPI()
        install_agent_api(app, self.temp.name, self.runtime, "test-secret", lambda: [], lambda: {})
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test-secret"}

    def rpc(self, method, params=None):
        return self.client.post("/api/skill-agent/mcp", headers=self.headers,
                                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).json()

    def test_authenticated_and_origin_checked(self):
        self.assertEqual(self.client.get("/api/skill-agent/catalog").status_code, 401)
        result = self.client.get("/api/skill-agent/catalog", headers=self.headers)
        self.assertEqual(result.status_code, 200)
        cross = self.client.get("/api/skill-agent/catalog", headers={**self.headers, "Origin": "https://other.example"})
        self.assertEqual(cross.status_code, 403)

    def test_openapi_has_typed_protected_routes(self):
        result = self.client.get("/api/skill-agent/openapi", headers=self.headers).json()
        self.assertIn("/api/skill-agent/tasks", result["paths"])
        self.assertIn("AgentTaskRequest", result["components"]["schemas"])

    def test_rpc_reuses_durable_registry(self):
        self.assertEqual(self.rpc("initialize")["result"]["protocolVersion"], "2025-03-26")
        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual({tool["name"] for tool in tools}, {"read", "move", "get_task", "cancel_task"})
        result = self.rpc("tools/call", {"name": "read", "arguments": {"request_id": "mcp_request_000001", "args": {}}})
        self.assertFalse(result["result"]["isError"])
        self.assertEqual(len(self.runtime.status()["tasks"]), 1)
        self.assertEqual(self.calls, [])

    def test_mcp_model_cannot_manufacture_physical_permission(self):
        rejected = self.rpc("tools/call", {"name": "move", "arguments": {"request_id": "mcp_request_000001", "args": {}, "observing": True}})
        self.assertIn("error", rejected)
        no_permission = self.rpc("tools/call", {"name": "move", "arguments": {"request_id": "mcp_request_000001", "args": {}}})
        self.assertIn("fresh explicit operator permission", no_permission["error"]["message"])
        self.assertEqual(self.runtime.status()["tasks"], [])

    def test_grant_has_one_skill_one_request_scope(self):
        endpoint = "/api/skill-agent/mcp-permission/move"
        self.assertEqual(self.client.post(endpoint, headers=self.headers, json={}).status_code, 409)
        granted = self.client.post(endpoint, headers=self.headers, json={"observing": True}).json()
        arguments = {"request_id": "mcp_request_000001", "args": {}, "authorization_id": granted["authorization_id"]}
        self.assertIn("result", self.rpc("tools/call", {"name": "move", "arguments": arguments}))
        self.assertIn("result", self.rpc("tools/call", {"name": "move", "arguments": arguments}))
        second = self.rpc("tools/call", {"name": "move", "arguments": {**arguments, "request_id": "mcp_request_000002"}})
        self.assertIn("already used", second["error"]["message"])
        self.assertEqual(len(self.runtime.status()["tasks"]), 1)

    def test_image_is_bound_to_recorded_task_not_latest_detection(self):
        search_id = "a" * 32
        folder = Path(self.temp.name) / "data/object-searches" / search_id
        folder.mkdir(parents=True)
        (folder / "result.jpg").write_bytes(b"old-task-image")
        self.runtime.ports["capture"] = SkillPort("capture", "capture", object_schema(), (),
            lambda args, context: {"state": "succeeded", "evidence": {"observation": {"id": search_id}}})
        captured = self.runtime.submit("capture_request_001", {"steps": [{"skill": "capture"}]})
        self.runtime.execute(captured)
        unrelated = self.runtime.submit("unrelated_request_1", {"steps": [{"skill": "read"}]})
        owned = self.client.get("/api/skill-agent/tasks/" + captured["id"] + "/images/" + search_id, headers=self.headers)
        self.assertEqual(owned.content, b"old-task-image")
        denied = self.client.get("/api/skill-agent/tasks/" + unrelated["id"] + "/images/" + search_id, headers=self.headers)
        self.assertEqual(denied.status_code, 404)


if __name__ == "__main__":
    unittest.main()
