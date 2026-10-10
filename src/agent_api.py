"""Protected additive REST/OpenAPI and thin stateless MCP for the same registry."""
import asyncio
import json
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlparse
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from agent_intent import prepare_intent


class AgentPlanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1500)


class AgentTaskRequest(BaseModel):
    request_id: str = Field(min_length=16, max_length=100)
    plan: dict
    observing: bool = False
    ttl_s: float = Field(default=1200, ge=1, le=1200)


class AgentTemplateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    plan: dict


class AgentPermissionRequest(BaseModel):
    observing: bool = False


def install_agent_api(app, root, runtime, token, places, stop, model=None):
    root = Path(root)
    # MCP cannot manufacture observing=true. Operators grant one bounded skill
    # separately through the protected REST/UI flow; grants vanish on restart.
    grants = {}

    def auth(request: Request):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not token or not secrets.compare_digest(supplied, token):
            raise HTTPException(401, "Authentication required")
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Cross-origin control blocked")

    def operation(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    router = APIRouter(prefix="/api/skill-agent", dependencies=[Depends(auth)], tags=["Onboard skill agent"])

    @app.get("/agent", include_in_schema=False)
    def agent_page():
        return HTMLResponse((root / "src/agent.html").read_text(), headers={"Cache-Control": "no-store"})

    @app.get("/agent-ui.js", include_in_schema=False)
    def agent_script():
        return Response((root / "src/agent-ui.js").read_text(), media_type="text/javascript", headers={"Cache-Control": "no-store"})

    @router.get("/catalog")
    def catalog():
        return {"skills": runtime.catalog(), "templates": runtime.templates()}

    @router.get("/status")
    def status():
        return runtime.status()

    @router.post("/plan")
    def plan(request: AgentPlanRequest):
        result = operation(prepare_intent, request.text, runtime, places(), model)
        if result.get("stop_requested"):
            runtime.cancel_all("Voice/text STOP")
            result["result"] = stop()
        return result

    @router.post("/tasks")
    def submit(request: AgentTaskRequest):
        return operation(runtime.submit, request.request_id, request.plan, request.observing, request.ttl_s)

    @router.get("/tasks/{identifier}")
    def task(identifier: str):
        return operation(runtime.get, identifier)

    @router.post("/tasks/{identifier}/cancel")
    def cancel(identifier: str):
        return operation(runtime.cancel, identifier)

    @router.get("/tasks/{identifier}/images/{search_id}")
    def task_image(identifier: str, search_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", search_id):
            raise HTTPException(404, "Image not found")
        task = operation(runtime.get, identifier)
        observations = [step.get("evidence", {}).get("observation", {}) for step in task.get("result", {}).get("steps", [])]
        observations += [event.get("details", {}).get("evidence", {}).get("observation", {}) for event in task.get("events", [])]
        if not any(observation.get("id") == search_id for observation in observations):
            raise HTTPException(404, "Image does not belong to this task evidence")
        path = root / "data/object-searches" / search_id / "result.jpg"
        if not path.is_file():
            raise HTTPException(404, "Recorded image unavailable")
        return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=120"})

    @router.get("/tasks/{identifier}/events")
    async def events(identifier: str, request: Request):
        operation(runtime.get, identifier)

        async def stream():
            cursor = 0
            for _ in range(240):
                if await request.is_disconnected():
                    break
                row = runtime.get(identifier)
                pending = [event for event in row["events"] if event["id"] > cursor]
                for event in pending:
                    cursor = event["id"]
                    yield "id: " + str(cursor) + "\nevent: skill\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                if row["state"] not in ("accepted", "running"):
                    yield "event: result\ndata: " + json.dumps(row, ensure_ascii=False) + "\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(.5)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @router.get("/templates")
    def templates():
        return runtime.templates()

    @router.post("/templates")
    def template(request: AgentTemplateRequest):
        return operation(runtime.save_template, request.name, request.plan)

    @router.get("/openapi")
    def openapi():
        full = app.openapi()
        components = dict(full.get("components", {}))
        components["securitySchemes"] = {"ExplorerBearer": {"type": "http", "scheme": "bearer"}}
        return {**full, "paths": {path: item for path, item in full["paths"].items() if path.startswith("/api/skill-agent")},
                "components": components, "security": [{"ExplorerBearer": []}]}

    @router.post("/mcp-permission/{skill}")
    def mcp_permission(skill: str, request: AgentPermissionRequest):
        if request.observing is not True:
            raise HTTPException(409, "Confirm prepared area and operator observation")
        if skill not in runtime.ports or not runtime.ports[skill].physical:
            raise HTTPException(409, "Select one physical skill")
        identifier = secrets.token_urlsafe(24)
        grants[identifier] = {"skill": skill, "expires": time.monotonic() + 120, "request_id": None}
        return {"authorization_id": identifier, "skill": skill, "expires_in_s": 120,
                "claim": "One explicit operator grant; does not clear STOP or readiness checks"}

    @router.post("/mcp")
    async def mcp(request: Request):
        body = await request.json()
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        identifier = body.get("id")
        method = body["method"]
        params = body.get("params") or {}
        if not isinstance(params, dict):
            return {"jsonrpc": "2.0", "id": identifier, "error": {"code": -32602, "message": "params must be an object"}}
        if method == "notifications/initialized":
            return Response(status_code=202)
        try:
            if method == "initialize":
                result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "explorer-skills", "version": "1.0"},
                          "instructions": "Physical skills return durable task IDs. Inspect get_task for actual results; accepted does not mean completed."}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                tools = []
                for port in runtime.ports.values():
                    schema = {"type": "object", "properties": {"args": port.schema,
                              "request_id": {"type": "string", "minLength": 16, "maxLength": 100},
                              "authorization_id": {"type": "string"}},
                              "required": ["args", "request_id"], "additionalProperties": False}
                    tools.append({"name": port.name, "description": port.description, "inputSchema": schema})
                for name, description in (("get_task", "Read a durable task and its evidence"), ("cancel_task", "Cancel only this task; no HOME or gripper release")):
                    tools.append({"name": name, "description": description, "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"], "additionalProperties": False}})
                result = {"tools": tools}
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                if name in ("get_task", "cancel_task"):
                    if not isinstance(arguments, dict) or set(arguments) != {"task_id"}:
                        raise ValueError("Only task_id is accepted")
                    outcome = runtime.get(arguments["task_id"]) if name == "get_task" else runtime.cancel(arguments["task_id"])
                else:
                    if not isinstance(arguments, dict) or set(arguments) - {"args", "request_id", "authorization_id"}:
                        raise ValueError("Invalid tool invocation envelope")
                    observing = False
                    if name in runtime.ports and runtime.ports[name].physical:
                        permission = grants.get(arguments.get("authorization_id"))
                        request_id = arguments.get("request_id")
                        if not permission or permission["skill"] != name or time.monotonic() >= permission["expires"]:
                            raise ValueError("Physical MCP skill needs a fresh explicit operator permission")
                        if permission["request_id"] not in (None, request_id):
                            raise ValueError("MCP permission was already used by another request")
                        permission["request_id"] = request_id
                        observing = True
                    outcome = runtime.submit(arguments.get("request_id"), {"steps": [{"skill": name, "args": arguments.get("args", {})}]}, observing)
                result = {"content": [{"type": "text", "text": json.dumps(outcome, ensure_ascii=False)}], "isError": False}
            else:
                return {"jsonrpc": "2.0", "id": identifier, "error": {"code": -32601, "message": "Method not found"}}
            return {"jsonrpc": "2.0", "id": identifier, "result": result}
        except (ValueError, TypeError, KeyError) as exc:
            return {"jsonrpc": "2.0", "id": identifier, "error": {"code": -32602, "message": str(exc)}}

    app.include_router(router)
    return runtime
