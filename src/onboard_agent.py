"""Durable bounded skill tasks over Explorer's existing actuator owners.

The registry validates every caller, including local models and MCP clients.
Restarting the process interrupts work; no old physical command is replayed.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid


TERMINAL = {"succeeded", "failed", "cancelled", "interrupted", "unknown", "waiting"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def validate(schema, value, location="args"):
    """Small strict JSON-schema subset; no coercion of booleans into numbers."""
    kind = schema["type"]
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "number": type(value) in (int, float), "integer": type(value) is int}[kind]
    if not valid:
        raise ValueError(location + ": expected " + kind)
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(location + ": unsupported value")
    if kind == "object":
        properties = schema.get("properties", {})
        if set(value) - set(properties):
            raise ValueError(location + ": unexpected fields")
        if set(schema.get("required", [])) - set(value):
            raise ValueError(location + ": missing required fields")
        for key, item in value.items():
            validate(properties[key], item, location + "." + key)
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100):
            raise ValueError(location + ": array length out of bounds")
        for index, item in enumerate(value):
            validate(schema["items"], item, location + "[" + str(index) + "]")
    if kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 1000):
            raise ValueError(location + ": text length out of bounds")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ValueError(location + ": text format invalid")
    if kind in ("number", "integer"):
        if not math.isfinite(value) or not schema.get("minimum", -1e12) <= value <= schema.get("maximum", 1e12):
            raise ValueError(location + ": number out of bounds")
    return copy.deepcopy(value)


@dataclass(frozen=True)
class SkillPort:
    name: str
    description: str
    schema: dict
    resources: tuple
    operation: object
    cancel: object = None
    units: str = "none"
    frame: str = "none"
    requirements: tuple = ()
    physical: bool = False

    def catalog(self):
        return {"name": self.name, "description": self.description, "inputSchema": self.schema,
                "resources": list(self.resources), "physical": self.physical, "units": self.units,
                "frame": self.frame, "requirements": list(self.requirements), "version": "1"}


class TaskContext:
    def __init__(self, runtime, task, step_index, timeout_s):
        self.runtime = runtime
        self.id = task["id"]
        self.generation = task["generation"]
        self.observing = task["observing"]
        self.step_index = step_index
        self.deadline = time.monotonic() + min(timeout_s, max(0, task["expires"] - time.time()))

    def permit(self):
        row = self.runtime.get(self.id, events=False)
        if row["generation"] != self.generation or row["state"] != "running":
            raise InterruptedError("Task cancelled or superseded")
        if time.monotonic() >= self.deadline or time.time() >= row["expires"]:
            raise TimeoutError("Skill deadline expired")

    def event(self, phase, details):
        self.permit()
        self.runtime.event(self.id, phase, {"step": self.step_index, **details})


class OnboardAgent:
    def __init__(self, root, ports, autostart=True):
        self.root = Path(root)
        self.path = self.root / "data/onboard-agent.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ports = {port.name: port for port in ports}
        if len(self.ports) != len(ports):
            raise ValueError("Duplicate callable skill")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", port.name) or not callable(port.operation)
               or port.schema.get("type") != "object" for port in ports):
            raise ValueError("Callable skill needs a valid name, object schema and explicit operation")
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.executing = None
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS agent_tasks(
              id TEXT PRIMARY KEY, request_id TEXT UNIQUE, state TEXT, created REAL,
              updated REAL, expires REAL, generation INTEGER, observing INTEGER,
              plan TEXT, result TEXT, current_step INTEGER);
            CREATE TABLE IF NOT EXISTS agent_events(
              id INTEGER PRIMARY KEY, task_id TEXT, at REAL, phase TEXT, details TEXT);
            CREATE TABLE IF NOT EXISTS agent_templates(
              name TEXT, version INTEGER, created REAL, plan TEXT,
              PRIMARY KEY(name,version));
            """)
            db.execute("UPDATE agent_tasks SET state=?,generation=generation+1,updated=?,result=? WHERE state IN (?,?)",
                       ("interrupted", time.time(), encoded({"reason": "Process restart; explicit new request required"}), "accepted", "running"))
        self.thread = threading.Thread(target=self.loop, daemon=True, name="explorer-skill-worker")
        if not self.templates():
            seeds = {
                "Текущее наблюдение и состояние": {"steps": [{"skill": "observe_scene"}, {"skill": "get_robot_status"}]},
                "Посмотреть по сторонам": {"steps": [{"skill": "look_direction", "args": {"view": "left"}},
                    {"skill": "observe_scene"}, {"skill": "look_direction", "args": {"view": "right"}},
                    {"skill": "observe_scene"}, {"skill": "look_direction", "args": {"view": "forward"}}]},
                "Найти носок и остановиться": {"steps": [{"skill": "find_object", "args": {"label": "sock", "places": []}}]},
                "Исследовать и показать отчёт": {"steps": [{"skill": "explore_area", "args": {"max_goals": 5}}, {"skill": "inventory_report"}]},
            }
            for name, plan in seeds.items():
                if all(step["skill"] in self.ports for step in plan["steps"]):
                    self.save_template(name, plan)
        if autostart:
            self.thread.start()

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.path, timeout=3)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def catalog(self):
        return [port.catalog() for port in self.ports.values()]

    def validate_plan(self, plan):
        if not isinstance(plan, dict) or set(plan) != {"steps"}:
            raise ValueError("Plan must contain only steps")
        if not isinstance(plan["steps"], list) or not 1 <= len(plan["steps"]) <= 12:
            raise ValueError("Plan needs 1..12 bounded steps")
        result = {"steps": []}
        for index, step in enumerate(plan["steps"]):
            if not isinstance(step, dict) or set(step) - {"skill", "args", "timeout_s", "attempts", "if_result"}:
                raise ValueError("Invalid skill step")
            name = step.get("skill")
            if name not in self.ports:
                raise ValueError("Skill is not in the callable allowlist: " + str(name))
            port = self.ports[name]
            args = validate(port.schema, step.get("args", {}))
            timeout = step.get("timeout_s", 600 if port.physical else 110)
            attempts = step.get("attempts", 1)
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 1200:
                raise ValueError("Step time budget: 1..1200 seconds")
            if type(attempts) is not int or not 1 <= attempts <= (1 if port.physical else 2):
                raise ValueError("Physical skills cannot be automatically repeated")
            accepted = {"skill": name, "args": args, "timeout_s": timeout, "attempts": attempts}
            condition = step.get("if_result")
            if condition is not None:
                if (not isinstance(condition, dict) or set(condition) != {"step", "field", "equals"}
                    or type(condition["step"]) is not int or not 0 <= condition["step"] < index
                    or condition["field"] not in ("state", "found")
                    or type(condition["equals"]) not in (bool, str)):
                    raise ValueError("Conditions refer only to earlier observable state/found fields")
                accepted["if_result"] = copy.deepcopy(condition)
            result["steps"].append(accepted)
        if len(encoded(result)) > 16000:
            raise ValueError("Plan too large")
        return result

    def submit(self, request_id, plan, observing=False, ttl_s=1200):
        if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{16,100}", request_id):
            raise ValueError("A stable request_id of 16..100 characters is required")
        plan = self.validate_plan(plan)
        if type(observing) is not bool or type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or not 1 <= ttl_s <= 1200:
            raise ValueError("Invalid authorization or deadline")
        if any(self.ports[step["skill"]].physical for step in plan["steps"]) and not observing:
            raise ValueError("Confirm the prepared area and observation before a physical task")
        value = encoded(plan)
        with self.lock, self.db() as db:
            old = db.execute("SELECT * FROM agent_tasks WHERE request_id=?", (request_id,)).fetchone()
            if old:
                if old["plan"] != value or bool(old["observing"]) != observing:
                    raise ValueError("request_id belongs to a different command")
                identifier = old["id"]
            else:
                identifier = uuid.uuid4().hex
                now = time.time()
                db.execute("INSERT INTO agent_tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (identifier, request_id, "accepted", now, now, now + ttl_s, 1, int(observing), value, "{}", -1))
                self._event(db, identifier, "accepted", {"plan": plan, "expires_at": now + ttl_s})
        self.wake.set()
        return self.get(identifier)

    @staticmethod
    def row(row):
        result = dict(row)
        result["plan"] = json.loads(result["plan"])
        result["result"] = json.loads(result["result"])
        result["observing"] = bool(result["observing"])
        return result

    def get(self, identifier, events=True):
        with self.db() as db:
            row = db.execute("SELECT * FROM agent_tasks WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("Task not found")
            result = self.row(row)
            if events:
                result["events"] = [dict(item, details=json.loads(item["details"])) for item in db.execute(
                    "SELECT * FROM agent_events WHERE task_id=? ORDER BY id LIMIT 500", (identifier,))]
        return result

    def status(self):
        with self.db() as db:
            tasks = [self.row(row) for row in db.execute("SELECT * FROM agent_tasks ORDER BY created DESC LIMIT 30")]
        return {"tasks": tasks, "busy": self.executing is not None, "catalog": self.catalog(),
                "browser_required": False, "motor_owner": "existing Explorer executors", "version": 1}

    @staticmethod
    def _event(db, identifier, phase, details):
        db.execute("INSERT INTO agent_events(task_id,at,phase,details) VALUES(?,?,?,?)",
                   (identifier, time.time(), phase, encoded(details)))

    def event(self, identifier, phase, details):
        with self.db() as db:
            self._event(db, identifier, phase, details)

    def finish(self, task, state, result):
        with self.db() as db:
            updated = db.execute("UPDATE agent_tasks SET state=?,updated=?,result=? WHERE id=? AND generation=? AND state=?",
                                 (state, time.time(), encoded(result), task["id"], task["generation"], "running"))
            if updated.rowcount:
                self._event(db, task["id"], state, result)

    def cancel(self, identifier, reason="Operator cancellation"):
        with self.lock:
            task = self.get(identifier, events=False)
            if task["state"] in TERMINAL:
                return task
            with self.db() as db:
                db.execute("UPDATE agent_tasks SET state=?,generation=generation+1,updated=?,result=? WHERE id=?",
                           ("cancelled", time.time(), encoded({"reason": reason}), identifier))
                self._event(db, identifier, "cancelled", {"reason": reason})
            execution = self.executing
        if execution and execution[0] == identifier:
            port = self.ports[execution[1]]
            if port.cancel:
                try:
                    port.cancel(identifier)
                except Exception as exc:
                    self.event(identifier, "cancel_error", {"reason": str(exc), "physical_stop_confirmed": False})
        self.wake.set()
        return self.get(identifier)

    def cancel_all(self, reason="STOP or manual takeover"):
        with self.db() as db:
            identifiers = [row[0] for row in db.execute("SELECT id FROM agent_tasks WHERE state IN (?,?)", ("accepted", "running"))]
        return {"cancelled": [self.cancel(identifier, reason)["id"] for identifier in identifiers]}

    def cancel_resources(self, resources, reason="Manual resource takeover"):
        resources = set(resources)
        with self.db() as db:
            tasks = [self.row(row) for row in db.execute("SELECT * FROM agent_tasks WHERE state IN (?,?)", ("accepted", "running"))]
        identifiers = [task["id"] for task in tasks if any(resources.intersection(self.ports[step["skill"]].resources)
                       for step in task["plan"]["steps"])]
        return {"cancelled": [self.cancel(identifier, reason)["id"] for identifier in identifiers]}

    def templates(self):
        with self.db() as db:
            return [dict(row, plan=json.loads(row["plan"])) for row in db.execute(
                "SELECT * FROM agent_templates ORDER BY name,version DESC")]

    def save_template(self, name, plan):
        if not isinstance(name, str) or not re.fullmatch(r"[\w -]{1,64}", name):
            raise ValueError("Template name needs 1..64 letters, numbers, spaces or dashes")
        plan = self.validate_plan(plan)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("SELECT coalesce(max(version),0)+1 FROM agent_templates WHERE name=?", (name,)).fetchone()[0]
            db.execute("INSERT INTO agent_templates VALUES(?,?,?,?)", (name, version, time.time(), encoded(plan)))
        return {"name": name, "version": version, "plan": plan}

    def loop(self):
        while not self.closed.is_set():
            with self.db() as db:
                row = db.execute("SELECT * FROM agent_tasks WHERE state=? ORDER BY created LIMIT 1", ("accepted",)).fetchone()
            if row:
                self.execute(self.row(row))
            else:
                self.wake.wait(.2)
                self.wake.clear()

    def execute(self, task):
        with self.lock, self.db() as db:
            changed = db.execute("UPDATE agent_tasks SET state=?,updated=? WHERE id=? AND state=? AND generation=?",
                                 ("running", time.time(), task["id"], "accepted", task["generation"]))
            if not changed.rowcount:
                return
        results = []
        state = "succeeded"
        try:
            for index, step in enumerate(task["plan"]["steps"]):
                port = self.ports[step["skill"]]
                context = TaskContext(self, task, index, step["timeout_s"])
                context.permit()
                condition = step.get("if_result")
                skipped = False
                if condition:
                    previous = results[condition["step"]]
                    observed = previous.get("state") if condition["field"] == "state" else previous.get("evidence", {}).get("found")
                    skipped = observed != condition["equals"]
                if skipped:
                    result = {"state": "skipped", "evidence": {"condition": condition, "observed": observed}}
                else:
                    with self.lock:
                        context.permit()
                        self.executing = (task["id"], port.name)
                    with self.db() as db:
                        db.execute("UPDATE agent_tasks SET current_step=?,updated=? WHERE id=?", (index, time.time(), task["id"]))
                    context.event("skill_started", {"skill": port.name, "args": step["args"], "resources": list(port.resources)})
                    result = None
                    for attempt in range(step["attempts"]):
                        context.permit()
                        try:
                            result = port.operation(copy.deepcopy(step["args"]), context)
                            break
                        except ValueError:
                            if attempt + 1 == step["attempts"]:
                                raise
                            context.event("read_retry", {"attempt": attempt + 1})
                    context.permit()
                    if not isinstance(result, dict) or result.get("state") not in {"succeeded", "failed", "unknown", "waiting"} or "evidence" not in result:
                        raise ValueError("Executor did not return an evidence-bearing outcome")
                    encoded(result)
                    context.event("skill_finished", {"skill": port.name, **result})
                results.append(result)
                if result["state"] not in {"succeeded", "skipped"}:
                    state = result["state"]
                    break
            self.finish(task, state, {"steps": results, "all_requested_steps_completed": len(results) == len(task["plan"]["steps"]) and state == "succeeded"})
        except InterruptedError:
            pass
        except TimeoutError as exc:
            self.cancel_owned(task["id"])
            self.finish(task, "failed", {"reason": str(exc), "steps": results})
        except ValueError as exc:
            self.finish(task, "waiting", {"reason": str(exc), "steps": results, "automatic_retry": False})
        except Exception as exc:
            self.cancel_owned(task["id"])
            self.finish(task, "failed", {"reason": str(exc), "steps": results})
        finally:
            with self.lock:
                if self.executing and self.executing[0] == task["id"]:
                    self.executing = None

    def cancel_owned(self, identifier):
        execution = self.executing
        if execution and execution[0] == identifier:
            port = self.ports[execution[1]]
            if port.cancel:
                try:
                    port.cancel(identifier)
                except Exception as exc:
                    self.event(identifier, "cancel_error", {"reason": str(exc), "physical_stop_confirmed": False})

    def shutdown(self):
        self.cancel_all("Agent worker shutdown")
        self.closed.set()
        self.wake.set()
