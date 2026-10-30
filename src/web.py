"""Minimal local control panel: manual motion, camera and sensor telemetry only."""
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from urllib.parse import urlparse

import rclpy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import uvicorn

ROOT = Path(os.environ.get("EXPLORER_ROOT", "/home/vlad/Explorer"))
TOKEN = (ROOT / "config/access_token").read_text().strip()
app = FastAPI(docs_url=None, redoc_url=None)
rclpy.init()
node = Node("explorer_web")
request_pub = node.create_publisher(String, "/explorer/request", 1)


def read_state(name, max_age=3.0):
    try:
        value = json.loads((ROOT / "data" / name).read_text())
        value["stale"] = time.time() - float(value["at"]) > max_age
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return {"stale": True, "available": False}


def emit(operation, **values):
    request = {"op": operation, "id": str(uuid.uuid4()), "at": time.monotonic(), **values}
    request_pub.publish(String(data=json.dumps(request, allow_nan=False)))
    return {"queued": True, "id": request["id"]}


class LidarCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.scans = {}

    def receive(self, name, message):
        points = []
        for index, distance in enumerate(message.ranges):
            if index % 2 or not math.isfinite(distance) or distance < message.range_min or distance > message.range_max:
                continue
            angle = message.angle_min + index * message.angle_increment
            points.append([round(distance * math.cos(angle), 3), round(distance * math.sin(angle), 3)])
        with self.lock:
            self.scans[name] = {
                "at": time.time(), "frame": message.header.frame_id, "points": points,
                "valid": len(points),
                "nearest_m": min((math.hypot(x, y) for x, y in points), default=None),
            }

    def status(self):
        with self.lock:
            value = json.loads(json.dumps(self.scans))
        now = time.time()
        for scan in value.values():
            scan["stale"] = now - scan["at"] > 1.0
        return {"at": now, "scans": value}


lidar_cache = LidarCache()
node.create_subscription(LaserScan, "/scan0", lambda message: lidar_cache.receive("scan0", message), qos_profile_sensor_data)
node.create_subscription(LaserScan, "/scan1", lambda message: lidar_cache.receive("scan1", message), qos_profile_sensor_data)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def reference_arm():
    from arm_model import ArmModel
    if not hasattr(reference_arm, "instance"):
        reference_arm.instance = ArmModel()
    return reference_arm.instance


from teaching import TeachingController
teaching = TeachingController(ROOT)
if (ROOT / "config/controller-profile.json").exists():
    from native_arm import NativeManualArm as ManualArm
else:
    from manual_arm import ManualArm
manual_arm = ManualArm(ROOT, node, reference_arm)
teaching.move = manual_arm.move
teaching.stop_revision = lambda: manual_arm.stop_revision
if getattr(manual_arm, "native", False):
    if manual_arm.profile.get("manual_reference_version") == 1:
        teaching.arm_reference = manual_arm.reference
    else:
        teaching.measured_reference = manual_arm.reference


def prepare_arm_geometry():
    """Load the unchanged arm model once so keyboard/gamepad arm control is ready."""
    try:
        manual_arm.prepare_geometry()
    except Exception as exc:
        manual_arm.error = str(exc)


threading.Thread(target=prepare_arm_geometry, daemon=True).start()


def stop_all():
    try:
        manual_arm.stop()
    except Exception:
        pass
    last = None
    # DDS publication is asynchronous. Confirm the core latched STOP before
    # reporting success, and republish if a depth-1 queue was briefly busy.
    for _ in range(3):
        last = emit("stop", initiator="manual_control")
        deadline = time.monotonic() + .4
        while time.monotonic() < deadline:
            try:
                state = json.loads((ROOT / "data/status.json").read_text())
                if (state.get("stop_latched") is True
                        and state.get("last_request", {}).get("id") == last["id"]):
                    return {"core_confirmed": True, "request_id": last["id"]}
            except (OSError, ValueError, TypeError):
                pass
            time.sleep(.01)
    raise ValueError("Controller did not confirm STOP")


def resume_manual():
    from manual_resume import resume_confirmed
    return resume_confirmed(emit, lambda: json.loads((ROOT / "data/status.json").read_text()))


from gamepad_panel import GamepadPanel
gamepad_panel = GamepadPanel(
    ROOT, teaching, stop_all,
    lambda velocity: emit("drive", velocity=velocity, source="manual"),
    lambda: emit("manual_release", initiator="manual_teleop"),
    resume=resume_manual, takeover=lambda: None,
)
gamepad_panel.bind_arm(manual_arm, reference_arm)


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path != "/":
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not secrets.compare_digest(supplied, TOKEN):
            return JSONResponse({"error": "Authentication required"}, 401)
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"error": "Cross-origin control blocked"}, 403)
    return await call_next(request)


@app.get("/")
def index():
    return HTMLResponse((ROOT / "src/index.html").read_text(), headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    state = read_state("status.json", 2)
    state["controller"] = read_state("controller-state.json", 2)
    state["power_telemetry"] = read_state("power.json", 3)
    state["camera"] = read_state("camera-frame.json", 2)
    state["teleop"] = gamepad_panel.teleop.status()
    state["gamepad"] = gamepad_panel.status()
    state["arm"] = manual_arm.status()
    try:
        memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
        state["host"] = {
            "ram_available_mb": round(memory["MemAvailable"] / 1024),
            "ram_total_mb": round(memory["MemTotal"] / 1024),
            "temperature_c": float(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000,
            "load_average": list(os.getloadavg()),
        }
    except (OSError, ValueError, KeyError):
        state["host"] = {}
    return state


@app.get("/api/frame")
def frame():
    path = ROOT / "data/frame-raw.jpg"
    try:
        fresh = 0 <= time.time() - path.stat().st_mtime < 2
    except OSError:
        fresh = False
    if not fresh:
        raise HTTPException(503, "Нет свежего кадра камеры")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/lidar")
def lidar():
    return lidar_cache.status()


SERVICE_NAMES = {
    "mcu": "explorer-mcu.service", "watchdog": "explorer-watchdog.service",
    "control": "explorer-control.service", "power": "explorer-power.service",
    "camera": "explorer-camera.service", "camera_cache": "explorer-camera-cache.service",
    "web": "explorer-web.service",
}


@app.get("/api/diagnostics")
def diagnostics():
    services = {}
    for name, unit in SERVICE_NAMES.items():
        result = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=2)
        services[name] = result.stdout.strip() or "unknown"
    disk = shutil.disk_usage(ROOT)
    return {"services": services, "disk_free_gb": round(disk.free / 1024**3, 1),
            "disk_total_gb": round(disk.total / 1024**3, 1),
            "uptime_s": float(Path("/proc/uptime").read_text().split()[0])}


class PanelLease(BaseModel):
    enabled: bool = False
    observing: bool = False


class GamepadMode(BaseModel):
    mode: str
    drive_profile: str | None = None
    arm_speed: str | None = None


class TeleopInput(BaseModel):
    source: str
    keys: list[str] = Field(default_factory=list, max_length=20)
    observing: bool = False
    arm_mode: str = "cartesian"


class TeleopAction(BaseModel):
    source: str
    observing: bool = False


class TeleopSelect(TeleopAction):
    neutral: bool = False


class TeleopMode(BaseModel):
    source: str
    mode: str


class TeleopSpeed(BaseModel):
    source: str
    speed: str


class ArmHome(BaseModel):
    observing: bool = False
    space_clear: bool = False


def mapped(function, *args):
    try:
        return function(*args)
    except (OSError, ValueError, TimeoutError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/teleop")
def teleop_status(): return gamepad_panel.teleop.status()


@app.get("/api/teleop/controls")
def teleop_controls():
    from manual_controls import scheme
    return scheme()


@app.post("/api/teleop/select")
def teleop_select(command: TeleopSelect): return mapped(gamepad_panel.teleop.select_source, command.source, command.observing, command.neutral)


@app.post("/api/teleop/mode")
def teleop_mode(command: TeleopMode): return mapped(gamepad_panel.teleop.set_arm_mode, command.source, command.mode)


@app.post("/api/teleop/speed")
def teleop_speed(command: TeleopSpeed): return mapped(gamepad_panel.teleop.set_arm_speed, command.source, command.speed)


@app.post("/api/teleop/input")
def teleop_input(command: TeleopInput):
    if command.source != "keyboard":
        raise HTTPException(400, "Browser endpoint accepts keyboard only")
    from manual_teleop import keyboard_inputs
    if gamepad_panel.teleop.arm_mode != command.arm_mode:
        raise HTTPException(409, "Сначала явно переключите режим руки")
    precision = bool({"ShiftLeft", "ShiftRight"} & set(command.keys))
    return mapped(gamepad_panel.teleop.update, "keyboard", keyboard_inputs(command.keys, command.arm_mode), command.observing, precision)


@app.post("/api/teleop/stop")
def teleop_stop():
    gamepad_panel.heartbeat(False)
    return gamepad_panel.teleop.stop()


@app.post("/api/teleop/resume")
def teleop_resume(command: TeleopAction): return mapped(gamepad_panel.teleop.resume, command.source, command.observing)


@app.post("/api/teleop/disconnect")
def teleop_disconnect(command: TeleopAction): return gamepad_panel.teleop.disconnect(command.source)


@app.post("/api/gamepad/panel")
def panel_lease(command: PanelLease): return gamepad_panel.heartbeat(command.enabled and command.observing)


@app.post("/api/gamepad/mode")
def gamepad_mode(command: GamepadMode): return mapped(gamepad_panel.select, command.mode, command.drive_profile, command.arm_speed)


@app.get("/api/arm/manual")
def arm_status(): return manual_arm.status()


@app.post("/api/arm/prepare")
def arm_prepare(command: ArmHome):
    if not command.observing or not command.space_clear:
        raise HTTPException(409, "Подтвердите присутствие и свободное пространство вокруг руки")
    return mapped(manual_arm.prepare_geometry)


@app.on_event("shutdown")
def shutdown():
    stop_all()
    node.destroy_node()
    if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
