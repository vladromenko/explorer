"""Validate a settled factory command estimate without claiming encoder feedback."""
import json
import math
from pathlib import Path
import time


class CommandPosePending(ValueError):
    pass


def settled_reference(root, boot, image_stamp=None):
    root=Path(root)
    state=json.loads((root/"data/arm-state.json").read_text())
    if state.get("boot_id") != boot:
        raise ValueError("Factory arm reference belongs to a different boot")
    fault=root/"data/arm-telemetry-fault.json"
    if fault.exists() and json.loads(fault.read_text()).get("at",0) > state.get("at",0):
        raise ValueError("Factory arm reference invalidated by a link fault")
    values=state.get("servo_deg")
    if not isinstance(values,list) or len(values)!=6 or any(type(q) not in (int,float) or not math.isfinite(q) for q in values):
        raise ValueError("Factory arm command coordinates unavailable")
    if state.get("phase") == "command_in_progress":
        raise CommandPosePending("Factory arm is moving; metric camera pose is pending")
    if state.get("phase") != "command_elapsed_observation_required":
        raise ValueError("Factory arm reference is unknown")
    stamp=state.get("at");runtime=state.get("runtime_ms")
    if (type(stamp) not in (int,float) or not math.isfinite(stamp)
            or type(runtime) not in (int,float) or not math.isfinite(runtime) or not 0<=runtime<=60000):
        raise ValueError("Factory arm command timing unavailable")
    settled=stamp+runtime/1000+.12
    observation=time.time() if image_stamp is None else image_stamp
    if type(observation) not in (int,float) or not math.isfinite(observation):
        raise ValueError("Camera exposure time unavailable")
    if observation<settled:
        raise CommandPosePending("Camera exposure predates settled factory command estimate")
    return dict(state,measured=False,attained=False,pose_source="command_estimate",settled_at=settled)
