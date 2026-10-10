"""Onboard camera views through the existing MoveIt executor, with base held.

Joint pose and hand-eye are command estimates: observations after settling only.
"""
import json
import math
from pathlib import Path
import time
import numpy as np
from lerobot_bridge import write_json

VIEWS={"forward":[90,90,60,15,90],"left":[60,90,60,15,90],"right":[120,90,60,15,90]}


class CameraViews:
    def __init__(self,root,arm,trajectory,model):
        self.root=Path(root);self.arm=arm;self.trajectory=trajectory;self.model=model
        self.state={"phase":"idle","view":None,"measured_joints":False}

    def status(self):
        return dict(self.state,views=VIEWS,source="onboard_arm_camera",
            base_motion_during_pan=False,pose_source="command_estimate")

    def fresh_frame(self,after=0):
        metadata=self.root/"data/camera-frame.json"
        perception=json.loads((metadata if metadata.exists() else self.root/"data/perception.json").read_text())
        stamp=perception.get("image_stamp")
        if type(stamp) not in (int,float) or not math.isfinite(stamp) or not after<stamp<=time.time() or time.time()-stamp>2:
            raise ValueError("Нет свежего изображения бортовой камеры")
        return stamp

    def geometry(self,pose):
        accepted=json.loads((self.root/"config/handeye-accepted.json").read_text())
        if accepted.get("execution_authorized") is not True or accepted.get("reference_mount")!="arm4":
            raise ValueError("Нет принятой привязки бортовой камеры к руке")
        model=self.model()
        with model.lock:
            model.set_state(pose[:5],0.)
            transform=np.asarray(model.state.get_global_link_transform("arm4"))@np.asarray(accepted["camera_to_mount_reference"])
            if model.collision() or transform[2,3]<.22:raise ValueError("Обзорная поза пересекает корпус/пол")
        if transform[2,2]>=0 or transform[0,2]<.65:raise ValueError("Камера не направлена вперёд и немного вниз")
        return {"camera_to_base_estimate":transform.tolist(),"optical_axis_base_estimate":transform[:3,2].tolist(),
            "handeye_source":accepted.get("physical_validation_sha256"),"measured_joints":False}

    def move(self,view,permit,progress=None):
        if view not in VIEWS:raise ValueError("Unknown camera view")
        permit()
        reference=json.loads((self.root/"data/arm-state.json").read_text())
        if reference.get("boot_id")!=self.arm.boot or reference.get("phase")!="command_elapsed_observation_required":
            raise ValueError("Нет завершённой команды положения бортовой камеры")
        fault=self.root/"data/arm-telemetry-fault.json"
        if fault.exists() and json.loads(fault.read_text()).get("at",0)>reference.get("at",0):
            raise ValueError("Camera arm reference was invalidated by a link fault")
        start=reference.get("servo_deg")
        if not isinstance(start,list) or len(start)!=6 or any(type(value) not in (int,float) or not math.isfinite(value) for value in start):
            raise ValueError("Invalid camera arm command estimate")
        goal=list(VIEWS[view])+[start[5]]
        geometry=self.geometry(goal)
        axis=geometry.get("optical_axis_base_estimate")
        if view!="forward" and (axis is None or (axis[1] if view=="left" else -axis[1])<.2):
            raise ValueError("Camera view disagrees with base-frame left/right")
        self.state=dict(phase="positioning",view=view,measured_joints=False)
        started=time.monotonic()
        progress=progress or (lambda phase,details:None)
        try:
            if max(abs(a-b) for a,b in zip(start,goal))>.5:
                self.arm.reference()  # Actuation still checks stationary base and arm power budget.
                progress("planning_started",{"view":view,"goal_deg":goal})
                plan=self.trajectory.plan(goal);permit()
                planning_s=time.monotonic()-started
                progress("planning_completed",{"planning_duration_s":planning_s,"plan_id":plan["plan_id"]})
                request=self.trajectory.start_local(plan["plan_id"],permit)
                session=request["session"];deadline=time.monotonic()+120
                previous_timing=None
                while self.trajectory.status().get("busy"):
                    permit()
                    if time.monotonic()>deadline:raise ValueError("Camera positioning timed out")
                    timing=self.arm.status().get("active_path_timing")
                    if timing and previous_timing is None:
                        progress("execution_timing",timing)
                        previous_timing=timing
                    time.sleep(.05)
                result=self.trajectory.status()
                if result.get("session")!=session or result.get("phase") not in ("reached","command_completed"):
                    raise ValueError("Camera movement failed: "+str(result.get("reason")))
            settled=time.time();deadline=time.monotonic()+3
            stamp=None
            while stamp is None and time.monotonic()<deadline:
                permit()
                try:stamp=self.fresh_frame(settled)
                except (ValueError,OSError):time.sleep(.05)
            if stamp is None:raise ValueError("No camera frame after arm settled")
            self.state=dict(phase="ready",view=view,servo_deg=goal,settled_at=settled,image_stamp=stamp,**geometry)
            self.state["total_duration_s"]=time.monotonic()-started
            write_json(self.root/"data/camera-view.json",self.state)
            return self.status()
        except Exception as exc:
            self.state.update(phase="failed",reason=str(exc));self.trajectory.stop();raise

    def navigation_guard(self,state):
        self.fresh_frame()
        arm=json.loads((self.root/"data/arm-state.json").read_text())
        pose=arm.get("servo_deg",[])
        if arm.get("boot_id")!=self.arm.boot or arm.get("phase")!="command_elapsed_observation_required":
            raise ValueError("Camera arm reference is not settled")
        if len(pose)!=6 or any(type(value) not in (int,float) or not math.isfinite(value) for value in pose):
            raise ValueError("Invalid camera arm command estimate")
        fault=self.root/"data/arm-telemetry-fault.json"
        if fault.exists() and json.loads(fault.read_text()).get("at",0)>arm.get("at",0):
            raise ValueError("Camera arm reference was invalidated by a link fault")
        moving=any(abs(value)>.001 for value in state.get("velocity",[0,0,0]))
        if moving and max(abs(a-b) for a,b in zip(pose[:5],VIEWS["forward"]))>.5:
            raise ValueError("Для поездки бортовая камера должна смотреть вперёд")
