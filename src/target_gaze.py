"""Bounded 2D gaze correction through the common arm velocity publisher."""
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
import cv2


def gaze_rates(model, mount, pose, center, image_size, intrinsic, distortion=None):
    q = np.asarray(pose, dtype=float)
    k = np.asarray(intrinsic, dtype=float).reshape(3, 3)
    if q.shape != (6,) or not np.isfinite(q).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Gaze geometry unavailable")
    mount = np.asarray(mount, dtype=float).reshape(4, 4)
    def rotation(angles):
        with model.lock:
            model.set_state(list(angles[:5]), 0.)
            return (np.asarray(model.state.get_global_link_transform("arm4")) @ mount)[:3, :3]
    current = rotation(q)
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float)
    if not np.isfinite(distortion).all():
        raise ValueError("Gaze camera distortion invalid")
    ray = cv2.undistortPoints(np.asarray(center, dtype=float).reshape(1, 1, 2), k, distortion)[0, 0]
    direction = current @ np.r_[ray, 1.]
    jacobian = np.zeros((2, 2))
    for column, index in enumerate((0, 3)):
        shifted = q.copy()
        shifted[index] += .15
        ray = rotation(shifted).T @ direction
        if ray[2] <= .05:
            raise ValueError("Selected target is behind the camera")
        projected = cv2.projectPoints(ray.reshape(1, 3), np.zeros(3), np.zeros(3), k, distortion)[0][0, 0]
        jacobian[:, column] = (projected-np.asarray(center))/.15
    error = np.asarray([image_size[0]/2, image_size[1]/2])-np.asarray(center)
    # Pixel deadband avoids quantized servo toggling; no digital image shifting.
    error[np.abs(error) < 12] = 0.
    desired = error*.8
    correction = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+np.eye(2)*.5, desired)
    if not np.isfinite(correction).all():
        raise ValueError("Gaze singularity")
    velocity = [0.]*6
    velocity[0], velocity[3] = np.clip(correction, -6., 6.).tolist()
    return velocity


class TargetGaze:
    def __init__(self, root, vision, arm, model):
        self.root = Path(root)
        self.vision, self.arm, self.model = vision, arm, model
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.owner = None
        self.generation = 0
        self.state = {"phase":"idle"}

    def cancel(self, identifier=None):
        if identifier is None or identifier == self.owner:
            self.cancelled.set()
            self.arm.stop_stream("gaze_tracking", self.generation)
        return dict(self.state, stopping=True)

    def run(self, args, context):
        if not self.lock.acquire(False):
            raise ValueError("Gaze is already active")
        try:
            accepted = json.loads((self.root / "config/handeye-accepted.json").read_text())
            if accepted.get("execution_authorized") is not True or accepted.get("reference_mount") != "arm4":
                raise ValueError("Accepted camera mount geometry required")
            from arm_commissioning import coordinated_status
            status = json.loads((self.root / "data/status.json").read_text())
            coordinated = status.get("mode") == "MANUAL" and status.get("stop_latched") is not True
            reference = self.arm.reference(coordinated_status) if coordinated else self.arm.reference()
            origin = np.asarray(reference["servo_deg"])
            self.owner = context.id
            self.generation += 1
            generation = self.generation
            self.cancelled.clear()
            end = time.monotonic()+args.get("duration_s", 10.)
            centred = 0
            self.state = {"phase":"tracking", "target_id":args["target_id"], "owner":context.id}
            def permit():
                context.permit()
                if self.cancelled.is_set():
                    raise InterruptedError("Gaze cancelled")
                return True
            while time.monotonic() < end:
                permit()
                observation = self.vision.tick()
                if observation.get("target_id") != args["target_id"] or observation.get("phase") != "tracking":
                    raise ValueError("Selected visual target lost; gaze stopped")
                if not 0 <= time.time()-observation["image_stamp"] < .3:
                    raise ValueError("Gaze frame expired")
                arm = json.loads((self.root / "data/arm-state.json").read_text())
                pose = arm.get("q_estimated_deg", arm["servo_deg"])
                if np.max(np.abs(np.asarray(pose)-origin)) > 15.:
                    raise ValueError("Bounded gaze range reached; choose a new viewpoint")
                if observation.get("intrinsic_k") is None or not observation.get("camera_info_version"):
                    raise ValueError("Current color-frame camera calibration unavailable")
                intrinsic = observation["intrinsic_k"]
                rates = gaze_rates(self.model(), accepted["camera_to_mount_reference"], pose,
                    observation["center_px"], observation["image_size"], intrinsic, observation.get("distortion_d"))
                self.arm.stream_velocity(rates, [0.]*3, generation, time.monotonic()+.35, permit,
                    owner="gaze_tracking", coordinated=coordinated)
                self.state.update(image_stamp=observation["image_stamp"], requested_deg_s=rates,
                    joint_state_source="command_estimate", physical_centering_measured_by_pixels=True)
                centred = centred+1 if not any(rates) else 0
                if centred >= 5:
                    break
                self.cancelled.wait(.08)
            # Neutral actively brakes with the same q/v/a. STOP/cancel adds no target.
            permit()
            self.arm.stream_velocity([0.]*6, [0.]*3, generation, time.monotonic()+.35, permit,
                owner="gaze_tracking", coordinated=coordinated)
            deadline = time.monotonic()+.4
            while self.arm.streaming_active and time.monotonic() < deadline:
                permit()
                time.sleep(.02)
            if self.arm.streaming_active:
                self.arm.stop_stream("gaze_tracking", generation)
            self.state.update(phase="centred" if centred >= 5 else "duration_complete", finished=time.time())
            return {"state":"succeeded", "evidence":dict(self.state, measured_joint_positions=False)}
        except Exception as exc:
            self.state.update(phase="stopped", reason=str(exc))
            self.arm.stop_stream("gaze_tracking", self.generation)
            raise
        finally:
            self.owner = None
            self.lock.release()
