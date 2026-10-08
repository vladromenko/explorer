"""Bounded operator-selected RGB-D approach in the current provisional map."""
from collections import deque
import json
import math
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np


def selected_depth_point(depth, camera_matrix, distortion, u, v):
    image = np.asarray(depth, dtype=float)
    if image.ndim != 2 or not all(math.isfinite(float(value)) for value in (u, v)):
        raise ValueError("Неверная точка изображения")
    x, y = int(round(u)), int(round(v))
    if not 8 <= x < image.shape[1]-8 or not 8 <= y < image.shape[0]-8:
        raise ValueError("Выберите предмет дальше от края кадра")
    patch = image[y-7:y+8, x-7:x+8]
    centre = image[y-2:y+3, x-2:x+3]
    valid = patch[np.isfinite(patch) & (patch > .20) & (patch < 3.5)]
    central = centre[np.isfinite(centre) & (centre > .20) & (centre < 3.5)]
    if len(valid) < 80 or len(central) < 12:
        raise ValueError("На выбранном предмете нет достаточной глубины")
    distance = float(np.median(central))
    spread = float(np.median(np.abs(valid-distance)))
    if spread > .10 or float(np.percentile(valid,90)-np.percentile(valid,10)) > .18 or abs(float(np.median(valid))-distance) > .08:
        raise ValueError("Глубина под пальцем неоднозначна; выберите середину предмета")
    intrinsic = np.asarray(camera_matrix, dtype=float).reshape(3,3)
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if not np.isfinite(intrinsic).all() or intrinsic[0,0] <= 0 or intrinsic[1,1] <= 0 or not np.isfinite(coefficients).all() or len(coefficients) not in (4,5,8,12,14):
        raise ValueError("Калибровка камеры недоступна")
    ray = cv2.undistortPoints(np.array([[[float(x),float(y)]]]), intrinsic, coefficients)[0,0]
    return np.array([ray[0]*distance,ray[1]*distance,distance,1.]), distance, spread


def approach_goal(camera_point, camera_to_base, robot_pose, stand_off=.55, max_step=.8):
    transform = np.asarray(camera_to_base, dtype=float)
    if transform.shape != (4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]):
        raise ValueError("Положение камеры относительно корпуса не подтверждено")
    base = transform @ np.asarray(camera_point,dtype=float)
    distance = math.hypot(float(base[0]),float(base[1]))
    if not math.isfinite(distance) or float(base[0]) <= .20 or distance > 3.5:
        raise ValueError("Выбранная точка не находится впереди в рабочем диапазоне")
    if distance <= stand_off+.05:
        raise ValueError("Робот уже близко к выбранной точке")
    advance = min(distance-stand_off,max_step)
    dx,dy = advance*base[0]/distance,advance*base[1]/distance
    yaw = float(robot_pose["yaw"]);c,s=math.cos(yaw),math.sin(yaw)
    return dict(x=float(robot_pose["x"]+c*dx-s*dy),y=float(robot_pose["y"]+s*dx+c*dy),
        start_x=float(robot_pose["x"]),start_y=float(robot_pose["y"]),
        yaw=yaw,advance_m=float(advance),object_distance_m=float(distance),stand_off_m=stand_off)


def safe_approach_step(goal,preview):
    """Choose the longest path the current planner accepts; never bypass it."""
    start_x,start_y=goal["start_x"],goal["start_y"]
    last_error=None
    for fraction in (1.0,.75,.5,.25,.125):
        step=goal["advance_m"]*fraction
        if step>=.08:
            x=start_x+(goal["x"]-start_x)*fraction
            y=start_y+(goal["y"]-start_y)*fraction
            try:preview(x,y)
            except ValueError as exc:
                if not str(exc).startswith("No safe path:"):
                    raise
                last_error=exc
            else:
                return dict(goal,x=x,y=y,advance_m=step,partial_approach=fraction<1.0)
    raise ValueError("Нет безопасного пути даже для короткого подхода: "+str(last_error))


class TargetFrames:
    def __init__(self,root,pose,epoch):
        self.root=Path(root);self.pose=pose;self.epoch=epoch
        self.lock=threading.Lock();self.frames=deque(maxlen=8)

    def _state(self):
        state=json.loads((self.root/"data/status.json").read_text())
        if not 0<=time.time()-state.get("at",0)<.8 or state.get("stop_latched") is not True:
            raise ValueError("Для выбора цели сначала остановите шасси")
        velocity=state.get("odom_velocity",[])
        if len(velocity)!=3 or any(not math.isfinite(float(value)) or abs(float(value))>.01 for value in velocity):
            raise ValueError("Шасси ещё движется")
        view=json.loads((self.root/"data/camera-view.json").read_text())
        arm=json.loads((self.root/"data/arm-state.json").read_text())
        if view.get("phase")!="ready" or view.get("view")!="forward" or view.get("servo_deg")!=arm.get("servo_deg"):
            raise ValueError("Сначала установите камеру руки в обзор вперёд и остановите руку")
        return view,arm

    def capture(self):
        view,arm=self._state()
        with np.load(self.root/"data/rgbd-snapshot.npz",allow_pickle=False) as raw:
            stamp=float(raw["stamp"]);frame=str(raw["frame"].item())
            rgb=raw["rgb"].copy();depth=raw["depth"].copy();intrinsic=raw["k"].copy();distortion=raw["d"].copy()
        if not 0<=time.time()-stamp<3.0 or stamp<=float(view.get("settled_at",0)) or float(arm.get("updated_at",0))>stamp:
            raise ValueError("Кадр устарел или рука двигалась во время съёмки")
        if frame!="camera_color_optical_frame" or depth.shape!=rgb.shape[:2]:
            raise ValueError("Глубина не совмещена с цветной камерой")
        pose=self.pose();epoch=self.epoch()
        image_ok,jpeg=cv2.imencode(".jpg",rgb,[cv2.IMWRITE_JPEG_QUALITY,78])
        if not image_ok:raise ValueError("Не удалось подготовить кадр")
        item=dict(id=uuid.uuid4().hex,stored_at=time.time(),stamp=stamp,depth=depth,k=intrinsic,d=distortion,
            camera_to_base=view.get("camera_to_base_estimate"),pose=pose,epoch=epoch,
            width=rgb.shape[1],height=rgb.shape[0])
        with self.lock:self.frames.append(item)
        return item["id"],jpeg.tobytes()

    def goal(self,identifier,u,v):
        with self.lock:frame=next((item for item in self.frames if item["id"]==identifier),None)
        if frame is None or not 0<=time.time()-frame["stored_at"]<5:
            raise ValueError("Кадр сменился; выберите предмет ещё раз")
        view,arm=self._state()
        if view.get("camera_to_base_estimate")!=frame["camera_to_base"] or float(arm.get("updated_at",0))>frame["stamp"]:
            raise ValueError("Положение руки изменилось после кадра")
        if self.epoch()!=frame["epoch"]:raise ValueError("Текущая карта изменилась")
        pose=self.pose()
        drift=math.hypot(pose["x"]-frame["pose"]["x"],pose["y"]-frame["pose"]["y"])
        turn=abs(math.atan2(math.sin(pose["yaw"]-frame["pose"]["yaw"]),math.cos(pose["yaw"]-frame["pose"]["yaw"])))
        if drift>.05 or turn>.08:raise ValueError("Положение робота изменилось после кадра")
        point,depth,spread=selected_depth_point(frame["depth"],frame["k"],frame["d"],u,v)
        goal=approach_goal(point,frame["camera_to_base"],pose)
        return dict(goal,frame_id=identifier,image_stamp=frame["stamp"],depth_m=depth,
            depth_spread_m=spread,pose_source="command_estimate",map_position_verified=False)
