"""Bounded visual following through ExplorerPorts.navigation_skill, no actuator I/O."""
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np

from agent_ports import object_schema
from onboard_agent import SkillPort
from visual_navigation import RGBDFrame, VisualNavigation


class VisualInterruption(ValueError):
    def __init__(self, reason, state="blocked", evidence=None):
        super().__init__(reason)
        self.state = state
        self.evidence = evidence or {}


class GuardedContext:
    """Navigation owner polls this permit; each poll revalidates its live scene."""
    def __init__(self, parent, guard):
        self.parent = parent
        self.guard = guard
        self.id = parent.id
        self.observing = parent.observing
        self.deadline = parent.deadline

    def permit(self):
        self.parent.permit()
        self.guard()

    def event(self, phase, details):
        self.permit()
        self.parent.event(phase, details)


class VisualNavigationAdapter:
    def __init__(self, root, vision, maps, missions, views, ports, algorithm=None,
                 clock=time.time, monotonic=time.monotonic, sleeper=time.sleep):
        self.root = Path(root)
        self.vision = vision
        self.maps = maps
        self.missions = missions
        self.views = views
        self.ports = ports
        self.algorithm = algorithm or VisualNavigation()
        self.clock = clock
        self.monotonic = monotonic
        self.sleeper = sleeper

    def _json(self, path):
        return json.loads((self.root / path).read_text())

    def snapshot(self):
        """Read an atomic cached RGB-D snapshot and the actual installed acceptance APIs."""
        now = self.clock()
        pose = self.maps.pose()
        if pose.get("provisional") is True or pose.get("localization_verified") is not True:
            raise VisualInterruption("Поза карты provisional: повторная локализация ещё не принята")
        if pose.get("frame", pose.get("frame_id")) != "map":
            raise VisualInterruption("Положение робота не выражено в текущей карте map")
        age = pose.get("age_s")
        if type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age < .5:
            raise VisualInterruption("Нет свежего времени TF позы робота в карте")
        accepted_bytes = (self.root / "config/handeye-accepted.json").read_bytes()
        accepted = json.loads(accepted_bytes)
        if accepted.get("execution_authorized") is not True or accepted.get("reference_mount") != "arm4" or accepted.get("camera_frame") != "camera_color_optical_frame":
            raise VisualInterruption("Не принята привязка камеры к arm4")
        validation_hash = accepted.get("physical_validation_sha256")
        validation_path = accepted.get("physical_validation_record")
        if not isinstance(validation_path, str) or not validation_path.startswith("data/handeye-validations/"):
            raise VisualInterruption("Нет записи физической проверки hand-eye")
        record_path = (self.root / validation_path).resolve()
        if not record_path.is_relative_to((self.root / "data/handeye-validations").resolve()):
            raise VisualInterruption("Недопустимый путь принятой проверки hand-eye")
        if hashlib.sha256(record_path.read_bytes()).hexdigest() != validation_hash:
            raise VisualInterruption("Запись принятой проверки hand-eye изменилась")
        view = self.views.status()
        arm = self._json("data/arm-state.json")
        if view.get("phase") != "ready" or view.get("view") != "forward":
            raise VisualInterruption("Сначала остановите базу и установите принятый обзор камеры вперёд")
        if arm.get("phase") != "command_elapsed_observation_required" or arm.get("servo_deg") != view.get("servo_deg"):
            raise VisualInterruption("Рука движется или исходная обзорная поза изменилась")
        if arm.get("boot_id") != self.views.arm.boot:
            raise VisualInterruption("Обзорная поза руки относится к предыдущему включению")
        fault_path = self.root / "data/arm-telemetry-fault.json"
        if fault_path.exists() and self._json("data/arm-telemetry-fault.json").get("at", 0) > arm.get("at", 0):
            raise VisualInterruption("Связь руки инвалидировала привязку обзорного кадра")
        geometry = self.views.geometry(arm["servo_deg"])
        if geometry.get("handeye_source") != validation_hash:
            raise VisualInterruption("Геометрия камеры не относится к принятой проверке hand-eye")
        content = (self.root / "data/rgbd-snapshot.npz").read_bytes()
        with np.load(io.BytesIO(content), allow_pickle=False) as raw:
            if "depth_stamp" not in raw:
                raise VisualInterruption("В RGB-D snapshot нет фактического depth_stamp; синхронизация не доказана")
            stamp = float(raw["stamp"])
            depth_stamp = float(raw["depth_stamp"])
            camera_frame = str(raw["frame"].item())
            identifier = str(raw["frame_id"].item()) if "frame_id" in raw else "rgbd:" + hashlib.sha256(content).hexdigest()
            arrays = {name: raw[name].copy() for name in ("rgb", "depth", "k", "d")}
        settled = view.get("settled_at")
        arm_stamp = arm.get("updated_at", arm.get("at"))
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (settled, arm_stamp)) or not settled < stamp or arm_stamp > stamp:
            raise VisualInterruption("Кадр снят до завершения или во время команды руки")
        return RGBDFrame(**arrays, image_stamp=stamp, depth_stamp=depth_stamp, frame_id=identifier,
            camera_frame=camera_frame, camera_to_base=geometry["camera_to_base_estimate"], transform_stamp=stamp,
            transform_version=hashlib.sha256(accepted_bytes).hexdigest(), transform_accepted=True,
            pose={"frame_id": "map", **{key: pose[key] for key in ("x", "y", "yaw")}},
            pose_stamp=now - age, pose_verified=True, map_epoch=self.maps.epoch())

    def _observe(self, kind, args):
        frame = self.snapshot()
        target = None
        if kind == "target":
            target = self.vision.tick()
            if target.get("target_id") != args["target_id"]:
                raise VisualInterruption("Идентификатор выбранного предмета изменился", "lost")
            result = self.algorithm.target_step(frame, target, stand_off_m=args.get("stand_off_m", .65), now=self.clock())
        else:
            result = self.algorithm.line_step(frame, args["color"], endpoint_map=args.get("endpoint_map"), now=self.clock())
        result.setdefault("evidence", {}).update(arm_pose_source="command_estimate",
            camera_transform_scope="accepted_stationary_frames_after_completed_commands",
            measured_joint_feedback=False)
        if result.get("state") in ("lost", "blocked"):
            raise VisualInterruption(result.get("reason", "Зрительное наблюдение недоступно"), result["state"], result.get("evidence"))
        return frame, result

    @staticmethod
    def _world_point(frame, point):
        c, s = math.cos(frame.pose["yaw"]), math.sin(frame.pose["yaw"])
        return np.array([frame.pose["x"] + c * point[0] - s * point[1],
                         frame.pose["y"] + s * point[0] + c * point[1]])

    def _guard(self, kind, args, frozen, proposal, until):
        if self.monotonic() >= until:
            raise VisualInterruption("Ограниченный сеанс зрительного сопровождения завершён", "budget_exhausted")
        current, observed = self._observe(kind, args)
        if current.map_epoch != frozen.map_epoch or current.transform_version != frozen.transform_version:
            raise VisualInterruption("Карта или принятая привязка камеры изменились")
        if observed["state"] == "reached":
            raise VisualInterruption("Измеренная дистанция до цели достигнута", "reached", observed["evidence"])
        if kind == "target":
            before = self._world_point(frozen, proposal["evidence"]["target_point_base_m"])
            latest = self._world_point(current, observed["evidence"]["target_point_base_m"])
            if np.linalg.norm(before - latest) > .1:
                raise VisualInterruption("Предмет переместился: прежняя навигационная цель отменена", "lost", observed["evidence"])
        else:
            goal = proposal["poses"][-1]
            if proposal["motion"] == "navigate" and math.hypot(goal["x"] - current.pose["x"], goal["y"] - current.pose["y"]) <= .12:
                raise VisualInterruption("Короткий шаг достигнут по свежей позе карты", "step_reached", observed["evidence"])
            points = np.asarray([self._world_point(current, point) for point in observed["evidence"]["measured_line_points_base_m"]])
            frozen_points = np.asarray([self._world_point(frozen, point) for point in proposal["evidence"]["measured_line_points_base_m"]])
            distances = np.min(np.linalg.norm(points[:, None] - frozen_points[None, :], axis=2), axis=1)
            # An in-place rotation has no translational line goal; still require the fresh floor line.
            if proposal["motion"] != "rotate" and np.count_nonzero(distances <= .12) < 3:
                raise VisualInterruption("Прежняя цель больше не подтверждается видимой линией пола", "lost", observed["evidence"])

    def _run(self, kind, args, context):
        duration = args.get("duration_s", 10)
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 1 <= duration <= 30:
            raise ValueError("Duration must be 1..30 seconds")
        until = min(context.deadline, self.monotonic() + duration)
        completed = []
        last_frame = None
        try:
            while self.monotonic() < until:
                context.permit()
                frame, proposal = self._observe(kind, args)
                if proposal["state"] == "reached":
                    return {"state": "succeeded", "evidence": dict(proposal["evidence"], completed=True, completion="measured_target_distance" if kind == "target" else "declared_map_endpoint", navigation_steps=completed)}
                if frame.frame_id == last_frame:
                    self.sleeper(.04)
                else:
                    if self.clock() >= proposal["evidence"]["valid_until"]:
                        raise VisualInterruption("Предложение устарело до отправки в Nav2", "lost")
                    goal = proposal["poses"][-1]
                    context.event("visual_navigation_proposal", proposal)
                    self.maps.preview(goal["x"], goal["y"])
                    if self.clock() >= proposal["evidence"]["valid_until"]:
                        raise VisualInterruption("Предложение устарело во время проверки Nav2; нужен новый кадр", "lost")
                    guarded = GuardedContext(context, lambda: self._guard(kind, args, frame, proposal, until))
                    guarded.permit()
                    spec = {"kind": "navigate_current", "x": goal["x"], "y": goal["y"], "yaw": goal["yaw"],
                            "holonomic": proposal["motion"] != "rotate", "position_only": False, "max_goals": 1}
                    try:
                        outcome = self.ports.navigation_skill(spec, guarded)
                    except VisualInterruption as exc:
                        if exc.state != "step_reached":
                            raise
                        self.ports.cancel(context.id)
                        outcome = {"state": "succeeded", "evidence": {"completion": "fresh_map_pose_step", "observation": exc.evidence, "nav2_result_verified": False}}
                    if outcome.get("state") != "succeeded":
                        return {"state": "failed", "evidence": {"reason": "Навигационная проверка шага не пройдена", "navigation": outcome, "navigation_steps": completed}}
                    completed.append({"frame_id": frame.frame_id, "goal": goal, "navigation": outcome})
                    last_frame = frame.frame_id
            return {"state": "waiting", "evidence": {"completed": False, "visual_state": "budget_exhausted", "reason": "Сеанс достиг заданного бюджета", "navigation_steps": completed}}
        except VisualInterruption as exc:
            self.ports.cancel(context.id)
            state = "succeeded" if exc.state == "reached" else "waiting" if exc.state == "budget_exhausted" else "failed"
            return {"state": state, "evidence": {"reason": str(exc), "visual_state": exc.state, "completed": exc.state == "reached", "observation": exc.evidence, "navigation_steps": completed}}
        except (InterruptedError, TimeoutError):
            self.ports.cancel(context.id)
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.ports.cancel(context.id)
            return {"state": "failed", "evidence": {"completed": False, "reason": str(exc), "navigation_steps": completed}}
        except Exception:
            self.ports.cancel(context.id)
            raise

    def follow_line(self, args, context):
        return self._run("line", args, context)

    def follow_target(self, args, context):
        return self._run("target", args, context)

    def skill_ports(self):
        duration = {"type": "number", "minimum": 1, "maximum": 30}
        resources = ("base", "arm", "camera", "navigation")
        requirements = ("fresh_aligned_RGBD", "accepted_stationary_handeye", "verified_map_pose", "Nav2_clearance")
        endpoint = object_schema({"frame_id": {"type": "string", "enum": ["map"]}, "map_epoch": {"type": "string", "minLength": 1, "maxLength": 80}, "x": {"type": "number"}, "y": {"type": "number"}}, ("frame_id", "map_epoch", "x", "y"))
        return [SkillPort("follow_line", "Следовать измеренной цветной линии пола короткими проверенными шагами", object_schema({"color": {"type": "string", "enum": ["red", "yellow", "green", "blue", "white", "black"]}, "duration_s": duration, "endpoint_map": endpoint}, ("color",)), resources, self.follow_line, self.ports.cancel, "metres/radians/seconds", "map", requirements, True),
                SkillPort("follow_selected_target", "Приблизиться к выделенному предмету по глубине, сохраняя дистанцию", object_schema({"target_id": {"type": "string", "minLength": 1, "maxLength": 80}, "stand_off_m": {"type": "number", "minimum": .4, "maximum": 1.2}, "duration_s": duration}, ("target_id",)), resources, self.follow_target, self.ports.cancel, "metres/radians/seconds", "map", requirements, True)]
