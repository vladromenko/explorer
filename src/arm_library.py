"""Saved poses and collision scenes on the existing model; preview-only APIs.

This module has no publisher, serial owner, trajectory executor or startup hook.
Calling any preview or saving a pose never moves the robot or changes J6.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
import numpy as np
from arm_commissioning import HARD_LIMITS


def _name(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 80:
        raise ValueError("Название должно содержать от 1 до 80 символов")
    return value.strip()


def _vector(value, count, field):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError("Invalid " + field)
    if any(type(v) not in (int, float) for v in value):
        raise ValueError("Numeric " + field + " required")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite " + field)
    return result


def _pose(value):
    result = _vector(value, 6, "servo_deg")
    if any(not lo <= number <= hi for number, (lo, hi) in zip(result, HARD_LIMITS)):
        raise ValueError("Поза выходит за существующие пределы суставов")
    return result


def validate_scene(obstacles):
    if not isinstance(obstacles, list) or len(obstacles) > 32:
        raise ValueError("Сцена допускает до 32 явно заданных препятствий")
    result = []
    names = set()
    for index, item in enumerate(obstacles):
        if not isinstance(item, dict):
            raise ValueError("Invalid scene obstacle")
        identifier = _name(item.get("id", "box_" + str(index)))
        if identifier in names:
            raise ValueError("Повторяющийся ID препятствия")
        center = _vector(item.get("center"), 3, "box center")
        size = _vector(item.get("size"), 3, "box size")
        if any(not 0 < v <= 10 for v in size) or any(abs(v) > 10 for v in center):
            raise ValueError("Недопустимый размер или положение препятствия")
        names.add(identifier)
        result.append(dict(id=identifier, center=center, size=size))
    return result


class ArmLibrary:
    def __init__(self, root, model, planner):
        self.root = Path(root)
        self.model = model
        self.planner = planner
        self.path = self.root / "data/arm-library.json"
        self.lock = threading.RLock()

    def _state(self):
        if not self.path.exists():
            return dict(schema_version=1, poses={}, scenes={})
        state = json.loads(self.path.read_text())
        if (state.get("schema_version") != 1 or not isinstance(state.get("poses"), dict)
                or not isinstance(state.get("scenes"), dict)):
            raise ValueError("Unsupported arm library schema; original file retained")
        return state

    def _save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False))
        temporary.replace(self.path)

    def _assets(self):
        path = self.root / "config/arm_model.json"
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def _builtins(self):
        result = {}
        source = self.root / "src/camera_views.py"
        if source.exists():
            # Read the literal from the existing implementation without
            # importing its hardware-adjacent dependencies or duplicating it.
            for statement in ast.parse(source.read_text()).body:
                if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name)
                        and target.id == "VIEWS" for target in statement.targets):
                    views = ast.literal_eval(statement.value)
                    for name, values in views.items():
                        pose = _vector(values, 5, "camera view")
                        if any(not lo <= q <= hi for q, (lo, hi) in zip(pose, HARD_LIMITS[:5])):
                            raise ValueError("Camera view exceeds existing joint limits")
                        result["camera_" + name] = dict(name="camera_" + name, servo_deg=pose,
                            gripper_policy="preserve_current", source="src/camera_views.py:VIEWS",
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            provenance="existing_camera_view", measured=False,
                            current_hardware_acceptance=False, preview_required=True)
        startup = self.root / "config/factory-arm-startup.json"
        if startup.exists():
            config = json.loads(startup.read_text())
            values = _pose(config["pose_deg"])
            result["factory_reference"] = dict(name="factory_reference", servo_deg=values,
                gripper_policy="explicit_configured_pose", source="config/factory-arm-startup.json",
                provenance="existing_startup_configuration", measured=False,
                current_hardware_acceptance=False, preview_required=True)
        acceptance = self.root / "config/arm_commissioning_report.json"
        if acceptance.exists():
            report = json.loads(acceptance.read_text())
            if report.get("tested_home_command") is not None:
                values = _pose(report["tested_home_command"])
                result["historical_observed_home"] = dict(name="historical_observed_home", servo_deg=values,
                    gripper_policy="explicit_historical_pose", source="config/arm_commissioning_report.json",
                    provenance="historical_visual_observation", evidence_date=report.get("date"),
                    historical_visual_observation=report.get("home_motion_visually_observed") is True,
                    measured=False, current_hardware_acceptance=False, preview_required=True)
        return result

    def catalog(self):
        with self.lock:
            state = self._state()
            poses = dict(self._builtins(), **state["poses"])
            return dict(schema_version=1, poses=list(poses.values()), scenes=list(state["scenes"].values()),
                frame="base_footprint", position_units="metre", joint_units="servo_degree",
                positional_axes=5, gripper_axis=6, measured=False, executed=False,
                execution_allowed=False, model_manifest_sha256=self._assets())

    def save_pose(self, name, servo_deg, operator_observed=False):
        name = _name(name)
        pose = _pose(servo_deg)
        if name in self._builtins():
            raise ValueError("Имя уже используется исходной позой")
        for jaw in (0.0, -0.2, -0.4, -0.6, -0.8):
            if not self.model().path(pose[:5], pose[:5], jaw)["valid"]:
                raise ValueError("Сохраняемая поза пересекает модель или пол")
        record = dict(name=name, servo_deg=pose, gripper_policy="explicit_saved_command",
            created_at=time.time(), provenance="operator_saved_command_estimate",
            operator_visual_confirmation=operator_observed is True,
            measured=False, current_hardware_acceptance=False, preview_required=True,
            model_manifest_sha256=self._assets())
        with self.lock:
            state = self._state()
            state["poses"][name] = record
            self._save(state)
        return dict(record, executed=False)

    def save_scene(self, name, obstacles):
        name = _name(name)
        boxes = validate_scene(obstacles)
        record = dict(name=name, obstacles=boxes, frame="base_footprint", units="metre",
            source="explicit_operator_boxes", measured_environment=False, created_at=time.time(),
            note="Статическая модель для предпросмотра; не обновляется автоматически камерой")
        with self.lock:
            state = self._state()
            state["scenes"][name] = record
            self._save(state)
        return record

    def scene(self, name=None):
        if name is None:
            return []
        with self.lock:
            item = self._state()["scenes"].get(_name(name))
            if item is None:
                raise ValueError("Сохранённая сцена не найдена")
            if item.get("frame") != "base_footprint" or item.get("units") != "metre":
                raise ValueError("Unsupported saved scene frame or units")
            return validate_scene(item["obstacles"])

    def named_pose(self, name, current):
        current = _pose(current)
        name = _name(name)
        with self.lock:
            poses = dict(self._builtins(), **self._state()["poses"])
            item = poses.get(name)
        if item is None:
            raise ValueError("Сохранённая поза не найдена")
        values = list(item["servo_deg"])
        if item["gripper_policy"] == "preserve_current":
            values.append(current[5])
        return dict(item, resolved_servo_deg=_pose(values), measured=False, executed=False)

    @staticmethod
    def _jaw(value):
        number = float(value)
        if not math.isfinite(number) or not -1.54 <= number <= 0:
            raise ValueError("Недопустимая геометрия раскрытия захвата")
        return number

    def preview_joint(self, start, goal, scene=None, gripper_rad=-0.3):
        start, goal = _pose(start), _pose(goal)
        jaw = self._jaw(gripper_rad)
        boxes = self.scene(scene)
        result = self.planner.plan(start[:5], goal[:5], jaw, boxes)
        return dict(result, requested_start_deg=start, requested_goal_deg=goal,
            scene=scene, joint6_policy="separate_aperture_command_not_part_of_position_IK",
            gripper_geometry_rad=jaw, gripper_geometry_calibrated=False,
            measured=False, executed=False, execution_allowed=False,
            preview_only=True, model_manifest_sha256=self._assets())

    def preview_named(self, name, start, scene=None, gripper_rad=-0.3):
        pose = self.named_pose(name, start)
        result = self.preview_joint(start, pose["resolved_servo_deg"], scene, gripper_rad)
        return dict(result, named_pose=pose)

    def preview_cartesian(self, start, target_xyz, scene=None, gripper_rad=-0.3,
            step_m=0.01, max_points=24, time_budget_s=15.0):
        start = _pose(start)
        target = np.asarray(_vector(target_xyz, 3, "target XYZ"))
        jaw = self._jaw(gripper_rad)
        if (not math.isfinite(step_m) or not 0.002 <= step_m <= 0.03
                or type(max_points) is not int or not 2 <= max_points <= 48
                or not math.isfinite(time_budget_s) or not 0.1 <= time_budget_s <= 60):
            raise ValueError("Недопустимый бюджет Cartesian-предпросмотра")
        model = self.model()
        initial = model.fk(start[:5], jaw)
        if initial["collision"]:
            raise ValueError("Исходная поза пересекает модель")
        origin = np.asarray(initial["xyz"])
        distance = float(np.linalg.norm(target - origin))
        count = max(1, math.ceil(distance / step_m))
        if count + 1 > max_points:
            raise ValueError("Путь длиннее бюджета предпросмотра; разбейте его на части")
        deadline = time.monotonic() + time_budget_s
        waypoints = [start]
        segments = []
        cartesian_points = [origin.tolist()]
        reason = None
        for index in range(1, count + 1):
            if reason is None:
                if time.monotonic() >= deadline:
                    reason = "Cartesian preview exceeded its time budget"
                else:
                    xyz = origin + (target - origin) * index / count
                    solution = model.ik(xyz.tolist(), waypoints[-1][:5], jaw, max_step_deg=10)
                    if not solution["solved"] or solution["collision"]:
                        reason = "Недостижимая или конфликтующая Cartesian-точка " + str(index)
                    else:
                        goal = _pose(list(solution["servo_deg"]) + [start[5]])
                        segment = self.preview_joint(waypoints[-1], goal, scene, jaw)
                        if not segment.get("planned"):
                            reason = "MoveIt не нашёл свободный путь к Cartesian-точке " + str(index)
                        else:
                            # Do not substitute an arbitrary OMPL detour for
                            # a requested Cartesian line. Verify its samples.
                            previous_xyz = np.asarray(cartesian_points[-1])
                            delta = xyz - previous_xyz
                            denominator = float(delta @ delta)
                            corridor_ok = True
                            joints = segment["servo_waypoints"]
                            for left, right in zip(joints, joints[1:]):
                                left, right = np.asarray(left), np.asarray(right)
                                samples = max(1, math.ceil(float(np.max(np.abs(right-left))) / 0.5))
                                for u in np.linspace(0.0, 1.0, samples + 1):
                                    point = np.asarray(model.fk((left+(right-left)*u).tolist(), jaw)["xyz"])
                                    fraction = float((point - previous_xyz) @ delta) / denominator if denominator > 1e-12 else 0.0
                                    closest = previous_xyz + np.clip(fraction, 0.0, 1.0) * delta
                                    if np.linalg.norm(point - closest) > 0.003:
                                        corridor_ok = False
                            if time.monotonic() >= deadline:
                                reason = "Cartesian preview exceeded its time budget"
                            elif not corridor_ok:
                                reason = "Путь OMPL выходит из коридора Cartesian-линии"
                            else:
                                waypoints.append(goal)
                                cartesian_points.append(xyz.tolist())
                                segments.append(segment)
        return dict(planned=reason is None, reason=reason, positional_axes=5,
            orientation_priority="position_with_local_seed_and_preserved_roll",
            arbitrary_6D_orientation_supported=False, servo_waypoints=waypoints,
            cartesian_waypoints_m=cartesian_points, segments=segments,
            completed_fraction=(len(waypoints) - 1) / count, gripper_preserved_deg=start[5],
            frame="base_footprint", preview_only=True, measured=False,
            executed=False, execution_allowed=False, model_manifest_sha256=self._assets())

    def preview_random(self, start, scene=None, gripper_rad=-0.3, seed=0,
            radius_deg=20.0, count=3, attempts=24):
        start = _pose(start)
        if (type(seed) is not int or not math.isfinite(radius_deg) or not 0 < radius_deg <= 45
                or type(count) is not int or not 1 <= count <= 8
                or type(attempts) is not int or not count <= attempts <= 64):
            raise ValueError("Недопустимый бюджет случайного предпросмотра")
        rng = random.Random(seed)
        results = []
        tried = 0
        for _ in range(attempts):
            if len(results) < count:
                tried += 1
                goal = [max(lo, min(hi, q + rng.uniform(-radius_deg, radius_deg)))
                    for q, (lo, hi) in zip(start[:5], HARD_LIMITS[:5])] + [start[5]]
                fk = self.model().fk(goal[:5], self._jaw(gripper_rad))
                if not fk["collision"]:
                    preview = self.preview_joint(start, goal, scene, gripper_rad)
                    if preview.get("planned"):
                        results.append(dict(preview, target_xyz_m=fk["xyz"]))
        return dict(previews=results, requested_count=count, accepted_count=len(results),
            attempted=tried, seed=seed, generated_from_existing_joint_limits=True,
            gripper_preserved_deg=start[5], preview_only=True, measured=False,
            executed=False, execution_allowed=False)
