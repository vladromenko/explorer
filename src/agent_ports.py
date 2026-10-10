"""Explicit skill adapters; every physical call reuses the installed executors."""
import copy
import json
from pathlib import Path
import threading
import time
from onboard_agent import SkillPort


def object_schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1, "maxLength": 80}
PLACES = {"type": "array", "items": TEXT, "minItems": 1, "maxItems": 12}


class ExplorerPorts:
    def __init__(self, root, status, missions, navigation, finder, world, views, arm,
                 trajectory, delivery, surveys, profiles=None, before_grounding=None):
        self.root = Path(root)
        self.robot_status = status
        self.missions = missions
        self.navigation = navigation
        self.finder = finder
        self.world = world
        self.views = views
        self.arm = arm
        self.trajectory = trajectory
        self.delivery = delivery
        self.surveys = surveys
        self.profiles = profiles
        self.before_grounding = before_grounding
        self.lock = threading.RLock()
        self.owned = {}

    @staticmethod
    def outcome(evidence, state="succeeded"):
        return {"state": state, "evidence": evidence, "at": time.time()}

    def own(self, context, kind, identifier):
        with self.lock:
            self.owned[context.id] = (kind, identifier)
        try:
            context.event("executor_accepted", {"executor": kind, "executor_id": identifier, "completed": False})
            context.permit()
        except Exception:
            self.cancel(context.id)
            self.release(context)
            raise

    def release(self, context):
        with self.lock:
            self.owned.pop(context.id, None)

    def cancel(self, identifier):
        with self.lock:
            owner = self.owned.get(identifier)
        if owner:
            kind, child = owner
            if kind == "navigation":
                active = self.navigation.status().get("active") or {}
                if active.get("id") == child:
                    self.navigation.cancel()
            if kind == "delivery":
                active = self.delivery.status().get("active") or {}
                if active.get("id") == child:
                    self.delivery.cancel()
            if kind == "grounding":
                self.finder.cancel(child)
            if kind == "arm":
                status = self.trajectory.status()
                if status.get("busy") and status.get("session") == child:
                    self.trajectory.stop()
            if kind == "view":
                self.trajectory.stop()

    def read_json(self, name, ttl=2):
        row = json.loads((self.root / "data" / name).read_text())
        stamp = row.get("image_stamp", row.get("at", 0))
        if type(stamp) not in (int, float) or not 0 <= time.time() - stamp <= ttl:
            raise ValueError("Fresh observation unavailable: " + name)
        return row

    def status(self, args, context):
        context.permit()
        return self.outcome(self.robot_status())

    def observe(self, args, context):
        context.permit()
        frame = self.read_json("perception.json")
        return self.outcome({"image_stamp": frame.get("image_stamp"), "objects": frame.get("objects", [])[:40],
                             "frame": "camera", "identity_verified": False, "image_endpoint": "/api/frame?raw=true"})

    def memory(self, args, context):
        context.permit()
        from object_finder import english_label
        try:
            label = english_label(args["label"])
        except ValueError:
            label = args["label"]
        return self.outcome(self.world.query(label))

    def places(self, args, context):
        context.permit()
        return self.outcome({"places": self.missions.places()})

    def save_place(self, args, context):
        context.permit()
        return self.outcome(self.missions.save_place(args["name"]))

    def report(self, args, context):
        context.permit()
        recent = self.surveys.recent()[:30]
        exports=[]
        if recent and recent[0].get("mission"):
            from scene_reports import SceneReports
            report=SceneReports(self.root).export(recent[0]["mission"])
            exports=[dict(id=report["id"],mission=report["report"]["mission"],
                artifacts={kind:"/api/scene-reports/"+report["id"]+"/report."+kind for kind in ("json","csv","html")})]
        context.permit()
        return self.outcome({"observations": recent, "exports":exports,"world": self.world.status(), "generated_at": time.time(),
                             "deduplicated_inventory_verified": False,
                             "claim": "Recorded observations; detector hypotheses are not a verified inventory"})

    def detect(self, args, context):
        context.permit()
        if self.before_grounding:
            self.before_grounding()
        operation = lambda: self.finder.start(args["label"])
        request = self.profiles.admit("grounding", operation) if self.profiles else operation()
        self.own(context, "grounding", request["id"])
        try:
            while self.finder.status().get("busy"):
                context.permit()
                time.sleep(.1)
            context.permit()
            result = self.finder.status()
            if result.get("id") != request["id"]:
                raise ValueError("Object observation identity changed")
            if result.get("phase") != "ready":
                raise ValueError("Object detection failed: " + str(result.get("error", result.get("phase"))))
            return self.outcome({"observation": result, "found": bool((result.get("result") or {}).get("objects")),
                                 "identity_verified": False, "image_endpoint": "/api/skill-agent/tasks/" + context.id + "/images/" + request["id"]})
        except Exception:
            self.cancel(context.id)
            raise
        finally:
            self.release(context)

    def navigation_skill(self, spec, context):
        context.permit()
        spec = dict(spec, observing=context.observing)
        request = self.navigation.start(spec)
        self.own(context, "navigation", request["id"])
        previous_phase = None
        try:
            while True:
                context.permit()
                state = self.navigation.status()
                active = state.get("active")
                if not active:
                    last = state.get("last") or {}
                    if last.get("id") != request["id"]:
                        raise ValueError("Navigation result identity changed")
                    if last.get("state") != "succeeded":
                        return self.outcome(last, "failed")
                    evidence = dict(last)
                    if "found" in last.get("result", {}):
                        evidence["found"] = last["result"]["found"]
                    return self.outcome(evidence)
                if active.get("id") != request["id"]:
                    raise ValueError("Navigation ownership changed")
                phase = active.get("phase")
                if phase != previous_phase:
                    context.event("executor_progress", {"phase": phase, "executor_id": request["id"]})
                    previous_phase = phase
                time.sleep(.15)
        except Exception:
            self.cancel(context.id)
            raise
        finally:
            self.release(context)

    def navigate(self, args, context):
        place = next((place for place in self.missions.places() if place["name"] == args["place"]), None)
        if not place or not place.get("compatible_map"):
            raise ValueError("Saved place unavailable in the current map")
        return self.navigation_skill({"kind": "navigate_current", "x": place["x"], "y": place["y"], "yaw": place["yaw"]}, context)

    def route(self, args, context):
        return self.navigation_skill({"kind": "patrol", "places": args["places"]}, context)

    def explore(self, args, context):
        spec = {"kind": "map_room" if args.get("map_name") else "survey_room", "max_goals": args.get("max_goals", 5)}
        if args.get("map_name"):
            spec["map_name"] = args["map_name"]
        return self.navigation_skill(spec, context)

    def find(self, args, context):
        return self.navigation_skill({"kind": "find_object", "object_query": args["label"], "places": args.get("places", [])}, context)

    def look(self, args, context):
        context.permit()
        self.own(context, "view", context.id)
        try:
            result=self.views.move(args["view"], context.permit,
                progress=lambda phase,details:context.event(phase,details))
            return self.outcome(result)
        except Exception:
            self.cancel(context.id)
            raise
        finally:
            self.release(context)

    def move(self, args, context):
        context.permit()
        plan = self.trajectory.plan(args["goal_deg"])
        context.permit()
        return self.execute_trajectory({"plan_id": plan["plan_id"]}, context)

    def execute_trajectory(self, args, context):
        context.permit()
        request = self.trajectory.start_local(args["plan_id"], context.permit)
        self.own(context, "arm", request["session"])
        try:
            while self.trajectory.status().get("busy"):
                context.permit()
                time.sleep(.05)
            context.permit()
            result = self.trajectory.status()
            if result.get("session") != request["session"]:
                raise ValueError("Arm result identity changed")
            if result.get("phase") not in ("reached", "command_completed"):
                return self.outcome(result, "failed")
            # Command-only elapsed motion is useful, but is not measured attainment.
            return self.outcome(result, "succeeded" if result.get("physically_verified") is True else "unknown")
        except Exception:
            self.cancel(context.id)
            raise
        finally:
            self.release(context)

    def gripper(self, args, context, opening):
        profile = json.loads((self.root / "config/local-grasp-profile.json").read_text())
        if profile.get("experimental_execution_authorized") is not True:
            raise ValueError("No accepted local gripper profile")
        reference = self.arm.reference()
        pose = reference.get("servo_deg") if isinstance(reference, dict) else reference
        if not isinstance(pose, list) or len(pose) != 6:
            # reference() has differing legacy return contracts; accepted state is explicit.
            state = json.loads((self.root / "data/arm-state.json").read_text())
            pose = state.get("servo_deg")
        if not isinstance(pose, list) or len(pose) != 6:
            raise ValueError("No valid six-joint command reference")
        goal = list(pose)
        goal[5] = profile["open_command_deg" if opening else "hold_command_deg"]
        result = self.move({"goal_deg": goal}, context)
        result["evidence"]["contact_verified"] = False
        result["evidence"]["force_measured"] = False
        return result

    def deliver(self, args, context):
        context.permit()
        goal = {"object_query": args["label"], "destination": {"name": args["destination"]}}
        request = self.delivery.start(goal)
        self.own(context, "delivery", request["id"])
        phase = None
        try:
            while True:
                context.permit()
                state = self.delivery.status()
                active = state.get("active")
                if not active:
                    last = state.get("last") or {}
                    if last.get("id") != request["id"]:
                        raise ValueError("Delivery result identity changed")
                    return self.outcome(last, "succeeded" if last.get("delivered") is True else "failed")
                if active.get("id") != request["id"]:
                    raise ValueError("Delivery ownership changed")
                if active.get("phase") != phase:
                    phase = active.get("phase")
                    context.event("executor_progress", {"phase": phase, "executor_id": request["id"]})
                time.sleep(.15)
        except Exception:
            self.cancel(context.id)
            raise
        finally:
            self.release(context)

    def catalog(self):
        def port(name, description, schema, operation, resources=(), physical=False, units="none", frame="none", requirements=()):
            return SkillPort(name, description, schema, resources, operation, self.cancel if physical or name == "detect_objects" else None,
                             units, frame, requirements, physical)
        return [
            port("get_robot_status", "Свежесть датчиков, батарея, режим и загрузка", object_schema(), self.status),
            port("observe_scene", "Свежие структурированные наблюдения текущей камеры", object_schema(), self.observe, ("camera",)),
            port("detect_objects", "Поиск названного предмета в текущем RGB-D кадре без поездки", object_schema({"label": TEXT}, ("label",)), self.detect, ("vision_inference",)),
            port("query_memory", "Где и когда предмет видели; прошлое наблюдение не доказывает наличие сейчас", object_schema({"label": TEXT}, ("label",)), self.memory),
            port("list_places", "Сохранённые места и совместимость с текущей картой", object_schema(), self.places),
            port("save_place", "Сохранить текущую остановленную позу как именованное место", object_schema({"name": TEXT}, ("name",)), self.save_place, ("map",)),
            port("inventory_report", "Отчёт по реальным сохранённым осмотрам без выдуманного подсчёта", object_schema(), self.report),
            port("look_direction", "Плавный обзор камерой на руке при остановленной базе", object_schema({"view": {"type": "string", "enum": ["forward", "left", "right"]}}, ("view",)), self.look, ("arm", "camera"), True, "servo degrees", "base", ("arm_reference", "accepted_handeye", "stationary_base")),
            port("move_arm", "Конечный MoveIt-путь; расчётная поза не выдаётся за измеренную", object_schema({"goal_deg": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 270}, "minItems": 6, "maxItems": 6}}, ("goal_deg",)), self.move, ("arm",), True, "servo degrees", "servo", ("arm_reference", "MoveIt_path", "stationary_base")),
            port("execute_trajectory", "Исполнить ранее рассчитанный свежий MoveIt plan_id", object_schema({"plan_id": {"type": "string", "pattern": r"[0-9a-f]{32}", "maxLength": 32}}, ("plan_id",)), self.execute_trajectory, ("arm",), True, "radians and seconds", "joint", ("fresh_validated_plan",)),
            port("open_gripper", "Плавно раскрыть пальцы по принятому профилю без сброса других суставов", object_schema(), lambda args, context: self.gripper(args, context, True), ("arm",), True, "servo degrees", "J6", ("accepted_gripper_profile",)),
            port("close_gripper", "Сомкнуть по профилю; это не подтверждение контакта или захвата", object_schema(), lambda args, context: self.gripper(args, context, False), ("arm",), True, "servo degrees", "J6", ("accepted_gripper_profile",)),
            port("navigate_to", "Доехать к сохранённому месту текущей карты", object_schema({"place": TEXT}, ("place",)), self.navigate, ("base", "arm", "camera", "navigation"), True, "metres and radians", "map", ("live_navigation_readiness", "current_map_place")),
            port("follow_route", "Осмотреть явную последовательность сохранённых мест", object_schema({"places": PLACES}, ("places",)), self.route, ("base", "arm", "camera", "navigation"), True, "metres and radians", "map", ("live_navigation_readiness",)),
            port("explore_area", "Достижимые frontier, осмотры, возврат; необязательно сохранить новую карту", object_schema({"max_goals": {"type": "integer", "minimum": 1, "maximum": 20}, "map_name": {"type": "string", "pattern": r"[a-zA-Z0-9_-]{1,48}", "maxLength": 48}}), self.explore, ("base", "arm", "camera", "navigation"), True, "metres and radians", "map", ("mapping_readiness",)),
            port("find_object", "Проверить виды камеры и выбранные места; остановиться при находке, не хватать", object_schema({"label": TEXT, "places": {**PLACES, "minItems": 0}}, ("label",)), self.find, ("base", "arm", "camera", "vision_inference", "navigation"), True, "metres", "map", ("mapping_readiness", "fresh_RGBD")),
            port("deliver_object", "Принятая доставка носка с проверкой удержания и размещения; остальные предметы отклоняются исполнителем", object_schema({"label": TEXT, "destination": TEXT}, ("label", "destination")), self.deliver, ("base", "arm", "camera", "navigation", "vision_inference"), True, "metres and servo degrees", "map/base", ("accepted_delivery", "accepted_destination")),
        ]
