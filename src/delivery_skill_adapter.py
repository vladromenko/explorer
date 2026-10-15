"""Explicit opt-in bridge from DeliveryRobot to the shared task executor.

Importing or constructing this adapter never starts robot operations. Its caller
must supply an already commissioned DeliveryRobot and explicitly call execute.
The existing DeliveryTask and its routes remain independent and unchanged.
Blocking robot calls must cooperate through the owner-check callback supplied to
begin; Python cannot safely preempt an uncooperative hardware call.
"""
from collections.abc import Mapping
from dataclasses import replace
import math
import threading
import time
import uuid

if __package__:
    from .task_executor import Observation, TaskExecutor, TaskResult, TaskStopped
else:
    from task_executor import Observation, TaskExecutor, TaskResult, TaskStopped


class DeliverySkillAdapter:
    def __init__(self, robot, *, cancelled=None, clock=time.monotonic,
                 wall_clock=time.time, total_timeout=900., begin_timeout=120.,
                 stage_timeout=110., stage_timeouts=None):
        for name, value in (("total_timeout", total_timeout), ("begin_timeout", begin_timeout),
                            ("stage_timeout", stage_timeout)):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        self.robot = robot
        self.cancelled = cancelled or (lambda: False)
        self.clock, self.wall_clock = clock, wall_clock
        self.total_timeout, self.begin_timeout = total_timeout, begin_timeout
        self.stage_timeout, self.stage_timeouts = stage_timeout, stage_timeouts
        self._lock = threading.Lock()
        self._mid = None
        self._context = None
        self._inside_permit = False

    def _owner_check(self):
        if self.cancelled():
            raise TaskStopped("Delivery cancelled", state="cancelled")
        if self.clock() >= self._deadline:
            raise TaskStopped("Delivery deadline expired")
        if self._context is None:
            if self.clock() >= self._begin_deadline:
                raise TaskStopped("Delivery begin deadline expired")
        elif not self._inside_permit:
            self._context.check()

    def _guard(self, context):
        self._context = context
        self._inside_permit = True
        try:
            self._owner_check()
            self.robot.permit(self._mid)
            self._owner_check()
        finally:
            self._inside_permit = False

    def _observe(self, context):
        """Use raw fresh RGB-D even before find starts an object tracker.

        DeliveryRobot.observe requires established tracking, so using it for the
        executor's pre-find frame would prevent search from starting. Raw frames
        are freshness evidence only; verified hold/place use tracked intervals.
        """
        context.check()
        after = self.wall_clock()
        sample = self.robot.vision.wait_snapshot(self._owner_check, after_stamp=after)
        context.check()
        try:
            stamp = float(sample["stamp"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Delivery camera returned no numeric exposure timestamp") from None
        now = self.wall_clock()
        if (not math.isfinite(stamp) or
                not after < stamp <= now or now - stamp >= .4 or
                self._last_stamp is not None and stamp <= self._last_stamp):
            raise ValueError("Delivery camera did not acquire a fresh RGB-D exposure")
        self._last_stamp = stamp
        self._sequence += 1
        return Observation(self._sequence, self.clock(), context.base_revision,
                           {"source": "fresh_rgbd_snapshot", "stamp": stamp,
                            "frame": str(sample.get("frame", "camera")),
                            "semantic_identity_verified": False})

    @staticmethod
    def _mapping(value, name):
        if not isinstance(value, Mapping):
            raise ValueError("Missing completed robot result: " + name)
        if value.get("accepted") is True and value.get("completed") is not True:
            raise ValueError("Robot returned acceptance without completion: " + name)
        return dict(value)

    @classmethod
    def _motion(cls, value, name):
        value = cls._mapping(value, name)
        complete = value.get("command_completed") is True or value.get("attained") is True
        guarded = (value.get("guarded") is True and
                   isinstance(value.get("decision"), Mapping) and
                   value["decision"].get("action") == "hold" and
                   type(value.get("final_deg")) in (float, int) and math.isfinite(value["final_deg"]))
        if not complete and not guarded:
            raise ValueError("Robot motion completion not confirmed: " + name)
        return {"outcome": "success", "evidence": {"robot_result": value,
                "command_completion_only": True, "object_outcome_confirmed": False}}

    def _find(self, context):
        value = self._mapping(self.robot.find(), "find")
        if not isinstance(value.get("target"), Mapping) or not value["target"]:
            raise ValueError("Find returned no observed target")
        self._target = value
        # Search may navigate between its accepted places. Always invalidate the
        # previous camera/base reference, even when it stayed in the first place.
        return {"outcome": "success", "target": value, "base_moved": True}

    def _approach(self, context):
        value = self._mapping(self.robot.approach(self._target), "approach")
        if value.get("reobserve_required") is True:
            if value.get("base_repositioned") is not True:
                return {"outcome": "unknown", "reason": "Alignment requested without completed reposition"}
            self._need_reacquire = True
            return {"outcome": "retry", "base_moved": True, "evidence": {"robot_result": value}}
        if value.get("reachable") is not True or value.get("base_stationary_confirmed") is not True:
            return {"outcome": "unknown", "reason": "Reachable stationary approach not confirmed"}
        return {"outcome": "success", "evidence": {"robot_result": value}}

    def _reobserve(self, context):
        if self._need_reacquire:
            value = self._mapping(self.robot.find_aligned(), "find_aligned")
            if not isinstance(value.get("target"), Mapping) or not value["target"]:
                raise ValueError("Aligned search returned no fresh target")
            self._target = value
            self._need_reacquire = False
            # View recovery is allowed to retreat; treat it as a new base epoch.
            return {"outcome": "success", "target": value, "base_moved": True}
        held = self._mapping(self.robot.hold(), "hold_reobserve")
        if held.get("base_stationary_confirmed") is not True:
            return {"outcome": "unknown", "reason": "Reobservation base stationarity not confirmed"}
        self.robot.inspect()
        observed = self._mapping(self.robot.reobserve(self._target), "reobserve")
        if not observed:
            raise ValueError("Reobservation is empty")
        self._target = dict(self._target, target=observed)
        return {"outcome": "success", "target": self._target}

    def _pregrasp(self, context):
        value = self._mapping(self.robot.hold(), "hold_base")
        if value.get("base_stationary_confirmed") is not True:
            return {"outcome": "unknown", "reason": "Base stationarity not confirmed"}
        self.robot.inspect()
        aligned = self._mapping(self.robot.align(self._target), "align")
        if aligned.get("aligned") is not True:
            return {"outcome": "unknown", "reason": "Visual alignment not confirmed"}
        return self._motion(self.robot.pregrasp(self._target), "pregrasp")

    def _grasp(self, context):
        return self._motion(self.robot.grasp(self._target), "grasp")

    def _lift(self, context):
        # Baseline must follow closure and precede lift, while base is stationary.
        self._before_hold = self.robot.observe()
        context.check()
        return self._motion(self.robot.lift(), "lift")

    def _verification(self, value, name):
        value = self._mapping(value, name)
        outcome = value.get("outcome")
        evidence = value.get("evidence")
        if outcome not in {"success", "failure", "unknown"}:
            return {"outcome": "unknown", "reason": "Verifier returned no explicit measured outcome"}
        if outcome == "success" and (not isinstance(evidence, Mapping) or not evidence):
            return {"outcome": "unknown", "reason": "Verifier success has no measured evidence"}
        return {"outcome": outcome, "evidence": dict(evidence or {},
                confirmed=outcome == "success", source="DeliveryRobot." + name)}

    def _verify_hold(self, context):
        return self._verification(self.robot.verify_hold(self._before_hold), "verify_hold")

    def _carry(self, context):
        self._motion(self.robot.transport(), "transport")
        context.check()
        value = self._mapping(self.robot.carry(), "carry")
        if not all(type(value.get(key)) in (float, int) and math.isfinite(value[key]) for key in ("x", "y", "yaw")):
            return {"outcome": "unknown", "reason": "Navigation returned no completed measured pose"}
        return {"outcome": "success", "base_moved": True, "evidence": {"robot_result": value}}

    def _support(self, context):
        value = self._mapping(self.robot.hold(), "hold_destination")
        if value.get("base_stationary_confirmed") is not True:
            return {"outcome": "unknown", "reason": "Destination stationarity not confirmed"}
        return self._motion(self.robot.support(), "support")

    def _release(self, context):
        # Baseline follows support, before opening or withdrawal.
        self._before_release = self.robot.observe()
        context.check()
        return self._motion(self.robot.release(), "release")

    def _withdraw(self, context):
        return self._motion(self.robot.withdraw(), "withdraw")

    def _verify_place(self, context):
        return self._verification(self.robot.verify_place(self._before_release), "verify_place")

    def execute(self, goal=None):
        """Explicitly begin one owned hardware task; never invoked automatically."""
        if goal is not None and not isinstance(goal, Mapping):
            raise ValueError("Delivery goal must be a mapping")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Delivery skill adapter is already running")
        self._mid = uuid.uuid4().hex
        self._context = None
        self._inside_permit = False
        self._sequence = 0
        self._last_stamp = None
        self._target = None
        self._need_reacquire = False
        self._before_hold = self._before_release = None
        self._deadline = self.clock() + self.total_timeout
        self._begin_deadline = min(self._deadline, self.clock() + self.begin_timeout)
        result = None
        begin_attempted = False
        try:
            if not callable(getattr(getattr(self.robot, "vision", None), "wait_snapshot", None)):
                raise ValueError("Delivery requires a fresh raw RGB-D snapshot port before starting")
            self._owner_check()
            begin_attempted = True
            self.robot.begin(self._mid, check=self._owner_check,
                             **({"goal": dict(goal)} if goal is not None else {}))
            self._owner_check()
            remaining = self._deadline - self.clock()
            operations = {name: getattr(self, "_" + name) for name in (
                "find", "approach", "reobserve", "pregrasp", "grasp", "lift", "verify_hold",
                "carry", "support", "release", "withdraw", "verify_place")}
            executor = TaskExecutor(operations, self._observe, guard=self._guard,
                cancelled=self.cancelled, stop=lambda ctx: self.robot.stop(self._mid),
                clock=self.clock, total_timeout=remaining, stage_timeout=self.stage_timeout,
                stage_timeouts=self.stage_timeouts, max_find_attempts=1,
                max_alignment_attempts=4, max_grasp_attempts=1)
            result = executor.execute(dict(goal or {}))
        except Exception as error:
            state = error.state if isinstance(error, TaskStopped) else "cancelled" if self.cancelled() else "failed"
            stop_error = None
            if begin_attempted:
                try:
                    self.robot.stop(self._mid)
                except Exception as stop_exception:
                    stop_error = str(stop_exception)
            result = TaskResult(state, "begin", str(error), (), False, False, stop_error)
        finally:
            try:
                if begin_attempted:
                    self.robot.finish(self._mid, result is not None and result.state == "succeeded")
            except Exception as error:
                if result is None:
                    result = TaskResult("failed", "finish", str(error), (), False, False, str(error))
                else:
                    reason = (result.reason + "; " if result.reason else "") + "Delivery finish failed: " + str(error)
                    result = replace(result, state="failed", reason=reason,
                                     stop_error=(result.stop_error + "; " if result.stop_error else "") + str(error))
            self._mid = None
            self._context = None
            self._lock.release()
        return result

    run = execute
