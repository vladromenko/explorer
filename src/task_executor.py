"""Shared, synchronous task sequencing with explicit observation/evidence gates.

Ports own sensing, planning, actuation and measured outcome verification. They
must call ``context.check()`` inside long operations and declare any base motion.
Deadlines are cooperative: this module cannot interrupt a blocking hardware API.
No existing robot controller or simulator is selected or run by importing it.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import threading
import time


STAGES = (
    "find", "approach", "reobserve", "pregrasp", "grasp", "lift",
    "verify_hold", "carry", "support", "release", "withdraw", "verify_place",
)


@dataclass(frozen=True)
class Observation:
    sequence: int
    captured_at: float
    base_revision: int
    data: Mapping


@dataclass(frozen=True)
class TaskEvent:
    stage: str
    attempt: int
    at: float
    result: Mapping
    observation_sequence: int | None


@dataclass(frozen=True)
class TaskResult:
    state: str
    stage: str | None
    reason: str | None
    events: tuple
    held_verified: bool
    placed_verified: bool
    stop_error: str | None = None


class TaskStopped(RuntimeError):
    """A failed guard, cancellation or expired deadline stops the transaction."""

    def __init__(self, reason, *, state="failed"):
        super().__init__(reason)
        self.state = state


@dataclass
class TaskContext:
    goal: Mapping
    target: Mapping | None = None
    observation: Observation | None = None
    base_revision: int = 0
    stage: str | None = None
    results: dict = field(default_factory=dict)
    _check: object = field(default=None, repr=False)

    def check(self):
        """Ports call this during trajectories, waits and inference loops."""
        self._check()


class TaskExecutor:
    """Execute one bounded find-to-place transaction using supplied port calls.

    Each operation receives a TaskContext and returns a mapping with ``outcome``
    equal to success/retry/failure/unknown. Find and reobserve additionally return
    a nonempty ``target`` mapping. A verification success requires nonempty
    ``evidence`` with the exact boolean ``confirmed=True``; an ACK is insufficient.

    ``approach`` may return retry with ``base_moved=True``. Each correction forces
    a newly captured observation and target before another approach. Other retry
    requests stop. A failed hold can repeat manipulation only through the explicit
    optional ``recover_grasp`` port, which must safely reset the arm first.

    The observer must acquire a NEW measurement after invocation, with sequence
    greater than its predecessor and the context's current base_revision. Its
    timestamp must use the supplied monotonic clock domain. The guard should
    raise on unsafe/unsupported states. Stop is invoked on every unsuccessful run.
    """

    def __init__(self, operations, observe, *, guard=None, cancelled=None,
                 stop=None, clock=time.monotonic, total_timeout=900.,
                 stage_timeout=90., stage_timeouts=None, observation_max_age=2.,
                 max_find_attempts=3, max_alignment_attempts=4,
                 max_grasp_attempts=1):
        missing = set(STAGES) - set(operations)
        if missing or any(not callable(operations[name]) for name in STAGES):
            raise ValueError("Missing callable task operations: " + ", ".join(sorted(missing)))
        if not callable(observe):
            raise ValueError("An observation port is required")
        for name, value in (("total_timeout", total_timeout), ("stage_timeout", stage_timeout),
                            ("observation_max_age", observation_max_age)):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name, value in (("max_find_attempts", max_find_attempts),
                            ("max_alignment_attempts", max_alignment_attempts),
                            ("max_grasp_attempts", max_grasp_attempts)):
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError(name + " must be between one and twenty")
        self.operations = dict(operations)
        self.observe, self.guard = observe, guard or (lambda context: None)
        self.cancelled, self.stop = cancelled or (lambda: False), stop
        self.clock = clock
        self.total_timeout, self.stage_timeout = total_timeout, stage_timeout
        self.stage_timeouts = dict(stage_timeouts or {})
        for name, value in self.stage_timeouts.items():
            if name not in set(STAGES) | {"recover_grasp"} or type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid stage timeout: " + str(name))
        self.observation_max_age = observation_max_age
        self.max_find_attempts = max_find_attempts
        self.max_alignment_attempts = max_alignment_attempts
        self.max_grasp_attempts = max_grasp_attempts
        self._lock = threading.Lock()

    def run(self, goal):
        """Alias for ports that expose tasks as synchronous run calls."""
        return self.execute(goal)

    def execute(self, goal):
        if not isinstance(goal, Mapping):
            raise ValueError("Task goal must be a mapping")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Task executor is already running")
        context = TaskContext(goal=dict(goal))
        events, attempts = [], {}
        started = self.clock()
        deadline = started + self.total_timeout
        stage_deadline = deadline
        held, placed = False, False

        def check():
            if self.cancelled():
                raise TaskStopped("Task cancelled", state="cancelled")
            now = self.clock()
            if now >= deadline:
                raise TaskStopped("Task deadline expired")
            if now >= stage_deadline:
                raise TaskStopped("Stage deadline expired: " + str(context.stage))
            self.guard(context)

        context._check = check

        def fresh():
            check()
            requested = self.clock()
            previous = context.observation
            observation = self.observe(context)
            check()
            if not isinstance(observation, Observation):
                raise TaskStopped("Observation port did not return an Observation")
            if (type(observation.sequence) is not int or observation.sequence < 0 or
                    type(observation.base_revision) is not int or
                    observation.base_revision != context.base_revision or
                    not isinstance(observation.data, Mapping)):
                raise TaskStopped("Invalid observation identity or base revision")
            now = self.clock()
            if (type(observation.captured_at) not in (float, int) or
                    not math.isfinite(observation.captured_at) or
                    observation.captured_at < requested or observation.captured_at > now or
                    now - observation.captured_at > self.observation_max_age or
                    previous is not None and observation.sequence <= previous.sequence):
                raise TaskStopped("Stale observation; a new measurement is required")
            context.observation = observation
            return observation

        def stage(name):
            nonlocal stage_deadline
            context.stage = name
            stage_deadline = min(deadline, self.clock() + self.stage_timeouts.get(name, self.stage_timeout))
            check()
            fresh()
            attempts[name] = attempts.get(name, 0) + 1
            result = self.operations[name](context)
            check()
            if not isinstance(result, Mapping) or result.get("outcome") not in {"success", "retry", "failure", "unknown"}:
                raise TaskStopped("Missing explicit operation outcome: " + name)
            result = dict(result)
            if "base_moved" in result and type(result["base_moved"]) is not bool:
                raise TaskStopped("Invalid base motion declaration: " + name)
            if result.get("base_moved") is True:
                context.base_revision += 1
                context.target = None
                fresh()
            context.results[name] = result
            events.append(TaskEvent(name, attempts[name], self.clock(), result,
                                    context.observation.sequence))
            return result

        def require(name):
            result = stage(name)
            if result["outcome"] != "success":
                raise TaskStopped(name + ": " + str(result.get("reason", result["outcome"])))
            return result

        def target_from(result):
            target = result.get("target")
            if not isinstance(target, Mapping) or not target:
                raise TaskStopped("Target was not measured: " + str(context.stage))
            context.target = dict(target)

        def verify(name):
            result = require(name)
            evidence = result.get("evidence")
            if not isinstance(evidence, Mapping) or evidence.get("confirmed") is not True:
                raise TaskStopped("Measured evidence did not confirm " + name)

        state, reason, stop_error = "failed", None, None
        try:
            for attempt in range(self.max_find_attempts):
                result = stage("find")
                if result["outcome"] == "success":
                    target_from(result)
                    break
                if result["outcome"] != "retry" or attempt + 1 == self.max_find_attempts:
                    raise TaskStopped("find: " + str(result.get("reason", result["outcome"])))
            for attempt in range(self.max_alignment_attempts):
                result = stage("approach")
                if result["outcome"] == "success":
                    break
                if (result["outcome"] != "retry" or result.get("base_moved") is not True or
                        attempt + 1 == self.max_alignment_attempts):
                    raise TaskStopped("Approach did not converge within its correction budget")
                target_from(require("reobserve"))
            target_from(require("reobserve"))
            for grasp_attempt in range(self.max_grasp_attempts):
                require("pregrasp")
                require("grasp")
                require("lift")
                result = stage("verify_hold")
                if result["outcome"] == "success":
                    evidence = result.get("evidence")
                    if not isinstance(evidence, Mapping) or evidence.get("confirmed") is not True:
                        raise TaskStopped("Measured evidence did not confirm verify_hold")
                    held = True
                    break
                if (result["outcome"] != "failure" or grasp_attempt + 1 == self.max_grasp_attempts or
                        not callable(self.operations.get("recover_grasp"))):
                    raise TaskStopped("Hold unconfirmed: " + str(result.get("reason", result["outcome"])))
                require("recover_grasp")
                target_from(require("reobserve"))
            require("carry")
            require("support")
            require("release")
            require("withdraw")
            verify("verify_place")
            placed = True
            state = "succeeded"
        except TaskStopped as exc:
            state, reason = exc.state, str(exc)
        except Exception as exc:
            reason = type(exc).__name__ + ": " + str(exc)
        finally:
            if state != "succeeded" and self.stop is not None:
                try:
                    self.stop(context)
                except Exception as exc:
                    stop_error = type(exc).__name__ + ": " + str(exc)
            self._lock.release()
        return TaskResult(state, context.stage, reason, tuple(events), held, placed, stop_error)
