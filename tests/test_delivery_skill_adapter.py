"""Shared-executor binding tested against a fake synchronous robot only."""
from types import SimpleNamespace

import numpy as np
import pytest

from delivery_skill_adapter import DeliverySkillAdapter


class Clock:
    def __init__(self):
        self.now = 100.
    def monotonic(self):
        return self.now
    def wall(self):
        return self.now + 1000.


class Robot:
    def __init__(self, clock):
        self.clock = clock
        self.owner = None
        self.check = None
        self.calls = []
        self.tracking = False
        self.cancelled = False
        self.alignments = 0
        self.hold_outcome = "success"
        self.place_outcome = "success"
        self.sequence = 0
        self.before_hold = None
        self.before_release = None
        self.vision = SimpleNamespace(wait_snapshot=self.snapshot)

    def call(self, name):
        if self.check:
            self.check()
        self.calls.append(name)

    def begin(self, mid, check, goal=None):
        self.check = check
        check()
        self.owner = mid
        self.calls.append(("begin", mid, goal))
        check()

    def permit(self, mid):
        self.check()
        if self.owner != mid:
            raise ValueError("Robot owner changed")

    def snapshot(self, permit, after_stamp):
        permit()
        self.calls.append("raw_snapshot")
        self.clock.now += .01
        return {"stamp": np.asarray(self.clock.wall()), "frame": "wrist_rgb"}

    def find(self):
        self.call("find")
        self.tracking = True
        return {"place": "floor", "target": {"object_xyz": [.25, 0., .02]},
                "semantic_identity_verified": False}

    def find_aligned(self):
        self.call("find_aligned")
        self.tracking = True
        return {"place": "aligned", "target": {"object_xyz": [.23, 0., .02]}}

    def approach(self, target):
        self.call("approach")
        assert "target" in target
        if self.alignments:
            self.alignments -= 1
            self.tracking = False
            return {"base_repositioned": True, "reobserve_required": True}
        return {"reachable": True, "base_stationary_confirmed": True}

    def hold(self):
        self.call("hold")
        return {"base_stationary_confirmed": True}

    def inspect(self):
        self.call("inspect")
        if not self.tracking:
            raise ValueError("Tracker unavailable")
        return [{"observed": True}]

    def reobserve(self, target):
        self.call("reobserve")
        assert "target" in target
        return {"object_xyz": [.23, 0., .02], "object_id": "sock"}

    def align(self, target):
        self.call("align")
        return {"aligned": True}

    def motion(self, name):
        self.call(name)
        return {"command_completed": True, "attained": False, "measured": False}

    def pregrasp(self, target):
        return self.motion("pregrasp")

    def grasp(self, target):
        return self.motion("grasp")

    def observe(self):
        self.call("observe")
        if not self.tracking:
            raise ValueError("Tracked observation requested before find")
        self.sequence += 1
        return [{"at": self.sequence, "object_id": "sock"} for _ in range(3)]

    def lift(self):
        return self.motion("lift")

    def verify_hold(self, before):
        self.call("verify_hold")
        self.before_hold = before
        return {"outcome": self.hold_outcome, "evidence": {"verifier": "rgbd_lift_v2"}}

    def transport(self):
        return self.motion("transport")

    def carry(self):
        self.call("carry")
        return {"x": 1., "y": 0., "yaw": 0.}

    def support(self):
        return self.motion("support")

    def release(self):
        return self.motion("release")

    def withdraw(self):
        return self.motion("withdraw")

    def verify_place(self, before):
        self.call("verify_place")
        self.before_release = before
        return {"outcome": self.place_outcome, "evidence": {"verifier": "rgbd_place_v2"}}

    def stop(self, mid):
        self.calls.append(("stop", mid))
        if self.owner == mid:
            self.owner = None

    def finish(self, mid, success):
        self.calls.append(("finish", mid, success))
        if self.owner == mid:
            self.owner = None


def adapter(robot, clock, **options):
    return DeliverySkillAdapter(robot, clock=clock.monotonic, wall_clock=clock.wall,
                                cancelled=lambda: robot.cancelled, **options)


def test_construction_is_inert_and_execute_preserves_robot_stage_order():
    clock = Clock()
    robot = Robot(clock)
    bridge = adapter(robot, clock)
    assert robot.calls == []
    result = bridge.execute({"object_query": "sock", "destination": {"name": "basket"}})
    assert result.state == "succeeded", result.reason
    assert result.held_verified and result.placed_verified
    assert robot.calls[0][0] == "begin" and robot.calls[-1][0] == "finish"
    assert robot.calls[0][2]["object_query"] == "sock"
    assert robot.calls[-1][2] is True and robot.owner is None
    names = [name for name in robot.calls if isinstance(name, str)]
    assert names.index("find") < names.index("approach") < names.index("align") < names.index("pregrasp")
    assert names.index("grasp") < names.index("observe") < names.index("lift") < names.index("verify_hold")
    assert names.index("transport") < names.index("carry") < names.index("support") < names.index("release")
    assert robot.before_hold[0]["at"] == 1
    assert robot.before_release[0]["at"] == 2
    assert names.count("observe") == 2
    assert names[0] == "raw_snapshot"  # no tracker dependency before find


def test_base_reposition_forces_aligned_search_and_new_rgbd_before_retry():
    clock = Clock()
    robot = Robot(clock)
    robot.alignments = 2
    result = adapter(robot, clock).execute()
    assert result.state == "succeeded", result.reason
    names = [name for name in robot.calls if isinstance(name, str)]
    assert names.count("approach") == 3
    assert names.count("find_aligned") == 2
    indices = [index for index, name in enumerate(names) if name == "approach"]
    for first, second in zip(indices, indices[1:]):
        assert "find_aligned" in names[first + 1:second]
        assert "raw_snapshot" in names[first + 1:second]


def test_alignment_budget_stops_before_manipulation():
    clock = Clock()
    robot = Robot(clock)
    robot.alignments = 10
    result = adapter(robot, clock).execute()
    assert result.state == "failed"
    assert robot.calls.count("approach") == 4
    assert "grasp" not in robot.calls and "pregrasp" not in robot.calls
    assert robot.calls[-2][0] == "stop" and robot.calls[-1][0] == "finish"


@pytest.mark.parametrize("stage", ["verify_hold", "verify_place"])
@pytest.mark.parametrize("outcome", ["unknown", "failure"])
def test_unconfirmed_robot_verifier_never_becomes_success(stage, outcome):
    clock = Clock()
    robot = Robot(clock)
    setattr(robot, "hold_outcome" if stage == "verify_hold" else "place_outcome", outcome)
    result = adapter(robot, clock).execute()
    assert result.state == "failed"
    assert result.stage == stage
    if stage == "verify_hold":
        assert "carry" not in robot.calls
    assert robot.calls[-1][2] is False


@pytest.mark.parametrize("result", [{"accepted": True}, {"outcome": "success"},
                                    {"outcome": "success", "evidence": {}}])
def test_verification_needs_measured_evidence_not_acknowledgement(result):
    clock = Clock()
    robot = Robot(clock)
    robot.verify_hold = lambda before: result
    value = adapter(robot, clock).execute()
    assert value.state == "failed"
    assert "carry" not in robot.calls


def test_command_acceptance_does_not_count_as_grasp_completion():
    clock = Clock()
    robot = Robot(clock)
    robot.grasp = lambda target: {"accepted": True, "completed": False}
    result = adapter(robot, clock).execute()
    assert result.state == "failed" and result.stage == "grasp"
    assert "lift" not in robot.calls


def test_guarded_grasp_completion_preserves_evidence_without_claiming_hold():
    clock = Clock()
    robot = Robot(clock)
    robot.grasp = lambda target: {"guarded": True, "decision": {"action": "hold"}, "final_deg": 110.}
    robot.hold_outcome = "unknown"
    result = adapter(robot, clock).execute()
    grasp = next(event.result for event in result.events if event.stage == "grasp")
    assert grasp["outcome"] == "success"
    assert grasp["evidence"]["object_outcome_confirmed"] is False
    assert result.state == "failed" and not result.held_verified


def test_cached_raw_camera_blocks_before_find():
    clock = Clock()
    robot = Robot(clock)
    robot.vision.wait_snapshot = lambda permit, after_stamp: {"stamp": after_stamp}
    result = adapter(robot, clock).execute()
    assert result.state == "failed" and result.stage == "find"
    assert "find" not in robot.calls
    assert robot.calls[-2][0] == "stop" and robot.calls[-1][2] is False


def test_begin_deadline_uses_owner_check_and_finishes_partial_owner():
    clock = Clock()
    robot = Robot(clock)
    original = robot.begin
    def begin(*args, **options):
        original(*args, **options)
        clock.now += 2.
        options["check"]()
    robot.begin = begin
    result = adapter(robot, clock, begin_timeout=1.).execute()
    assert result.state == "failed" and result.stage == "begin"
    assert "begin deadline" in result.reason
    assert robot.owner is None
    assert robot.calls[-2][0] == "stop" and robot.calls[-1][0] == "finish"


def test_cancellation_inside_robot_loop_stops_owned_task():
    clock = Clock()
    robot = Robot(clock)
    def grasp(target):
        robot.cancelled = True
        robot.check()
        pytest.fail("Cancelled grasp continued")
    robot.grasp = grasp
    result = adapter(robot, clock).execute()
    assert result.state == "cancelled"
    assert "lift" not in robot.calls
    assert robot.owner is None
    assert robot.calls[-1][2] is False


def test_begin_budget_is_included_in_total_deadline():
    clock = Clock()
    robot = Robot(clock)
    original_begin = robot.begin
    def begin(*args, **options):
        original_begin(*args, **options)
        clock.now += 2.
    robot.begin = begin
    original_lift = robot.lift
    def lift():
        clock.now += 2.
        return original_lift()
    robot.lift = lift
    result = adapter(robot, clock, total_timeout=3.).execute()
    assert result.state == "failed"
    assert "deadline" in result.reason
    assert "carry" not in robot.calls


def test_cleanup_error_does_not_report_success_and_lock_can_be_reused():
    clock = Clock()
    robot = Robot(clock)
    bridge = adapter(robot, clock)
    original = robot.finish
    def finish(mid, success):
        original(mid, success)
        raise OSError("cleanup unavailable")
    robot.finish = finish
    result = bridge.execute()
    assert result.state == "failed" and "cleanup unavailable" in result.stop_error
    robot.finish = original
    assert bridge.execute().state == "succeeded"


def test_same_adapter_cannot_start_second_owner_during_find():
    clock = Clock()
    robot = Robot(clock)
    bridge = adapter(robot, clock)
    original = robot.find
    def find():
        with pytest.raises(RuntimeError, match="already running"):
            bridge.execute()
        return original()
    robot.find = find
    result = bridge.execute()
    assert result.state == "succeeded"
    assert len([row for row in robot.calls if isinstance(row, tuple) and row[0] == "begin"]) == 1


def test_missing_raw_camera_capability_blocks_before_begin_or_stop():
    clock = Clock()
    robot = Robot(clock)
    robot.vision = SimpleNamespace()
    result = adapter(robot, clock).execute()
    assert result.state == "failed" and "raw RGB-D" in result.reason
    assert robot.calls == []


def test_pre_cancelled_task_does_not_claim_or_stop_robot():
    clock = Clock()
    robot = Robot(clock)
    robot.cancelled = True
    result = adapter(robot, clock).execute()
    assert result.state == "cancelled"
    assert robot.calls == []


def test_lost_robot_ownership_blocks_further_actions_without_stopping_foreign_owner():
    clock = Clock()
    robot = Robot(clock)
    original = robot.grasp
    def grasp(target):
        result = original(target)
        robot.owner = "another-task"
        return result
    robot.grasp = grasp
    result = adapter(robot, clock).execute()
    assert result.state == "failed" and "owner changed" in result.reason
    assert "lift" not in robot.calls
    assert robot.owner == "another-task"


def test_package_import_is_inert_with_only_repository_root_on_search_path():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = (
        "import sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        "import src.delivery_skill_adapter as adapter; "
        "assert 'src.delivery_robot' not in sys.modules; "
        "assert 'delivery_robot' not in sys.modules; "
        "assert 'task_executor' not in sys.modules; "
        "assert adapter.DeliverySkillAdapter is not None"
    )
    result = subprocess.run([sys.executable, "-I", "-c", source], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
