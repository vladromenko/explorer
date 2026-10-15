"""Exercise the skill boundary with fake ports; no simulator contact oracle."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simulator import skill_port, skill_workshop
from task_executor import TaskResult


class SensorOnlyEnv:
    """Actor-compatible kinematics. Any read of task truth is an error."""
    def __init__(self):
        self.command = np.r_[np.ones(6) * 90., [0., 0., 0.]]
        self.pending_command = self.command.copy()
        self.data = SimpleNamespace(time=0., ctrl=np.ones(10))
        self.wheel_ids = [6, 7, 8, 9]
        self.steps = 0
        self.phase = "test"
        self.last_error = None

    def step(self, command):
        self.command = np.asarray(command).copy()
        self.pending_command = self.command.copy()
        self.steps += 1
        self.data.time += .05

    def state(self):
        raise AssertionError("Actor read privileged evaluator state")

    def base_pose(self):
        return [0., 0., 0.]


def test_actor_progress_does_not_read_scene_truth(tmp_path):
    env = SensorOnlyEnv()
    progress = []
    port = skill_port.SimulatorSkillPort(env, tmp_path / "execution", progress=progress.append)
    try:
        for _ in range(5):
            port.tick(env.command[:6])
        assert progress
        assert all("task" not in update and "sock_center_m" not in update for update in progress)
    finally:
        port.close()


def test_close_stops_wheels_even_when_observation_save_fails(tmp_path, monkeypatch):
    env = SensorOnlyEnv()
    port = skill_port.SimulatorSkillPort(env, tmp_path / "execution", cancelled=lambda: True)
    env.command[6:] = .05
    env.pending_command[6:] = .05
    original = type(tmp_path).write_text
    def write(path, *args, **kwargs):
        if path.name == "observations.json":
            raise OSError("disk write failed")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(type(tmp_path), "write_text", write)
    with pytest.raises(OSError, match="disk write failed"):
        port.close()
    assert np.all(env.command[6:] == 0)
    assert np.all(env.pending_command[6:] == 0)
    assert np.all(env.data.ctrl[env.wheel_ids] == 0)
    assert port.handle.closed


class TrialEnv:
    def __init__(self, success=False, reset_error=None):
        self.command = np.zeros(9)
        self.reset_error = reset_error
        self.success = success
        self.closed_port = False
        self.stopped = False
        self.truth_calls = 0

    def reset(self, **options):
        if self.reset_error:
            raise ValueError(self.reset_error)

    def state(self):
        self.truth_calls += 1
        assert self.stopped, "Independent evaluation ran before actor stop"
        return {"task": {"success": self.success}}


class FakeTrialPort:
    def __init__(self, env, folder, *args):
        self.env = env

    def operations(self):
        return {}

    def observe(self, context):
        raise AssertionError("Fake executor must not use a physics observation")

    def check(self, context=None):
        pass

    def tick(self, command, **options):
        self.env.stopped = True

    def stop_motion(self, *args):
        self.env.stopped = True

    def close(self):
        self.env.stopped = True
        self.env.closed_port = True


def fake_trial(monkeypatch, result):
    class Executor:
        def __init__(self, *args, **options):
            pass
        def execute(self, goal):
            return result
    monkeypatch.setattr(skill_port, "SimulatorSkillPort", FakeTrialPort)
    monkeypatch.setattr("src.task_executor.TaskExecutor", Executor)
    monkeypatch.setattr("simulator.explorer_model.model_metadata", lambda: {"source": "fixture"})


def test_trial_stops_actor_before_independent_outcome_read(tmp_path, monkeypatch):
    result = TaskResult("failed", "verify_hold", "occluded", (), False, False)
    fake_trial(monkeypatch, result)
    env = TrialEnv()
    report = skill_workshop.trial(env, tmp_path, "a" * 12, "sock_basket", 42)
    assert env.truth_calls == 1 and env.closed_port
    assert report["state"] == "diagnosed" and report["success"] is False
    assert report["execution"] == asdict(result)
    assert report["automatic_execution_allowed"] is False
    assert report["learned_residuals_executed"] is False
    assert report["simulation_ready"] is False and report["physical_ready"] is False


def test_executor_success_cannot_override_independent_failed_outcome(tmp_path, monkeypatch):
    fake_trial(monkeypatch, TaskResult("succeeded", "verify_place", None, (), True, True))
    report = skill_workshop.trial(TrialEnv(success=False), tmp_path, "b" * 12, "sock_basket", 3)
    assert report["execution"]["state"] == "succeeded"
    assert report["success"] is False
    assert report["qualification"] is False


def test_failed_setup_is_persisted_instead_of_remaining_running(tmp_path, monkeypatch):
    fake_trial(monkeypatch, TaskResult("failed", "find", "unused", (), False, False))
    with pytest.raises(ValueError, match="reset failed"):
        skill_workshop.trial(TrialEnv(reset_error="reset failed"), tmp_path, "c" * 12, "sock_basket", 2)
    saved = json.loads((tmp_path / "skill-workshops" / ("c" * 12) / "state.json").read_text())
    assert saved["state"] == "failed"
    assert "reset failed" in saved["reason"]


def test_listing_skips_invalid_json_shapes_and_keeps_valid_records(tmp_path):
    folder = tmp_path / "skill-workshops"
    for index, value in enumerate((None, [], {}, {"created": "bad"}, {"created": 1., "id": "valid"})):
        path = folder / str(index) / "state.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    rows = skill_workshop.list_workshops(tmp_path)
    assert [row["id"] for row in rows] == ["valid"]


def demonstration(root, name="demo", task="sock_basket", provenance=None):
    folder = root / "demonstrations" / name
    folder.mkdir(parents=True)
    (folder / "episode.json").write_text(json.dumps(dict(task_id=task, success=True,
        attempt_completed=True, generator="ground_truth_teacher", finished_at=1,
        model_provenance=provenance or {"geometry": "fixture"})))
    return folder


def test_preparation_is_finite_and_never_marks_execution_or_transfer_ready(tmp_path, monkeypatch):
    episode = demonstration(tmp_path)
    calls = []
    def dataset(episodes, output, **options):
        calls.append(("dataset", episodes))
        output.write_text("{}")
    def fit(dataset_path, output, **options):
        calls.append(("fit", dataset_path))
        output.write_text("{}")
    monkeypatch.setattr("simulator.phase_learning.prepare_phase_dataset", dataset)
    monkeypatch.setattr("simulator.phase_learning.fit_phase_residuals", fit)
    state = skill_workshop.prepare(tmp_path, "d" * 12, "sock_basket")
    assert state["state"] == "prepared"
    assert state["selected_episodes"] == [str(episode)]
    assert [call[0] for call in calls] == ["dataset", "fit"]
    assert state["learned_residuals_executed"] is False
    assert state["automatic_execution_allowed"] is False
    assert not state["simulation_ready"] and not state["physical_ready"]
    assert skill_workshop.prepare(tmp_path, "d" * 12, "sock_basket") == state
    assert len(calls) == 2


def test_preparation_operator_cancel_is_saved_and_does_not_fit(tmp_path, monkeypatch):
    demonstration(tmp_path)
    monkeypatch.setattr("simulator.phase_learning.fit_phase_residuals",
                        lambda *a, **k: pytest.fail("fit started after cancellation"))
    with pytest.raises(InterruptedError):
        skill_workshop.prepare(tmp_path, "e" * 12, "sock_basket", cancelled=lambda: True)
    saved = json.loads((tmp_path / "skill-workshops" / ("e" * 12) / "state.json").read_text())
    assert saved["state"] == "paused"
    assert saved["physical_ready"] is False


@pytest.mark.parametrize("evidence", [
    {},
    {"self_occlusion_not_resolved": True, "identity_association_verified": True},
    {"self_occlusion_not_resolved": False},
    {"self_occlusion_not_resolved": False, "identity_association_verified": 1},
])
def test_hold_and_place_reject_unresolved_or_unassociated_geometry(tmp_path, evidence):
    env = SensorOnlyEnv()
    env.jaw_center = lambda: [.3, 0., .12]
    port = skill_port.SimulatorSkillPort(env, tmp_path / "execution")
    candidate = dict(center=[.3, 0., .12], bounds_min=[.28, -.02, .10],
                     bounds_max=[.32, .02, .14], evidence=evidence)
    port.scene = lambda ctx: {"frame_id": "fresh-render", "object_candidates": [candidate]}
    port.destination = dict(bounds_min=[.2, -.1, .02], bounds_max=[.5, .1, .3])
    try:
        assert port.verify_hold(None)["outcome"] == "unknown"
        assert port.verify_place(None)["outcome"] == "unknown"
    finally:
        port.close()
