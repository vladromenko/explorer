"""Checks the simulator boundary and recordings without requiring an OpenGL context."""
import json
from pathlib import Path
import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from simulator.server import Control, Runtime, create_app


class FakeEnv:
    def __init__(self):
        self.command = [90., 110., 120., 90., 90., 40., 0., 0., 0.]
        self.pending = self.command.copy()
        self.steps = []
        self.reset_threads = []
        self.render_threads = []
        self.closed = False

    def reset(self, seed=None, options=None):
        self.reset_threads.append(threading.current_thread().name)
        return np.asarray(self.command), {}

    def state(self):
        return {'command': self.command.copy(), 'measured_sim': self.command[:6], 'base_pose': [0, 0, 0], 'task': {'success': False, 'grasp_contact': False}, 'sim_time': len(self.steps) * .05}

    def set_manual(self, command):
        self.pending = list(command)

    def capture_sample(self):
        return {'at': len(self.steps) * .05, 'observation': self.command.copy(), 'action': self.pending.copy(), 'simulation_time': len(self.steps) * .05}

    def step(self, action):
        self.command = list(action)
        self.steps.append(self.command.copy())
        return np.asarray(self.command), 0., False, False, {'success': False}

    def render_rgb(self, camera='wrist'):
        self.render_threads.append(threading.current_thread().name)
        return np.zeros((24, 32, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def client(tmp_path, start_loop=False):
    return TestClient(create_app(env_factory=FakeEnv, data_root=tmp_path, start_loop=start_loop), base_url='http://127.0.0.1:8766')


def auth(connection):
    return {'X-Explorer-Sim-Token': connection.get('/api/session').json()['token']}


def test_maintenance_cancel_is_owned_and_operator_cancel_takes_precedence(tmp_path):
    with client(tmp_path) as connection:
        headers=auth(connection);runtime=connection.app.state.runtime
        cycle=runtime.learning_cycles.create(['toy_box'])
        runtime.jobs=[dict(id='owner',kind='cycle',cycle_id=cycle['id'],status='running'),
                      dict(id='audit',kind='audit',status='running')]
        response=connection.post('/api/maintenance-stop/'+cycle['id'],json={},headers=headers)
        assert response.status_code==200
        token=response.json()['maintenance_token']
        assert runtime.cancel_job.is_set()
        assert runtime.cancellation_context==dict(kind='maintenance',cycle_id=cycle['id'],token=token)
        connection.post('/api/cancel',json={},headers=headers)
        assert runtime.cancellation_context=={} and runtime.cancel_job.is_set()


def test_maintenance_cancel_refuses_live_training_and_existing_operator_pause(tmp_path):
    with client(tmp_path) as connection:
        headers=auth(connection);runtime=connection.app.state.runtime
        cycle=runtime.learning_cycles.create(['toy_box'])
        runtime.jobs=[dict(id='owner',kind='cycle',cycle_id=cycle['id'],status='running'),
                      dict(id='learner',kind='train',status='running')]
        assert connection.post('/api/maintenance-stop/'+cycle['id'],json={},headers=headers).status_code==409
        assert not runtime.cancel_job.is_set()
        runtime.jobs.pop();cycle.update(state='paused',pause_kind='operator');runtime.learning_cycles.save(cycle)
        assert connection.post('/api/maintenance-stop/'+cycle['id'],json={},headers=headers).status_code==409
        assert not runtime.cancel_job.is_set()


def test_local_authenticated_mutations_only(tmp_path):
    with client(tmp_path) as connection:
        headers = auth(connection)
        assert connection.post('/api/control', json={}).status_code == 403
        assert connection.post('/api/control', json={}, headers=headers).status_code == 200
        assert connection.post('/api/control', json={}, headers={**headers, 'Origin': 'https://foreign.example'}).status_code == 403
        assert connection.get('/api/session', headers={'Host': 'foreign.example'}).status_code == 403
        assert connection.get('/api/session', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
        assert connection.get('/').headers['content-security-policy'].endswith("frame-ancestors 'none'")


def test_only_current_policy_independent_rollouts_grant_simulation_task_status(tmp_path):
    runtime = Runtime.__new__(Runtime)
    runtime.latest_job_folder = tmp_path
    bundle = {'checkpoint_sha256': {'model.safetensors': 'current'}}
    report = {'validation_kind': 'closed_loop_policy', 'model_weights_sha256': 'current',
              'heldout_independent': True, 'episodes': 10, 'successes': 10}
    path = tmp_path / 'physics-validation.json'
    path.write_text(json.dumps(report))
    runtime.refresh_model_info(bundle)
    assert runtime.model_info['task_validated_simulation'] is True
    assert runtime.model_info['hardware_accepted'] is False
    for values in ({'successes': 9}, {'episodes': 1, 'successes': 1},
                   {'validation_kind': 'teacher_physics_diagnostics'},
                   {'model_weights_sha256': 'other'}, {'heldout_independent': False},
                   {'model_checkpoint_sha256': {'config.json': 'changed'}}):
        path.write_text(json.dumps({**report, **values}))
        runtime.refresh_model_info(bundle)
        assert runtime.model_info['task_validated_simulation'] is False

    path.write_text(json.dumps(report))
    (tmp_path / 'preview-validation.json').write_text(json.dumps({**report, 'episodes': 1, 'successes': 0}))
    runtime.refresh_model_info(bundle)
    assert runtime.model_info['task_validated_simulation'] is True


def test_exact_nine_command_contract(tmp_path):
    with client(tmp_path) as connection:
        headers = auth(connection)
        assert connection.post('/api/control', json={'arm_velocity': [1] * 9}, headers=headers).status_code == 422
        assert connection.post('/api/control', json={'arm_velocity': [1] * 6, 'base': [.1, 0, 0]}, headers=headers).status_code == 200
        assert connection.post('/api/control', json={'base': [0, 0]}, headers=headers).status_code == 422


def test_recorded_annotation_cannot_claim_physics_success(tmp_path):
    with client(tmp_path) as connection:
        headers = auth(connection)
        assert connection.post('/api/record/start', json={}, headers=headers).status_code == 200
        result = connection.post('/api/record/stop', json={'annotation': 'success'}, headers=headers).json()
        assert result['operator_annotation'] == 'success'
        assert result['success'] is False
        assert result['physics_verified_success'] is False
        meta = json.loads(next(tmp_path.glob('demonstrations/*/episode.json')).read_text())
        assert meta['source'] == 'simulation'
        assert meta['state'] == 'complete'


def test_commands_expire_and_render_stays_on_owner(tmp_path):
    runtime = Runtime(FakeEnv, tmp_path)
    runtime.control(Control(arm_velocity=[0.] * 6, base=[.2, .1, .3]))
    runtime.start()
    deadline = time.monotonic() + 2
    while len(runtime.env.steps) < 11 and time.monotonic() < deadline:
        time.sleep(.025)
    runtime.close()
    assert len(runtime.env.steps) >= 11
    assert any(action[6] == .2 for action in runtime.env.steps)
    assert runtime.env.steps[-1][6:] == [0., 0., 0.]
    assert all(len(action) == 9 for action in runtime.env.steps)
    assert set(runtime.env.render_threads) == {'explorer-simulation'}
    assert runtime.env.closed


def test_manual_rows_are_pre_action_and_shutdown_saves(tmp_path):
    runtime = Runtime(FakeEnv, tmp_path)
    previous = runtime.env.command.copy()
    runtime.start_recording()
    runtime.control(Control(arm_velocity=[10., 0, 0, 0, 0, 0], base=[.1, 0, 0]))
    runtime.start()
    deadline = time.monotonic() + 2
    while runtime.record_frames < 2 and time.monotonic() < deadline:
        time.sleep(.02)
    runtime.close()
    folder = next(tmp_path.glob('demonstrations/*'))
    rows = [json.loads(line) for line in (folder / 'samples.jsonl').read_text().splitlines()]
    assert rows[0]['observation'] == previous
    assert rows[0]['action'][0] > previous[0]
    assert len(rows[0]['action']) == 9
    assert (folder / rows[0]['wrist_rgb']).is_file()
    assert json.loads((folder / 'episode.json').read_text())['state'] == 'complete'


def test_reset_runs_on_physics_owner_thread(tmp_path):
    with client(tmp_path, start_loop=True) as connection:
        assert connection.post('/api/reset', json={'seed': 12}, headers=auth(connection)).status_code == 200
        assert connection.app.state.runtime.env.reset_threads[-1] == 'explorer-simulation'


def test_job_cannot_start_during_recording(tmp_path):
    with client(tmp_path) as connection:
        headers = auth(connection)
        connection.post('/api/record/start', json={}, headers=headers)
        assert connection.post('/api/jobs/trials', json={'count': 1}, headers=headers).status_code == 409
        assert connection.post('/api/jobs/arbitrary', json={}, headers=headers).status_code == 404
        assert connection.get('/api/package').status_code == 404


def test_selected_episode_ids_cannot_escape_storage(tmp_path):
    with client(tmp_path) as connection:
        assert connection.post('/api/jobs/train', json={'episode_ids': ['../../other']}, headers=auth(connection)).status_code == 422


def test_failed_training_is_not_reported_complete(tmp_path):
    class FailedLearning:
        @staticmethod
        def train_job(**kwargs):
            return {'state': 'failed', 'error': 'Actual learner failed'}
    folder = tmp_path / 'demonstrations' / 'sample'
    folder.mkdir(parents=True)
    (folder / 'episode.json').write_text('{"success": true}')
    runtime = Runtime(FakeEnv, tmp_path, learning_module=FailedLearning)
    from simulator.server import JobRequest
    runtime.launch_job('train', JobRequest())
    runtime.job_thread.join(timeout=2)
    assert runtime.jobs[-1]['status'] == 'failed'
    assert runtime.latest_job_folder is None
    assert runtime.jobs[-1]['message'] == 'Actual learner failed'
    runtime.close()


def test_recorded_action_is_applied_endpoint_after_rate_limit(tmp_path):
    class LimitedEnv(FakeEnv):
        def step(self, action):
            limited = list(action)
            limited[0] = min(limited[0], self.command[0] + .1)
            return super().step(limited)
    runtime = Runtime(LimitedEnv, tmp_path)
    original = runtime.env.command.copy()
    runtime.start_recording()
    runtime.control(Control(arm_velocity=[30., 0, 0, 0, 0, 0]))
    runtime.start()
    deadline = time.monotonic() + 2
    while runtime.record_frames < 2 and time.monotonic() < deadline:
        time.sleep(.02)
    runtime.close()
    row = json.loads(next(tmp_path.glob('demonstrations/*/samples.jsonl')).read_text().splitlines()[0])
    assert row['observation'] == original
    assert abs(row['action'][0] - (original[0] + .1)) < 1e-8
    assert row['applied_action'] == row['action']
    assert row['action_alignment'] == 'same_row_pre_action'


def test_failed_attempts_are_diagnostics_not_training(tmp_path):
    folder = tmp_path / 'demonstrations' / 'failed-trial'
    folder.mkdir(parents=True)
    (folder / 'episode.json').write_text('{"success": false}')
    runtime = Runtime(FakeEnv, tmp_path)
    from simulator.server import JobRequest
    runtime.launch_job('train', JobRequest())
    runtime.job_thread.join(timeout=2)
    assert runtime.jobs[-1]['status'] == 'failed'
    assert 'Нет демонстраций' in runtime.jobs[-1]['message']
    runtime.close()


def test_latest_model_skips_failed_incomplete_and_corrupt_jobs(tmp_path):
    import os
    class BundleReader:
        @staticmethod
        def read_bundle(folder):
            if not (folder / 'model.safetensors').is_file():
                raise ValueError('Missing real checkpoint')
            return {'checkpoint_sha256': {'model.safetensors': 'verified'}}, folder
    for name, status, bundle, checkpoint, stamp in [
        ('good', 'validated_simulation', True, True, 10),
        ('failed-newer', 'failed', True, True, 20),
        ('unfinished-newer', 'training', True, True, 30),
        ('corrupt-newer', 'validated_simulation', True, False, 40),
        ('no-bundle-newer', 'validated_simulation', False, True, 50),
    ]:
        folder = tmp_path / 'runs' / name
        folder.mkdir(parents=True)
        (folder / 'job.json').write_text(json.dumps({'state': status}))
        os.utime(folder / 'job.json', (stamp, stamp))
        if bundle:
            (folder / 'bundle.json').write_text('{}')
        if checkpoint:
            (folder / 'model.safetensors').write_bytes(b'actual checkpoint fixture')
    runtime = Runtime(FakeEnv, tmp_path, learning_module=BundleReader)
    assert runtime.latest_job_folder.name == 'good'
    assert runtime.latest_model().name == 'good'
    runtime.close()


def test_held_key_at_joint_limit_clamps_without_pausing(tmp_path):
    from types import SimpleNamespace
    class BoundedEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.command[0] = 179.8
            self.action_space = SimpleNamespace(low=np.array([0, 0, 0, 0, 0, 30, -.8, -.72, -1.67]), high=np.array([180, 180, 180, 180, 270, 170, .8, .72, 1.67]))
        def set_manual(self, command):
            assert np.all(np.asarray(command) <= self.action_space.high)
            assert np.all(np.asarray(command) >= self.action_space.low)
            super().set_manual(command)
    runtime = Runtime(BoundedEnv, tmp_path)
    runtime.control(Control(arm_velocity=[45., 0, 0, 0, 0, 0]))
    runtime.start()
    time.sleep(.2)
    runtime.close()
    assert runtime.error is None
    assert runtime.env.steps[-1][0] == 180
    assert runtime.paused is False


def test_lost_heartbeat_stops_unfinished_pose_goal_and_base(tmp_path):
    runtime = Runtime(FakeEnv, tmp_path)
    runtime.control(Control(arm_target=[160., 110., 120., 90., 90., 40.], base=[.2, 0, 0]))
    runtime.start()
    deadline = time.monotonic() + 2
    while len(runtime.env.steps) < 13 and time.monotonic() < deadline:
        time.sleep(.02)
    runtime.close()
    assert runtime.pose_goal is None
    assert runtime.env.steps[-1][6:] == [0., 0., 0.]
    assert runtime.env.steps[-4][:6] == runtime.env.steps[-1][:6]
    assert runtime.env.steps[-1][0] < 160


def test_manual_scene_ignores_training_time_truncation(tmp_path):
    class ShortBudgetEnv(FakeEnv):
        def step(self, action):
            obs, reward, terminated, _, info = super().step(action)
            return obs, reward, terminated, True, info
    runtime = Runtime(ShortBudgetEnv, tmp_path)
    runtime.start()
    deadline = time.monotonic() + 2
    while len(runtime.env.steps) < 4 and time.monotonic() < deadline:
        time.sleep(.02)
    runtime.close()
    assert len(runtime.env.steps) >= 4
    assert runtime.done is False


def test_shutdown_cancels_and_persists_long_training_job(tmp_path):
    started = threading.Event()
    class CancellableLearning:
        @staticmethod
        def train_job(**kwargs):
            started.set()
            while not kwargs['cancelled']():
                time.sleep(.01)
            return {'state': 'cancelled', 'error': 'Cancelled during shutdown'}
    folder = tmp_path / 'demonstrations' / 'success'
    folder.mkdir(parents=True)
    (folder / 'episode.json').write_text('{"success": true}')
    runtime = Runtime(FakeEnv, tmp_path, learning_module=CancellableLearning)
    from simulator.server import JobRequest
    runtime.start()
    runtime.launch_job('train', JobRequest())
    assert started.wait(2)
    runtime.close()
    assert not runtime.job_thread.is_alive()
    assert not runtime.thread.is_alive()
    assert runtime.jobs[-1]['status'] == 'cancelled'
    assert json.loads(next((tmp_path / 'logs').glob('job-*.json')).read_text())['status'] == 'cancelled'
    assert runtime.log_file.closed


def test_data_root_respects_alternative_simulator_volume(tmp_path):
    import os
    import subprocess
    import sys
    environment = dict(os.environ, EXPLORER_SIM_VOLUME=str(tmp_path / 'External Sim'))
    result = subprocess.run([sys.executable, '-c', 'from simulator.server import DATA_ROOT; print(DATA_ROOT)'], env=environment, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(tmp_path / 'External Sim' / 'ExplorerSim')


def test_guide_html_formats_readably_and_escapes_active_content():
    from simulator.server import guide_html
    rendered = guide_html('# Руководство\n\nТекст <script>alert(1)</script> и **важное**.\n\n[Небезопасно](javascript:alert(1))\n\n| Ввод | Действие |\n|---|---|\n| W | Вперёд |\n\n```\n<img src=x onerror=alert(1)>\n```')
    assert '<h1>Руководство</h1>' in rendered
    assert '<strong>важное</strong>' in rendered
    assert '<table>' in rendered
    assert '<script>' not in rendered
    assert '<img' not in rendered
    assert 'href="javascript:' not in rendered
    assert '&lt;script&gt;' in rendered
    assert '&lt;img' in rendered


def test_heldout_seeds_skip_actual_training_provenance():
    from simulator.server import heldout_seed_plan
    plan = heldout_seed_plan({'episodes': [{'episode': {'seed': 10000}}, {'episode': {'seed': 10001}}, {'episode': {'seed': 10003}}, {'episode': {'seed': 12}}]}, 3)
    assert plan['seeds'] == [10002, 10004, 10005]
    assert plan['heldout_independent'] is True
    assert not set(plan['seeds']).intersection(plan['training_seeds'])
    unknown = heldout_seed_plan({'episodes': [{'episode': {'seed': 10000}}, {'episode': {'source': 'simulation'}}]}, 1)
    assert unknown['seeds'] == [10001]
    assert unknown['heldout_independent'] is False
    assert unknown['unknown_training_seed_count'] == 1


def test_mac_sleep_guard_is_scoped_and_closed_on_failure(tmp_path, monkeypatch):
    import simulator.server as server
    class Guard:
        terminated = False
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return 0
    guard = Guard()
    calls = []
    def spawn(command, **kwargs):
        calls.append(command)
        return guard
    monkeypatch.setattr(server.sys, 'platform', 'darwin')
    monkeypatch.setattr(server.subprocess, 'Popen', spawn)
    runtime = Runtime(FakeEnv, tmp_path)
    runtime.launch_job('train', server.JobRequest())
    runtime.job_thread.join(timeout=2)
    assert calls[0][:3] == ['/usr/bin/caffeinate', '-i', '-w']
    assert guard.terminated
    assert runtime.jobs[-1]['status'] == 'failed'
    runtime.close()


def test_combined_base_inputs_clamp_to_the_actual_environment(tmp_path):
    class BoundedEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace
            self.action_space = SimpleNamespace(low=np.array([0, 0, 0, 0, 0, 30, -.2, -.2, -.6]),
                high=np.array([180, 180, 180, 180, 270, 170, .2, .2, .6]))
        def set_manual(self, command):
            assert np.all(command >= self.action_space.low)
            assert np.all(command <= self.action_space.high)
            super().set_manual(command)
    runtime = Runtime(BoundedEnv, tmp_path)
    runtime.start()
    try:
        runtime.control(Control(base=[.4, -.4, 1.2]))
        deadline = time.monotonic()+2
        while not any(step[6] for step in runtime.env.steps) and time.monotonic() < deadline:
            time.sleep(.01)
        assert runtime.error is None
        assert any(step[6:] == [.2, -.2, .6] for step in runtime.env.steps)
    finally:
        runtime.close()


def test_browser_focus_release_does_not_cancel_background_trial(tmp_path):
    with client(tmp_path) as connection:
        runtime = connection.app.state.runtime
        runtime.jobs.append(dict(kind='trials', status='running'))
        runtime.control(Control(base=[.1, 0., 0.], arm_velocity=[1.,0.,0.,0.,0.,0.]))
        assert connection.post('/api/stop-manual', json={}, headers=auth(connection)).status_code == 200
        assert not runtime.cancel_job.is_set()
        assert runtime.command.base == [0.,0.,0.]
        assert runtime.command.arm_velocity == [0.,0.,0.,0.,0.,0.]
        # The explicit STOP action must still cancel a moving simulation trial.
        assert connection.post('/api/stop', json={}, headers=auth(connection)).status_code == 200
        assert runtime.cancel_job.is_set()
