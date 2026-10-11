"""Local-only interactive Explorer simulator. This module has no hardware transport."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib
import html
import re
import io
import json
import math
import os
import queue
from concurrent.futures import Future
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, field_validator

DATA_ROOT = Path(os.environ.get('EXPLORER_SIM_VOLUME', '/Volumes/ExplorerSim')).expanduser() / 'ExplorerSim'
GUIDE_FILE = Path(__file__).resolve().parent.parent / 'docs' / 'SIMULATOR-AND-TRANSFER.ru.md'
STATIC_ROOT = Path(__file__).parent / 'static'
JOINT_NAMES = ['Поворот основания', 'Подъём плеча', 'Локоть', 'Наклон кисти', 'Поворот кисти', 'Захват']


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def heldout_seed_plan(provenance: dict, count: int, start: int = 10000) -> dict:
    """Exclude known training seeds and state uncertainty for unlabelled episodes."""
    episodes = provenance.get('episodes', [])
    training = set()
    unknown = 0
    for entry in episodes:
        seed = entry.get('episode', {}).get('seed')
        if type(seed) is int and seed >= 0:
            training.add(seed)
        else:
            unknown += 1
    seeds = []
    candidate = start
    while len(seeds) < count:
        if candidate not in training:
            seeds.append(candidate)
        candidate += 1
    return {'seeds': seeds, 'training_seeds': sorted(training), 'heldout_independent': bool(episodes) and unknown == 0, 'unknown_training_seed_count': unknown}


def guide_html(source: str) -> str:
    """Render the fixed local guide with escaped text and basic Markdown blocks."""
    def inline(text):
        escaped = html.escape(text)
        escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
        def link(match):
            label, address = match.groups()
            decoded = html.unescape(address)
            if urlsplit(decoded).scheme in {'http', 'https'} or decoded.startswith('/') and not decoded.startswith('//'):
                return f'<a href="{html.escape(decoded, quote=True)}">{label}</a>'
            return label
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, escaped)
    blocks, paragraph, code_lines = [], [], []
    code_open, list_kind, table_open = False, None, False
    def flush():
        if paragraph:
            blocks.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()
    def close_list():
        nonlocal list_kind
        if list_kind:
            blocks.append(f'</{list_kind}>')
            list_kind = None
    for raw in source.splitlines():
        line = raw.strip()
        if line.startswith('```'):
            flush(); close_list()
            if code_open:
                blocks.append('<pre><code>' + html.escape('\n'.join(code_lines)) + '</code></pre>')
                code_lines.clear()
            code_open = not code_open
        elif code_open:
            code_lines.append(raw)
        elif line.startswith('|'):
            flush(); close_list()
            if not re.fullmatch(r'[\s|:-]+', line):
                cells = line.strip('|').split('|')
                cell = 'td' if table_open else 'th'
                if not table_open:
                    blocks.append('<div class="guide-table"><table>'); table_open = True
                blocks.append('<tr>' + ''.join(f'<{cell}>' + inline(value.strip()) + f'</{cell}>' for value in cells) + '</tr>')
        else:
            if table_open:
                blocks.append('</table></div>'); table_open = False
            heading = re.match(r'^(#{1,4})\s+(.+)$', line)
            item = re.match(r'^(?:([-*])|\d+\.)\s+(.+)$', line)
            if heading:
                flush(); close_list()
                level = len(heading.group(1))
                blocks.append(f'<h{level}>' + inline(heading.group(2)) + f'</h{level}>')
            elif item:
                flush()
                kind = 'ul' if item.group(1) else 'ol'
                if list_kind != kind:
                    close_list(); list_kind = kind; blocks.append(f'<{kind}>')
                blocks.append('<li>' + inline(item.group(2)) + '</li>')
            elif not line:
                flush(); close_list()
            else:
                paragraph.append(line)
    flush(); close_list()
    if table_open:
        blocks.append('</table></div>')
    if code_open:
        blocks.append('<pre><code>' + html.escape('\n'.join(code_lines)) + '</code></pre>')
    return '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Explorer · руководство</title><link rel="stylesheet" href="/static/style.css"></head><body class="guide-page"><main><a class="button" href="/">← К симулятору</a><article class="guide-article">' + ''.join(blocks) + '</article></main></body></html>'


class Control(BaseModel):
    arm_velocity: list[float] = Field(default_factory=lambda: [0.] * 6, min_length=6, max_length=6)
    base: list[float] = Field(default_factory=lambda: [0.] * 3, min_length=3, max_length=3)
    arm_target: list[float] | None = Field(default=None, min_length=6, max_length=6)

    @field_validator('arm_velocity', 'base', 'arm_target')
    @classmethod
    def finite(cls, value):
        if value is not None and not all(math.isfinite(item) for item in value):
            raise ValueError('Commands must contain finite numbers')
        return value


class Reset(BaseModel):
    seed: int = Field(default=1, ge=0, le=2**31 - 1)
    difficulty: str = Field(default='easy', pattern='^(easy|varied)$')


class StopRecording(BaseModel):
    annotation: str = Field(default='unknown', pattern='^(success|failure|unknown)$')


class JobRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=1000)
    steps: int = Field(default=3000, ge=1, le=10000000)
    batch_size: int = Field(default=16, ge=8, le=1024)
    seed: int = Field(default=100, ge=0, le=2**31 - 1)
    episode_ids: list[str] | None = Field(default=None, max_length=1000)

    @field_validator('episode_ids')
    @classmethod
    def safe_identifiers(cls, value):
        if value is not None and any(not item or len(item) > 100 or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for character in item) for item in value):
            raise ValueError('Некорректный идентификатор записи')
        return value


class Runtime:
    def __init__(self, env_factory: Callable, data_root: Path, learning_module=None, initialize=True):
        self.data_root = data_root.resolve()
        for name in ('demonstrations', 'runs', 'exports', 'logs'):
            (self.data_root / name).mkdir(parents=True, exist_ok=True)
        self.env_factory = env_factory
        self.env = env_factory() if initialize else None
        self.ready = threading.Event()
        self.initialization_error = None
        self.learning = learning_module
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.cancel_job = threading.Event()
        self.thread = None
        self.job_thread = None
        self.token = secrets.token_urlsafe(32)
        self.command = Control()
        self.command_at = 0.
        self.arm_target = np.zeros(6)
        self.pose_goal = None
        self.images = {}
        self.frame_at = 0.
        self.observation = None
        self.info = {}
        self.reward = 0.
        self.done = False
        self.paused = False
        self.error = None
        self.record_file = None
        self.record_meta = None
        self.record_frames = 0
        self.jobs = []
        self.logs = []
        self.log_file = (self.data_root / 'logs' / f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}.jsonl").open('a', encoding='utf8')
        self.latest_package = None
        self.latest_job_folder = None
        self.latest_model()
        packages = sorted((self.data_root / 'exports').glob('*.tar.gz'), key=lambda path: path.stat().st_mtime, reverse=True)
        self.latest_package = packages[0] if packages else None
        self.tasks = queue.Queue()
        self.cached_state = {}
        if self.env is not None:
            self.reset(Reset())

    def latest_model(self):
        """Restore only a finished job whose checkpoint passes bundle verification."""
        candidates = sorted((self.data_root / 'runs').glob('*/job.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        module = (self.learning or importlib.import_module('simulator.learning')) if candidates else None
        for file in candidates:
            try:
                job = json.loads(file.read_text(encoding='utf8'))
                if isinstance(job, dict) and job.get('state') in {'validated_simulation', 'trained_simulation_unvalidated'} and (file.parent / 'bundle.json').is_file():
                    module.read_bundle(file.parent)
                    self.latest_job_folder = file.parent
                    return file.parent
            except (ValueError, OSError, KeyError, TypeError, AttributeError):
                pass
        self.latest_job_folder = None
        return None

    def arm_limits(self):
        space = getattr(self.env, 'action_space', None)
        low = np.asarray(space.low[:6], float) if space is not None else np.array([0, 0, 0, 0, 0, 30.])
        high = np.asarray(space.high[:6], float) if space is not None else np.array([180, 180, 180, 180, 270, 170.])
        return low, high

    def base_limits(self):
        space = getattr(self.env, 'action_space', None)
        low = np.asarray(space.low[6:], float) if space is not None else np.array([-.2, -.2, -.6])
        high = np.asarray(space.high[6:], float) if space is not None else -low
        return low, high

    def log(self, message: str):
        with self.lock:
            self.logs.append({'time': datetime.now().astimezone().strftime('%H:%M:%S'), 'message': str(message)})
            self.logs = self.logs[-120:]
            if not self.log_file.closed:
                self.log_file.write(json.dumps(self.logs[-1], ensure_ascii=False) + '\n')
                self.log_file.flush()

    def reset(self, command: Reset):
        with self.lock:
            if self.record_file:
                self.stop_recording('unknown')
            self.observation, self.info = self.env.reset(seed=command.seed, options={'difficulty': command.difficulty})
            self.command = Control()
            self.command_at = 0.
            self.pose_goal = None
            state = self.env.state()
            self.arm_target = np.array(state.get('command', [0.] * 6)[:6], dtype=float)
            self.cached_state = state
            self.reward = 0.
            self.done = False
            self.paused = False
            self.error = None
            self.log(f'Сцена начата заново · seed {command.seed}')

    def on_owner(self, callback):
        if not self.thread or threading.current_thread() is self.thread:
            return callback()
        future = Future()
        self.tasks.put((callback, future))
        return future.result(timeout=7200)

    def physics_task(self, callback):
        try:
            return callback()
        finally:
            with self.lock:
                self.cached_state = self.env.state()
                self.arm_target = np.asarray(self.cached_state['command'][:6], dtype=float)
                self.command = Control()
                self.command_at = 0.
                self.pose_goal = None
                self.capture_images()
                self.frame_at = time.monotonic()

    def start(self):
        self.thread = threading.Thread(target=self.loop, name='explorer-simulation', daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=30):
            raise RuntimeError('Физическая модель не запустилась за 30 секунд')
        if self.initialization_error:
            raise RuntimeError(f'Не удалось создать сцену: {self.initialization_error}')

    def close(self):
        self.cancel_job.set()
        self.stop_event.set()
        self.stop_control()
        deadline = time.monotonic() + 30
        if self.thread:
            self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.job_thread:
            self.job_thread.join(timeout=max(0, deadline - time.monotonic()))
        with self.lock:
            self.stop_recording('unknown')
            self.log_file.close()
            if not self.thread:
                if self.env is not None:
                    self.env.close()

    def loop(self):
        try:
            if self.env is None:
                self.env = self.env_factory()
                self.reset(Reset())
        except Exception as exc:
            self.initialization_error = exc
            self.ready.set()
            if self.env is not None:
                self.env.close()
            return
        self.ready.set()
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            try:
                while not self.tasks.empty():
                    callback, future = self.tasks.get_nowait()
                    try:
                        future.set_result(callback())
                    except Exception as exc:
                        future.set_exception(exc)
                with self.lock:
                    if not self.paused and not self.done:
                        fresh = time.monotonic() - self.command_at < .35
                        velocity = np.array(self.command.arm_velocity if fresh else [0.] * 6)
                        if not fresh:
                            self.pose_goal = None
                        if self.pose_goal is not None:
                            self.arm_target += np.clip(self.pose_goal - self.arm_target, -30 * .05, 30 * .05)
                            if np.max(np.abs(self.pose_goal - self.arm_target)) < .05:
                                self.pose_goal = None
                        else:
                            self.arm_target += np.clip(velocity, -45, 45) * .05
                        self.arm_target = np.clip(self.arm_target, *self.arm_limits())
                        base = np.clip(self.command.base if fresh else [0.] * 3, *self.base_limits())
                        action = np.concatenate((self.arm_target, base))
                        self.env.set_manual(action)
                        sample = self.prepare_sample() if self.record_file else None
                        self.observation, self.reward, terminated, truncated, self.info = self.env.step(action)
                        self.done = bool(terminated)  # Manual scenes keep running beyond the training time budget.
                        self.cached_state = self.env.state()
                        if sample is not None:
                            self.save_sample(sample, self.cached_state['command'])
                        self.arm_target = np.asarray(self.cached_state.get('command', action)[:6], dtype=float)
                    if time.monotonic() - self.frame_at > .12:
                        self.capture_images()
                        self.frame_at = time.monotonic()
            except Exception as exc:
                with self.lock:
                    self.error = f'{type(exc).__name__}: {exc}'
                    self.paused = True
                    self.log(f'Симуляция приостановлена: {self.error}')
            next_tick += .05
            delay = next_tick - time.monotonic()
            if delay < -.25:
                next_tick = time.monotonic()
            self.stop_event.wait(max(0, delay))
        # Keep creation and cleanup of renderer on the simulation thread.
        while not self.tasks.empty():
            _, pending = self.tasks.get_nowait()
            pending.set_exception(RuntimeError('Симулятор закрывается'))
        self.env.close()

    def capture_images(self):
        for camera in ('overview', 'wrist'):
            try:
                rgb = self.env.render_rgb(camera=camera)
                output = io.BytesIO()
                Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(output, format='JPEG', quality=85)
                self.images[camera] = output.getvalue()
            except Exception as exc:
                self.log(f'Камера {camera}: {exc}')
                break
        if hasattr(self.env, 'render_depth'):
            depth = np.asarray(self.env.render_depth(), dtype=float)
            scale = np.clip(np.nan_to_num(depth, nan=3, posinf=3, neginf=0) / 3, 0, 1)
            near = np.array([248, 220, 111]); far = np.array([20, 69, 109])
            colors = (near[None, None, :] * (1 - scale[:, :, None]) + far[None, None, :] * scale[:, :, None]).astype(np.uint8)
            output = io.BytesIO()
            Image.fromarray(colors).save(output, format='JPEG', quality=85)
            self.images['depth'] = output.getvalue()

    def control(self, command: Control):
        with self.lock:
            if self.job_thread and self.job_thread.is_alive() and self.jobs[-1]['kind'] in {'trials', 'validate'}:
                return
            if command.arm_target is not None:
                self.pose_goal = np.clip(np.asarray(command.arm_target, dtype=float), *self.arm_limits())
            elif any(command.arm_velocity):
                self.pose_goal = None
            self.command = command
            self.command_at = time.monotonic()

    def stop_control(self):
        with self.lock:
            self.pose_goal = None
            self.command = Control()
            self.command_at = 0.

    def start_recording(self):
        with self.lock:
            if self.job_thread and self.job_thread.is_alive():
                raise HTTPException(409, 'Дождитесь завершения задания')
            if self.record_file:
                raise HTTPException(409, 'Запись уже идёт')
            identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(3)
            folder = self.data_root / 'demonstrations' / identifier
            folder.mkdir()
            (folder / 'frames').mkdir()
            self.record_file = (folder / 'samples.jsonl').open('w', encoding='utf8')
            self.record_meta = {'id': identifier, 'created_at': datetime.now(timezone.utc).isoformat(), 'source': 'simulation', 'folder': str(folder)}
            self.record_frames = 0
            self.log('Запись демонстрации началась')
            return self.record_meta

    def prepare_sample(self):
        sample = self.env.capture_sample()
        relative = f'frames/{self.record_frames:06d}.jpg'
        rgb = self.env.render_rgb(camera='wrist')
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(Path(self.record_meta['folder']) / relative, quality=90)
        sample.update(wrist_rgb=relative, source='simulation', action_alignment='same_row_pre_action')
        return sample

    def save_sample(self, sample, applied_action):
        sample['action'] = jsonable(applied_action)
        sample['applied_action'] = jsonable(applied_action)
        self.record_file.write(json.dumps(jsonable(sample), ensure_ascii=False) + '\n')
        self.record_frames += 1

    def stop_recording(self, annotation):
        with self.lock:
            if not self.record_file:
                return None
            self.record_file.flush()
            self.record_file.close()
            self.record_file = None
            meta = {**self.record_meta, 'frames': self.record_frames, 'operator_annotation': annotation, 'physics_verified_success': bool(self.cached_state.get('task', {}).get('success', False)), 'success': bool(self.cached_state.get('task', {}).get('success', False)), 'state': 'complete', 'fps': 20, 'action_alignment': 'same_row_pre_action', 'finished_at': datetime.now(timezone.utc).isoformat()}
            folder = Path(meta['folder'])
            try:
                from simulator.learning import CONTRACT_SHA256
                meta['sample_contract_sha256'] = CONTRACT_SHA256
            except ImportError:
                pass
            (folder / 'episode.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
            self.record_meta = None
            self.log(f'Демонстрация сохранена · {self.record_frames} кадров')
            return meta

    def snapshot(self):
        with self.lock:
            return jsonable({'state': self.cached_state, 'reward': self.reward, 'info': self.info, 'done': self.done, 'paused': self.paused, 'error': self.error, 'recording': bool(self.record_file), 'record_frames': self.record_frames, 'jobs': self.jobs[-20:], 'logs': self.logs[-40:], 'package_available': self.latest_package is not None, 'data_root': self.data_root, 'depth_available': 'depth' in self.images, 'frame_age': round(time.monotonic() - self.frame_at, 2) if self.frame_at else None})

    def episodes(self):
        output = []
        for file in sorted((self.data_root / 'demonstrations').glob('*/episode.json'), reverse=True):
            try:
                episode = json.loads(file.read_text(encoding='utf8'))
                output.append({'id': file.parent.name, 'frames': episode.get('frames', episode.get('samples', 0)), 'physics_verified_success': bool(episode.get('physics_verified_success', episode.get('success', False))), 'operator_annotation': episode.get('operator_annotation', 'unknown')})
            except (ValueError, OSError):
                pass
        return output

    def launch_job(self, kind, request):
        with self.lock:
            if self.record_file:
                raise HTTPException(409, 'Сначала закончите запись демонстрации')
            if self.job_thread and self.job_thread.is_alive():
                raise HTTPException(409, 'Дождитесь завершения текущего задания')
            self.cancel_job.clear()
            job = {'id': secrets.token_hex(6), 'kind': kind, 'status': 'running', 'progress': 0, 'message': 'Запуск…'}
            self.jobs.append(job)
            self.job_thread = threading.Thread(target=self.run_job, args=(job, request), name='explorer-learning', daemon=True)
            self.job_thread.start()
            return dict(job)

    def run_job(self, job, request):
        sleep_guard = None
        try:
            if self.stop_event.is_set() or self.cancel_job.is_set():
                raise InterruptedError('Задание остановлено')
            if sys.platform == 'darwin':
                try:
                    sleep_guard = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError as exc:
                    self.log(f'Не удалось удержать Mac от сна для задания: {exc}')
            module = self.learning or importlib.import_module('simulator.learning')
            def progress(message=None, **values):
                with self.lock:
                    if isinstance(message, dict):
                        job['details'] = jsonable({key: value for key, value in message.items() if key not in {'rgb', 'image', 'frame'}})
                        phase_names = {'exporting': 'Подготовка данных', 'training': 'Обучение ACT', 'validating': 'Проверка по записям', 'validated_simulation': 'Проверено по независимым записям', 'trained_simulation_unvalidated': 'Обучено; успешность задачи не подтверждена'}
                        phase = message.get('state', message.get('status'))
                        if phase:
                            job['message'] = phase_names.get(phase, str(phase))
                        if 'message' in message:
                            job['message'] = str(message['message'])
                        if 'progress' in message:
                            job['progress'] = message['progress']
                    elif message is not None:
                        job['message'] = str(message)
                    job.update(jsonable(values))
            def physics_progress(message=None, **values):
                if self.stop_event.is_set() or self.cancel_job.is_set():
                    raise InterruptedError('Задание остановлено')
                progress(message, **values)
                with self.lock:
                    self.cached_state = self.env.state()
                    if time.monotonic() - self.frame_at > .15:
                        self.capture_images()
                        self.frame_at = time.monotonic()
            if job['kind'] == 'trials':
                summaries = []
                for index in range(request.count):
                    if self.stop_event.is_set() or self.cancel_job.is_set():
                        raise InterruptedError('Задание остановлено')
                    folder = self.data_root / 'demonstrations' / f"trial-{job['id']}-{index:04d}"
                    progress(f'Попытка {index + 1} из {request.count}', progress=100 * index / request.count)
                    summary = self.on_owner(lambda folder=folder, index=index: self.physics_task(lambda: self.env.make_episode_trial(folder, request.seed + index, progress=physics_progress)))
                    summaries.append(summary)
                    current_state = self.on_owner(self.env.state)
                    with self.lock:
                        self.cached_state = current_state
                        self.arm_target = np.asarray(self.cached_state['command'][:6])
                        self.command = Control()
                        self.command_at = 0.
                result = {'episodes': len(summaries), 'physics_verified_successes': sum(bool(item.get('physics_verified_success', item.get('success', False))) for item in summaries), 'summaries': [{key: item.get(key) for key in ('id', 'success', 'physics_verified_success', 'frames', 'error', 'outcome')} for item in summaries]}
            elif job['kind'] == 'train':
                episodes = sorted(path.parent for path in (self.data_root / 'demonstrations').glob('*/episode.json'))
                if request.episode_ids is not None:
                    selected = set(request.episode_ids)
                    episodes = [path for path in episodes if path.name in selected]
                    if len(episodes) != len(selected):
                        raise ValueError('Выбранная запись не найдена')
                if not episodes:
                    raise ValueError('Сначала запишите показ или сгенерируйте попытки')
                successful = []
                for episode in episodes:
                    metadata = json.loads((episode / 'episode.json').read_text(encoding='utf8'))
                    if metadata.get('physics_verified_success', metadata.get('success', False)):
                        successful.append(episode)
                episodes = successful
                if not episodes:
                    raise ValueError('Нет демонстраций с подтверждённой физической доставкой. Неудачные попытки сохранены для диагностики; покажите успешную доставку или сгенерируйте новые попытки.')
                folder = self.data_root / 'runs' / job['id']
                result = module.train_job(job_folder=folder, episode_folders=episodes, steps=request.steps, batch_size=request.batch_size, seed=request.seed, progress=progress, cancelled=lambda: self.stop_event.is_set() or self.cancel_job.is_set())
                if result.get('state') == 'cancelled':
                    raise InterruptedError(result.get('error') or 'Задание остановлено')
                if result.get('state') == 'failed':
                    raise ValueError(result.get('error') or result.get('message') or 'Обучение завершилось с ошибкой')
                module.read_bundle(folder)
                self.latest_job_folder = folder
            elif job['kind'] == 'validate':
                folder = self.latest_model()
                if folder is None:
                    raise ValueError('Сначала обучите политику ACT; законченный проверенный файл модели не найден')
                bundle, _ = module.read_bundle(folder)
                model_weights_sha256 = bundle['checkpoint_sha256']['model.safetensors']
                actor = module.load_policy(folder, device='cpu')
                provenance = json.loads((folder / 'train' / 'explorer-provenance.json').read_text(encoding='utf8'))
                seed_plan = heldout_seed_plan(provenance, request.count)
                output = self.data_root / 'runs' / f"validation-{job['id']}"
                def validate_rollouts():
                    output.mkdir(parents=True, exist_ok=True)
                    results = []
                    for index, seed in enumerate(seed_plan['seeds']):
                        if self.stop_event.is_set() or self.cancel_job.is_set():
                            raise InterruptedError('Проверка остановлена')
                        progress(f'Проверка сцены {index + 1} из {request.count}', progress=100 * index / request.count)
                        summary = self.env.make_episode_trial(output / f'heldout-{index:04d}', seed=seed, policy=actor, progress=physics_progress)
                        results.append({key: summary.get(key) for key in ('id', 'success', 'error', 'outcome')})
                    report = {'validation_kind': 'closed_loop_policy', 'episodes': len(results), 'successes': sum(bool(item['success']) for item in results), 'physics_validated': False, 'real_hardware_accepted': False, 'results': results, **seed_plan}
                    (output / 'validation.json').write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding='utf8')
                    return report
                result = self.on_owner(lambda: self.physics_task(validate_rollouts))
                result.update(model_job=folder.name, model_weights_sha256=model_weights_sha256, validation_kind='closed_loop_policy')
                (folder / 'physics-validation.json').write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2), encoding='utf8')
            elif job['kind'] == 'export':
                folder = self.latest_model()
                if folder is None:
                    raise ValueError('Сначала обучите политику ACT; законченный проверенный файл модели не найден')
                transfer = importlib.import_module('simulator.transfer')
                result = transfer.export_package(folder, self.data_root / 'exports' / f"explorer-{job['id']}.tar.gz")
                candidate = Path(result.get('path', result.get('archive'))).resolve()
                if not candidate.is_relative_to(self.data_root) or not candidate.is_file():
                    raise ValueError('Пакет не найден внутри хранилища симулятора')
                self.latest_package = candidate
            (self.data_root / 'logs' / f"job-{job['id']}.json").write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2), encoding='utf8')
            with self.lock:
                job.update(status='complete', progress=100, message='Готово', result=jsonable(result))
                self.log(f"Задание {job['kind']} завершено")
        except Exception as exc:
            with self.lock:
                job.update(status='cancelled' if isinstance(exc, InterruptedError) or self.stop_event.is_set() or self.cancel_job.is_set() else 'failed', message=str(exc))
                (self.data_root / 'logs' / f"job-{job['id']}.json").write_text(json.dumps(jsonable(job), ensure_ascii=False, indent=2), encoding='utf8')
                self.log(f"Задание {job['kind']}: {exc}")
        finally:
            if sleep_guard is not None and sleep_guard.poll() is None:
                sleep_guard.terminate()
                try:
                    sleep_guard.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    sleep_guard.kill()
                    sleep_guard.wait(timeout=2)


def create_app(env_factory=None, data_root=DATA_ROOT, learning_module=None, start_loop=True):
    def default_env():
        from simulator.environment import ExplorerEnv
        return ExplorerEnv()

    @asynccontextmanager
    async def lifespan(app):
        runtime = Runtime(env_factory or default_env, Path(data_root), learning_module, initialize=not start_loop)
        app.state.runtime = runtime
        if start_loop:
            runtime.start()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title='Explorer · локальная симуляция', lifespan=lifespan)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        host = request.headers.get('host', '')
        hostname = urlsplit('http://' + host).hostname
        if hostname not in {'localhost', '127.0.0.1', '::1'}:
            return JSONResponse({'detail': 'Только локальный адрес'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != f'{request.url.scheme}://{host}':
            return JSONResponse({'detail': 'Запрос с другого сайта запрещён'}, status_code=403)
        if request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'detail': 'Запрос с другого сайта запрещён'}, status_code=403)
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            token = request.headers.get('x-explorer-sim-token', '')
            if not secrets.compare_digest(token, request.app.state.runtime.token):
                return JSONResponse({'detail': 'Перезагрузите страницу симулятора'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/session')
    def session(request: Request):
        return {'token': request.app.state.runtime.token, 'joint_names': JOINT_NAMES, 'simulation_only': True}

    @app.get('/api/state')
    def state(request: Request):
        return request.app.state.runtime.snapshot()

    @app.get('/api/episodes')
    def episodes(request: Request):
        return request.app.state.runtime.episodes()

    @app.post('/api/control')
    def control(command: Control, request: Request):
        request.app.state.runtime.control(command)
        return {'ok': True}

    @app.post('/api/floor-view')
    def floor_view(request: Request):
        runtime = request.app.state.runtime
        if runtime.job_thread and runtime.job_thread.is_alive():
            raise HTTPException(409, 'Дождитесь завершения задания')
        def pose():
            x, y, yaw = runtime.env.state()['base_pose']
            target = [x + .23 * math.cos(yaw), y + .23 * math.sin(yaw), .15]
            return runtime.env.solve_ik(target, gripper=30.)
        try:
            angles = runtime.on_owner(pose)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        runtime.control(Control(arm_target=jsonable(angles)))
        return {'ok': True}

    @app.post('/api/stop')
    def stop(request: Request):
        runtime = request.app.state.runtime
        runtime.stop_control()
        if runtime.jobs and runtime.jobs[-1]['kind'] in {'trials', 'validate'}:
            runtime.cancel_job.set()
        return {'ok': True}

    @app.post('/api/cancel')
    def cancel(request: Request):
        request.app.state.runtime.cancel_job.set()
        return {'ok': True}

    @app.post('/api/reset')
    def reset(command: Reset, request: Request):
        runtime = request.app.state.runtime
        if runtime.job_thread and runtime.job_thread.is_alive() and runtime.jobs[-1]['kind'] in {'trials', 'validate'}:
            raise HTTPException(409, 'Сначала остановите текущее задание')
        runtime.on_owner(lambda: runtime.reset(command))
        return {'ok': True}

    @app.post('/api/pause')
    def pause(request: Request):
        runtime = request.app.state.runtime
        with runtime.lock:
            if runtime.job_thread and runtime.job_thread.is_alive() and runtime.jobs[-1]['kind'] in {'trials', 'validate'}:
                raise HTTPException(409, 'Сначала остановите текущее задание')
            runtime.paused = not runtime.paused
            runtime.stop_control()
        return {'paused': runtime.paused}

    @app.post('/api/record/start')
    def record_start(request: Request):
        return request.app.state.runtime.start_recording()

    @app.post('/api/record/stop')
    def record_stop(command: StopRecording, request: Request):
        return request.app.state.runtime.stop_recording(command.annotation)

    @app.post('/api/jobs/{kind}')
    def job(kind: str, command: JobRequest, request: Request):
        if kind not in {'trials', 'train', 'validate', 'export'}:
            raise HTTPException(404, 'Неизвестное задание')
        return request.app.state.runtime.launch_job(kind, command)

    @app.get('/api/package')
    def package(request: Request):
        path = request.app.state.runtime.latest_package
        if not path:
            raise HTTPException(404, 'Сначала создайте пакет')
        return FileResponse(path, filename=path.name, media_type='application/gzip')

    @app.get('/{camera}.jpg')
    def camera_image(camera: str, request: Request):
        if camera not in {'overview', 'wrist', 'depth'}:
            raise HTTPException(404)
        content = request.app.state.runtime.images.get(camera)
        if content is None:
            raise HTTPException(503, 'Камера готовится')
        return Response(content, media_type='image/jpeg')

    @app.get('/guide')
    def guide():
        if not GUIDE_FILE.is_file():
            raise HTTPException(404, 'Руководство ещё не установлено')
        return HTMLResponse(guide_html(GUIDE_FILE.read_text(encoding='utf8')))

    @app.get('/')
    def index():
        return FileResponse(STATIC_ROOT / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC_ROOT), name='static')
    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description='Explorer simulator — local Mac only')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
