"""Publish real worker recordings without owning or advancing any simulation.

The viewer is deliberately separate from physics and ACT. It can be attached to
an already running server without interrupting a training checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request


def capture_overview(env, folder, last_capture):
    """At most two overview frames per wall-clock second; never step physics."""
    now = time.monotonic()
    if now - last_capture >= .5:
        from PIL import Image
        path = Path(folder) / 'overview-preview.jpg'
        temporary = path.with_suffix('.tmp')
        Image.fromarray(env.render_rgb('overview')).save(temporary, format='JPEG', quality=75)
        temporary.replace(path)
        return now
    return last_capture


def sources(root, snapshot, assignments):
    root = Path(root).resolve()
    jobs = snapshot.get('jobs', [])
    kinds = ('trials', 'audit', 'validate', 'train')
    active = next((job for job in reversed(jobs) if job.get('status') == 'running' and job.get('kind') in kinds), None)
    phase = active['kind'] if active else 'idle'
    views = []
    if active and phase == 'trials':
        details = active.get('details') or {}
        attempts = details.get('active_attempts', [])
        keys = [f"{active['id']}:{attempt}" for attempt in attempts]
        for key in list(assignments):
            if key not in keys:
                del assignments[key]
        for attempt, key in zip(attempts, keys):
            if key not in assignments:
                assignments[key] = next(slot for slot in range(8) if slot not in assignments.values())
            folder = root / 'demonstrations' / f"trial-{active['id']}-{attempt-1:04d}"
            views.append(dict(slot=assignments[key], folder=folder, live=True, label=f'Попытка {attempt}', identity=key))
    elif active and phase == 'audit':
        for worker in (active.get('details') or {}).get('active_workers', []):
            folder = root / 'feedback-audits' / active['id'] / 'views' / f"{worker['index']:04d}"
            views.append(dict(slot=worker['slot'], folder=folder, live=True, label='Сравнение настроек', identity=f"{active['id']}:{worker['index']}"))
    elif active and phase == 'validate':
        folders = sorted((root / 'runs' / f"validation-{active['id']}").glob('heldout-*'))
        if folders:
            views.append(dict(slot=0, folder=folders[-1], live=True, label='Самостоятельная проверка', identity=folders[-1].name))
    # Training has no active robot. Retain four distinct recorded scenes and
    # label them as examples, rather than animating a fictional running worker.
    if not views and phase in ('train', 'idle'):
        trial = next((job for job in reversed(jobs) if job.get('kind') == 'trials' and job.get('status') == 'complete'), None)
        if trial:
            folders = sorted((root / 'demonstrations').glob(f"trial-{trial['id']}-*"))[-4:]
        else:
            folders = sorted((root / 'demonstrations').glob('trial-*'), key=lambda folder:folder.stat().st_mtime)[-4:]
        for slot, folder in enumerate(folders):
            views.append(dict(slot=slot, folder=folder, live=False, label='Сохранённый пример', identity=folder.name))
    return phase, views


def publish(root, destination, snapshot, assignments, now=None):
    now = time.time() if now is None else now
    root, destination = Path(root).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    phase, sources_list = sources(root, snapshot, assignments)
    info = dict(updated_at=now, phase=phase, views=[])
    for source in sources_list:
        folder = source['folder'].resolve()
        if not folder.is_relative_to(root):
            raise ValueError('Worker view is outside simulator storage')
        frames = sorted((folder / 'frames').glob('*.jpg'))
        choices = {'overview':folder / 'overview-preview.jpg', 'wrist':frames[-1] if frames else None}
        cameras = {}
        captured_at = 0
        for camera, path in choices.items():
            if path and path.is_file():
                if not path.resolve().is_relative_to(root):
                    raise ValueError('Worker image is outside simulator storage')
                output = destination / f"worker-{source['slot']}-{camera}.jpg"
                stamp = path.stat().st_mtime
                if not output.exists() or output.stat().st_mtime != stamp:
                    temporary = output.with_suffix('.tmp')
                    shutil.copy2(path, temporary)
                    temporary.replace(output)
                captured_at = max(captured_at, stamp)
                cameras[camera] = '/static/learning-live/' + output.name
        if cameras:
            info['views'].append(dict(slot=source['slot'], identity=source['identity'], live=source['live'],
                label=source['label'], cameras=cameras, captured_at=captured_at,
                stale=source['live'] and now-captured_at>10))
    temporary = destination / 'state.tmp'
    temporary.write_text(json.dumps(info, ensure_ascii=False))
    temporary.replace(destination / 'state.json')
    return info


def watch(port, destination):
    """One publisher per static directory, with a recoverable process lock."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    lock = destination / 'publisher.pid'
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, 'w') as handle:
        handle.write(str(os.getpid()))
    assignments = {}; failures = 0
    try:
        while failures < 30:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state?compact=true', timeout=5) as response:
                    snapshot = json.load(response)
                publish(snapshot['data_root'], destination, snapshot, assignments)
                failures = 0
            except (OSError, ValueError, KeyError):
                failures += 1
            time.sleep(.75)
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


def start(port, destination):
    threading.Thread(target=watch, args=(port, destination), name='learning-views', daemon=True).start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--static', type=Path, default=Path(__file__).parent/'static'/'learning-live')
    args = parser.parse_args()
    watch(args.port, args.static)
