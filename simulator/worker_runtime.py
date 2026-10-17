"""Bounded recovery for native renderer hangs, outside the control policy."""
import os
from pathlib import Path
import time


class WorkerStalled(RuntimeError):
    pass


def stalled(spec, now=None, timeout=240.):
    """Use action/render progress, never the parent's continuously saved state."""
    now=time.time() if now is None else now
    path=Path(spec['folder'])/'progress.json'
    try:last=path.stat().st_mtime
    except FileNotFoundError:last=spec.get('started_at',now)
    return now-last>timeout


def terminate_pool(pool):
    """Only this executor's multiprocessing children; no global Python kills.

    A native GL call cannot observe a Python cancellation flag. Bound teardown
    before ProcessPoolExecutor.__exit__ (which otherwise waits forever).
    """
    children=list((getattr(pool,'_processes',None) or {}).values())
    owned=[p for p in children if p.pid and p.pid!=os.getpid()]
    for p in owned:
        if p.is_alive():p.terminate()
    deadline=time.monotonic()+5.
    for p in owned:p.join(timeout=max(0.,deadline-time.monotonic()))
    for p in owned:
        if p.is_alive():p.kill()
    for p in owned:p.join(timeout=2.)
    if any(p.is_alive() for p in owned):raise RuntimeError('Simulator workers did not exit; no replacement started')
