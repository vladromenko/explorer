"""Bounded recovery for native renderer hangs, outside the control policy."""
import os
from pathlib import Path
import time
import json
import subprocess
import sys


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
    if hasattr(pool,'terminate_owned'):return pool.terminate_owned()
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


class EpisodeFuture:
    def __init__(self,process,result,log):self.process,self.path,self.log=process,result,log
    def done(self):return self.process.poll() is not None
    def result(self):
        code=self.process.poll()
        if code is None:raise RuntimeError('Episode process has not finished')
        self.log.close()
        if code or not self.path.is_file():raise RuntimeError('Episode process failed; see '+self.log.name)
        return json.loads(self.path.read_text())


class EpisodePool:
    """Finite Popen children; no recyclable ProcessPool manager can deadlock.

    Each actor still owns its renderer and exits after exactly one episode.
    All waits in cancellation and context teardown are explicitly bounded.
    """
    def __init__(self,max_workers,command=None):
        self.maximum=max_workers;self.command=command or [sys.executable,'-m','simulator.worker_runtime'];self.children=[]
    def __enter__(self):return self
    def submit(self,spec):
        if sum(not f.done() for f in self.children)>=self.maximum:raise RuntimeError('Actor capacity exceeded')
        folder=Path(spec['folder']);coordination=folder.parent.parent/'worker-requests';coordination.mkdir(parents=True,exist_ok=True)
        request=coordination/(spec['episode_id']+'.json')
        with request.open('x') as f:json.dump(spec,f)
        log=(coordination/(spec['episode_id']+'.log')).open('wb')
        try:process=subprocess.Popen([*self.command,str(request)],stdout=log,stderr=log)
        except BaseException:log.close();raise
        future=EpisodeFuture(process,folder/'result.json',log);self.children.append(future);return future
    def terminate_owned(self):
        running=[f for f in self.children if not f.done()]
        for f in running:
            try:f.process.terminate()
            except ProcessLookupError:pass
        deadline=time.monotonic()+5.
        for f in running:
            try:f.process.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:f.process.kill()
        for f in running:f.process.wait(timeout=2.)
        for f in self.children:f.log.close()
    def __exit__(self,*args):self.terminate_owned()


if __name__=='__main__':
    from .skill_training import episode
    episode(json.loads(Path(sys.argv[1]).read_text()))
