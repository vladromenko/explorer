"""Keep native MoveIt initialization out of the control/web Python process."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import uuid
import time

MAX_MESSAGE=4_000_000


class ArmPlannerClient:
    def __init__(self,root,worker=None,timeout=90.):
        self.root=Path(root);self.worker=Path(worker or self.root/"src/arm_planner_worker.py")
        self.timeout=timeout;self.lock=threading.Lock();self.state_lock=threading.Lock();self.connection=None
        self.warmup_thread=None
        self.warmup_state=dict(phase="not_started",ready=False,executed=False)
        self.generation=0

    def connect(self,expected_generation=None):
        with self.state_lock:
            if expected_generation is not None and expected_generation!=self.generation:
                raise ValueError("Planning preparation cancelled before worker creation")
            if self.connection is not None and self.connection["process"].poll() is None:return self.connection
            if self.connection is not None:
                old=self.connection;self.connection=None
                old["stream"].close();old["socket"].close()
            parent,child=socket.socketpair();parent.settimeout(self.timeout)
            folder=self.root/"data";folder.mkdir(parents=True,exist_ok=True)
            log_path=folder/"arm-planner-worker.log"
            if log_path.exists() and log_path.stat().st_size>5_000_000:log_path.replace(log_path.with_suffix(".previous.log"))
            try:
                with log_path.open("ab") as log:
                    process=subprocess.Popen([sys.executable,str(self.worker),"--fd",str(child.fileno())],
                        pass_fds=(child.fileno(),),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                self.connection={"socket":parent,"stream":parent.makefile("rwb"),"process":process}
            except Exception:parent.close();raise
            finally:child.close()
            return self.connection

    def cancel(self,reason=None):
        with self.state_lock:
            connection=self.connection;self.connection=None
            self.generation+=1
            self.warmup_state=dict(self.warmup_state,phase="failed" if reason else "cancelled",
                ready=False,executed=False,error=reason)
        if connection is not None:
            try:connection["socket"].shutdown(socket.SHUT_RDWR)
            except OSError:pass
            connection["socket"].close()
            process=connection["process"]
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    try:process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:pass
            # The reader closes its stream after shutdown; closing it here could
            # block STOP on the BufferedReader's lock during a pending read.

    def plan(self,start_deg,goal_deg,gripper_rad,obstacles=()):
        return self._request(dict(start_deg=list(start_deg),goal_deg=list(goal_deg),
            gripper_rad=gripper_rad,obstacles=list(obstacles)))

    def warmup(self,expected_generation=None):
        """Load verified geometry/plugins ahead of use; never calculate a move."""
        return self._request(dict(operation="warmup"),expected_generation)

    def warmup_async(self):
        """One bounded background initialization; normal plans reuse its worker."""
        with self.state_lock:
            if self.warmup_thread is not None and self.warmup_thread.is_alive():
                return dict(self.warmup_state)
            if self.warmup_state.get("ready") and self.connection is not None:
                if self.connection["process"].poll() is None:return dict(self.warmup_state)
            self.warmup_state=dict(phase="initializing",ready=False,executed=False,started_at=time.time())
            generation=self.generation
            def initialize():
                started=time.monotonic()
                try:
                    result=self.warmup(generation)
                    record=dict(result,phase="ready",ready=True,executed=False,
                        wall_duration_s=time.monotonic()-started,finished_at=time.time())
                except Exception as exc:
                    record=dict(phase="failed",ready=False,executed=False,error=str(exc),
                        wall_duration_s=time.monotonic()-started,finished_at=time.time())
                with self.state_lock:
                    if self.generation==generation:self.warmup_state=record
                    else:self.warmup_state["last_warmup_error"]=record.get("error")
            self.warmup_thread=threading.Thread(target=initialize,daemon=True)
            self.warmup_thread.start()
            return dict(self.warmup_state)

    def status(self):
        with self.state_lock:
            live=self.connection is not None and self.connection["process"].poll() is None
            return dict(self.warmup_state,worker_alive=live,ready=live and self.warmup_state.get("ready") is True,executed=False,
                worker_pid=self.connection["process"].pid if live else None)

    def _request(self,request,expected_generation=None):
        with self.state_lock:
            generation=self.generation if expected_generation is None else expected_generation
        identifier=uuid.uuid4().hex
        message=json.dumps(dict(request,id=identifier),allow_nan=False).encode()+b"\n"
        if len(message)>MAX_MESSAGE:raise ValueError("Planning request is too large")
        with self.lock:
            started=time.monotonic()
            with self.state_lock:
                reused=self.connection is not None and self.connection["process"].poll() is None
            connection=self.connect(generation);stream=connection["stream"]
            try:
                stream.write(message);stream.flush();line=stream.readline(MAX_MESSAGE+1)
                if not line or len(line)>MAX_MESSAGE or not line.endswith(b"\n"):raise ValueError("Planning worker connection ended")
                response=json.loads(line)
                if response.get("id")!=identifier:raise ValueError("Planning result identity mismatch")
                if response.get("error"):raise ValueError(response["error"])
                if not isinstance(response.get("result"),dict):raise ValueError("Invalid planning worker result")
                result=response["result"]
                timing=dict(result.get("planner_timing",{}),worker_reused=reused,
                    client_wall_duration_s=time.monotonic()-started)
                result["planner_timing"]=timing
                with self.state_lock:
                    if generation!=self.generation:
                        raise ValueError("Planning result cancelled before delivery")
                    self.warmup_state=dict(self.warmup_state,phase="ready",ready=True,
                        planner_timing=timing,executed=False)
                return result
            except (OSError,ValueError) as exc:
                self.cancel(reason=str(exc));stream.close();raise ValueError("MoveIt worker: "+str(exc)) from exc
