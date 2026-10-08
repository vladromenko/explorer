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

MAX_MESSAGE=4_000_000


class ArmPlannerClient:
    def __init__(self,root,worker=None,timeout=90.):
        self.root=Path(root);self.worker=Path(worker or self.root/"src/arm_planner_worker.py")
        self.timeout=timeout;self.lock=threading.Lock();self.state_lock=threading.Lock();self.connection=None

    def connect(self):
        with self.state_lock:
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

    def cancel(self):
        with self.state_lock:connection=self.connection;self.connection=None
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
        identifier=uuid.uuid4().hex
        message=json.dumps(dict(id=identifier,start_deg=list(start_deg),goal_deg=list(goal_deg),
            gripper_rad=gripper_rad,obstacles=list(obstacles)),allow_nan=False).encode()+b"\n"
        if len(message)>MAX_MESSAGE:raise ValueError("Planning request is too large")
        with self.lock:
            connection=self.connect();stream=connection["stream"]
            try:
                stream.write(message);stream.flush();line=stream.readline(MAX_MESSAGE+1)
                if not line or len(line)>MAX_MESSAGE or not line.endswith(b"\n"):raise ValueError("Planning worker connection ended")
                response=json.loads(line)
                if response.get("id")!=identifier:raise ValueError("Planning result identity mismatch")
                if response.get("error"):raise ValueError(response["error"])
                if not isinstance(response.get("result"),dict):raise ValueError("Invalid planning worker result")
                return response["result"]
            except (OSError,ValueError) as exc:
                self.cancel();stream.close();raise ValueError("MoveIt worker: "+str(exc)) from exc
