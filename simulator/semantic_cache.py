"""Bounded, cancellable client for a shared pinned-model RGB inference process."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def endpoint(root):
    source=Path(__file__).resolve().parents[1]
    script=source/'bin/simulator-grounding-service.py';lock=source/'config/grounding-model.json'
    key=hashlib.sha256(script.read_bytes()+lock.read_bytes()+(source/'simulator/vision_backend.py').read_bytes()).hexdigest()[:12]
    return Path(root)/'maintenance'/('vision-'+key+'.sock'),script


def request(endpoint,message,*,cancelled=lambda:False,timeout=60.):
    end=time.monotonic()+timeout;data=b''
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(.25);connection.connect(str(endpoint))
        connection.sendall(json.dumps(message).encode()+b'\n')
        while b'\n' not in data:
            if cancelled():raise InterruptedError('Распознавание остановлено')
            if time.monotonic()>end:raise TimeoutError('Persistent vision worker timeout')
            try:block=connection.recv(4096)
            except socket.timeout:continue
            if not block:raise OSError('Vision worker disconnected')
            data+=block
            if len(data)>16384:raise ValueError('Oversized vision reply')
    reply=json.loads(data)
    if reply.get('error'):raise ValueError(reply['error'])
    return reply


def ensure_started(root,*,cancelled=lambda:False,startup_timeout=600.):
    """Cold imports/model hashes are startup, never a robot inference result.

    Keep the inference deadline separate. A finite startup allowance is useful
    when dataset verification competes for the external disk; record its phase
    so a quiet child cannot be mistaken for a running detector.
    """
    from .semantic_vision import vision_python
    root=Path(root);socket_path,script=endpoint(root)
    socket_path.parent.mkdir(exist_ok=True);spawned=None;started=time.monotonic()
    while True:
        if cancelled():raise InterruptedError('Распознавание остановлено')
        try:
            reply=request(socket_path,dict(kind='ping'),cancelled=cancelled,timeout=5.)
            if reply.get('ready') is not True:raise RuntimeError('Vision service did not confirm readiness')
            return socket_path
        except (FileNotFoundError,ConnectionRefusedError):
            if spawned is not None and spawned.poll() is not None:
                raise RuntimeError('Persistent vision service exited before readiness; vision-cache.log')
            if time.monotonic()-started>startup_timeout:
                raise TimeoutError('Persistent vision service did not start; see vision-startup.json and vision-cache.log')
            if spawned is None:
                with (root/'maintenance/vision-cache.log').open('ab') as log:
                    spawned=subprocess.Popen([str(vision_python(root)),str(script),str(root),str(socket_path)],
                        stdout=log,stderr=log,start_new_session=True,
                        env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2'))
            time.sleep(.1)


def infer(root,folder,*,cancelled=lambda:False):
    socket_path=ensure_started(root,cancelled=cancelled)
    return request(socket_path,dict(kind='infer',folder=str(Path(folder).resolve())),cancelled=cancelled,timeout=60.)


def stop_servers(root):
    """Drain only this storage root's read-only model workers, not any actors."""
    import psutil
    root=Path(root).resolve();processes=[]
    for process in psutil.process_iter(['pid','cmdline']):
        args=process.info['cmdline'] or []
        if len(args)<4 or not any(Path(a).name=='simulator-grounding-service.py' for a in args):continue
        if args[-2]!=str(root) or not Path(args[-1]).is_relative_to(root/'maintenance'):continue
        try:process.terminate();processes.append(process)
        except psutil.Error:continue
    _,alive=psutil.wait_procs(processes,timeout=5)
    return [p.pid for p in alive]
