#!/usr/bin/env python3
"""After an explicit UI action, drain only Explorer owners and unmount normally."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

volume=Path('/Volumes/ExplorerSim');support=Path.home()/'Library/Application Support/ExplorerSimulator'
report=support/'last-safe-disconnect.json'
pids=[int(value) for value in sys.argv[1:]]

def notify(text):
    try:subprocess.run(['/usr/bin/osascript','-e','display notification '+json.dumps(text,ensure_ascii=False)+' with title "Explorer"'],timeout=5,capture_output=True)
    except (OSError,subprocess.SubprocessError):pass

try:
    time.sleep(2.) # Let the HTTP response reach the UI first.
    owner=volume/'ExplorerSim/skill-training/owner.lock';deadline=time.monotonic()+90
    while owner.exists():
        with owner.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        if time.monotonic()>deadline:raise RuntimeError('Вычисления ещё сохраняются. Диск не отключён.')
        time.sleep(.5)
    for pid in pids:
        result=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
        command=result.stdout.strip()
        if not any(marker in command for marker in ('-m simulator.automatic_server','-m simulator.server --port 8766')):
            continue # Never signal an unrelated/reused PID.
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for _ in range(100):
        remaining=[]
        for pid in pids:
            result=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
            if any(marker in result.stdout for marker in ('-m simulator.automatic_server','-m simulator.server --port 8766')):remaining.append(pid)
        if not remaining:break
        time.sleep(.2)
    else:raise RuntimeError('Службы ещё завершают работу. Диск не отключён.')
    from simulator.semantic_cache import stop_servers
    if stop_servers(volume/'ExplorerSim'):raise RuntimeError('Модель зрения ещё завершает работу. Диск не отключён.')
    result=subprocess.run(['/usr/sbin/diskutil','unmount',str(volume)],capture_output=True,text=True,timeout=30)
    if result.returncode:raise RuntimeError('Службы остановлены, но диск занят другой программой. Извлеките его через Finder после её закрытия.')
    value=dict(ok=True,message='ExplorerSim отключён. Теперь можно отсоединить диск.',at=time.time())
except Exception as error:value=dict(ok=False,message=str(error),at=time.time())
report.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');notify(value['message'])
