#!/usr/bin/env python3
"""Install the small automatic UI beside the untouched active simulator.

A versioned source snapshot and separate loopback service avoid restarting an
existing physics/learning owner. No SSH, firmware or robot services are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SUPPORT=Path.home()/'Library/Application Support/ExplorerSimulator'
DATA=Path('/Volumes/ExplorerSim/ExplorerSim')


def health():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8767/api/session',timeout=1) as response:return json.load(response)
    except OSError:return None


def install():
    if not DATA.parent.is_mount():raise ValueError('Подключите диск ExplorerSim')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Сначала сохраните проверенный код в Git')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=SUPPORT/('automatic-source-'+revision[:12]);marker=DATA/'maintenance/automatic-install.json'
    if health():
        if marker.is_file() and json.loads(marker.read_text()).get('revision')==revision:return json.loads(marker.read_text())
        if not marker.is_file():raise ValueError('Порт занят другой службой; она не будет остановлена')
        previous=json.loads(marker.read_text())
        with urllib.request.urlopen('http://127.0.0.1:8767/api/state',timeout=5) as response:status=json.load(response)
        if status.get('requests'):raise ValueError('Существуют просьбы нового контура; требуется совместимое обновление с сохранением их исполнителя')
        command=subprocess.check_output(['/bin/ps','-p',str(previous['pid']),'-o','command='],text=True)
        if '-m simulator.automatic_server' not in command:raise ValueError('PID новой службы изменился; остановка отменена')
        os.kill(previous['pid'],signal.SIGTERM)
        for _ in range(60):
            if not health():break
            time.sleep(.2)
        if health():raise ValueError('Новая служба ещё не завершилась; исходники не менялись')

    names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'--','simulator','src','config','bin','docs'],cwd=ROOT,text=True).splitlines()
    source.mkdir(exist_ok=True);hashes={}
    for name in names:
        content=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        hashes[name]=hashlib.sha256(content).hexdigest()
    (source/'source-manifest.json').write_text(json.dumps(dict(revision=revision,files=hashes)))
    runtime=SUPPORT/'runtime/bin/python'
    environment=dict(os.environ,EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes',
        PYTHONPATH=str(source)+':'+str(source/'src'),HF_HOME='/Volumes/ExplorerSim/cache/huggingface',
        HF_HUB_DISABLE_TELEMETRY='1',PYTORCH_ENABLE_MPS_FALLBACK='1',TOKENIZERS_PARALLELISM='false')
    with (SUPPORT/'automatic-service.log').open('ab') as log,(SUPPORT/'automatic-service-error.log').open('ab') as error:
        process=subprocess.Popen([str(runtime),'-m','simulator.automatic_server'],cwd=source,env=environment,
            stdout=log,stderr=error,start_new_session=True)
    for _ in range(90):
        if health():break
        if process.poll() is not None:raise ValueError('Не удалось запустить новую службу; automatic-service-error.log')
        time.sleep(.5)
    if not health():raise TimeoutError('Новая служба не ответила; прежний симулятор оставлен без изменений')
    backup=DATA/'maintenance'/('app-before-automatic-'+revision[:12]);backup.mkdir(exist_ok=True)
    app=Path.home()/'Applications/Explorer Simulator.app/Contents/MacOS/launch'
    if app.exists():shutil.copy2(app,backup/'launch')
    launcher='''#!/bin/bash
set -euo pipefail
if ! mount | /usr/bin/grep -Fq ' on /Volumes/ExplorerSim ('; then
  /usr/bin/osascript -e 'display alert "Подключите диск ExplorerSim"'
  exit 0
fi
if ! /usr/bin/curl --fail --silent --max-time 1 http://127.0.0.1:8767/api/session >/dev/null; then
  cd {source}
  export EXPLORER_SIM_MESH_DIR=/Volumes/ExplorerSim/assets/m3_reference/meshes
  export PYTHONPATH={pythonpath}
  nohup {python} -m simulator.automatic_server >>{log} 2>>{errors} </dev/null &
fi
for attempt in {{1..90}}; do
  if /usr/bin/curl --fail --silent --max-time 1 http://127.0.0.1:8767/api/session >/dev/null; then
    open 'http://127.0.0.1:8767/'
    exit 0
  fi
  sleep 0.5
done
open {errors}
'''.format(source=shlex.quote(str(source)),pythonpath=shlex.quote(str(source)+':'+str(source/'src')),
        python=shlex.quote(str(runtime)),log=shlex.quote(str(SUPPORT/'automatic-service.log')),
        errors=shlex.quote(str(SUPPORT/'automatic-service-error.log')))
    # A separately installed neural entry point owns desktop navigation.  An
    # update to the archived controller service must not replace it with 8767.
    neural_launcher=(SUPPORT/'neural-launch.json').is_file()
    if not neural_launcher:
        app.parent.mkdir(parents=True,exist_ok=True);temporary=app.with_suffix('.tmp');temporary.write_text(launcher);temporary.chmod(0o755);temporary.replace(app)
    # Old bookmarks now open the single automatic interface. Only an HTML
    # entry point is replaced; old actor source and active workers are retained.
    old=SUPPORT/'source/simulator/static/index.html'
    if old.exists() and not neural_launcher:
        shutil.copy2(old,backup/'legacy-index.html')
        redirect='<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=http://127.0.0.1:8767/"><title>Explorer · обучение</title></head><body><a href="http://127.0.0.1:8767/">Открыть автоматическое обучение</a></body></html>'
        tmp=old.with_suffix('.tmp');tmp.write_text(redirect);tmp.replace(old)
    state=dict(state='installed',revision=revision,source=str(source),pid=process.pid,files=hashes,
        backup=str(backup),legacy_service_restarted=False,legacy_actor_modified=False,url='http://127.0.0.1:8767/',at=time.time())
    marker.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:state[k] for k in ('state','revision','source','pid','url','legacy_service_restarted')},ensure_ascii=False))
    return state

if __name__=='__main__':install()
