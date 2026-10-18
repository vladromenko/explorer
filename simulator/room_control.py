"""Local room-cycle ownership, UI state and an idle-only controller handoff."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .learning import write_json


def current(root):
    try:
        marker=json.loads((Path(root)/'room-learning/current.json').read_text());identifier=marker['id']
        if not re.fullmatch('[0-9a-f]{12}',identifier):return None
        value=json.loads((Path(root)/'room-learning'/identifier/'state.json').read_text())
        import psutil
        try:
            args=psutil.Process(value['pid']).cmdline()
            alive='simulator.room_validation' in args and identifier in args
        except (psutil.Error,KeyError):alive=False
        return dict(value,alive=alive,source=marker.get('source'))
    except (OSError,ValueError,KeyError):return None


def control(root,action):
    import fcntl
    root=Path(root)
    if not root.parent.is_mount():raise ValueError('Подключите диск ExplorerSim')
    with (root/'room-learning/queue.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _control(root,action)


def _control(root,action):
    import psutil
    root=Path(root);value=current(root)
    if value is None:raise ValueError('Нет сохранённого полного сценария')
    folder=root/'room-learning'/value['id']
    if action=='pause':
        (folder/'pause.request').touch();return dict(id=value['id'],pause_requested=True)
    if action!='resume':raise ValueError('Unknown room control')
    if value['alive']:return value
    if value['state'] not in ('operator_paused','resource_wait','running','waiting_for_local_learning','queued'):
        raise ValueError('Сначала требуется разбор сохранённой ошибки; повторная очередь не создаётся')
    for process in psutil.process_iter(['cmdline']):
        args=process.info['cmdline'] or []
        if any(str(folder) in arg for arg in args) and 'worker' in args:
            raise ValueError('Исполнитель ещё сохраняет данные')
    source=Path(value.get('source') or '')
    support=Path.home()/'Library/Application Support/ExplorerSimulator'
    if not source.resolve().is_relative_to(support.resolve()) or not re.fullmatch('room-source-[0-9a-f]{12}',source.name):
        raise ValueError('Нет проверенной установленной версии полного сценария')
    # One explicit Resume button resumes the prerequisite too, if the user's
    # earlier Pause stopped it. A waiting background owner never does this.
    from .neural_recovery import current as local_current
    local=local_current(root)
    if local and local['state']=='paused' and not local['alive']:
        prior=Path(local.get('runtime_source') or '')
        if (not prior.resolve().is_relative_to(support.resolve())
                or not re.fullmatch('desktop-source-[0-9a-f]{12}',prior.name)):
            raise ValueError('Нет сохранённой версии предыдущего ACT для продолжения')
        for pid in local.get('active_pids',[]):
            try:process=psutil.Process(pid)
            except psutil.NoSuchProcess:continue
            if any(str(root/'runs') in a for a in process.cmdline()):raise ValueError('ACT ещё сохраняет контрольную точку')
        local_folder=root/'neural-recovery'/local['id'];(local_folder/'pause.request').unlink(missing_ok=True)
        with (local_folder/'owner.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-m','simulator.neural_recovery','--root',str(root),'--id',local['id']],
                cwd=prior,env=dict(os.environ,PYTHONPATH=str(prior)+':'+str(prior/'src')),stdout=log,stderr=log,start_new_session=True)
        record={k:v for k,v in local.items() if k!='alive'};record.update(state='queued',pid=child.pid)
        write_json(local_folder/'state.json',record)
    (folder/'pause.request').unlink(missing_ok=True)
    state={k:v for k,v in value.items() if k not in ('source','alive')};state['state']='queued'
    write_json(folder/'state.json',state)
    env=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'src'))
    with (folder/'background.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-m','simulator.room_validation','--root',str(root),'--id',value['id']],
            cwd=source,env=env,stdout=log,stderr=log,start_new_session=True)
    return dict(id=value['id'],state='queued',pid=child.pid)


def publish_controller(root):
    """Called only AFTER local recovery exits and the compute lease is owned.

    Does not signal any training/physics worker. The old idle web process is
    the only eligible process; its frozen source remains intact for rollback.
    """
    import psutil
    import urllib.request
    source=Path(__file__).resolve().parents[1];manifest=source/'source-revision.json'
    if not manifest.exists():return dict(state='development_source_not_installed')
    root=Path(root);marker=root/'maintenance/room-controller.json'
    def get(route):
        with urllib.request.urlopen('http://127.0.0.1:8766'+route,timeout=30) as response:return json.load(response)
    owners=[]
    for process in psutil.process_iter(['cmdline','cwd']):
        args=process.info['cmdline'] or []
        if '-m' in args and args[args.index('-m')+1:]==['simulator.server','--port','8766']:owners.append(process)
    if len(owners)>1:raise ValueError('Несколько владельцев локального контроллера; сервисы не изменены')
    previous=owners[0] if owners else None
    if previous and Path(previous.cwd()).resolve()==source.resolve():return dict(state='installed',source=str(source))
    old=None
    if previous:
        snapshot=get('/api/state?compact=true')
        if any(j.get('status')=='running' for j in snapshot.get('jobs',[])):
            raise ValueError('Контроллер ещё владеет заданием; установка отложена')
        old=dict(args=previous.cmdline(),cwd=previous.cwd())
        previous.terminate()
        try:previous.wait(20)
        except psutil.TimeoutExpired:raise RuntimeError('Idle controller did not exit; no forced kill')
    support=Path.home()/'Library/Application Support/ExplorerSimulator'
    environment=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'src'),
        EXPLORER_SIM_MESH_DIR='/Volumes/ExplorerSim/assets/m3_reference/meshes')
    with (support/'neural-service.log').open('ab') as log,(support/'neural-service-error.log').open('ab') as error:
        child=subprocess.Popen([sys.executable,'-m','simulator.server','--port','8766'],cwd=source,
            env=environment,stdout=log,stderr=error,start_new_session=True)
    deadline=time.monotonic()+120
    while child.poll() is None and time.monotonic()<deadline:
        try:
            get('/api/session')
            result=dict(state='installed',pid=child.pid,source=str(source),previous=old,
                code_revision=json.loads(manifest.read_text())['revision'],at=time.time())
            write_json(marker,result);return result
        except OSError:time.sleep(1)
    if child.poll() is None:child.terminate();child.wait(20)
    if old:
        with (support/'neural-service-error.log').open('ab') as log:
            subprocess.Popen(old['args'],cwd=old['cwd'],stdout=log,stderr=log,start_new_session=True,
                env=dict(environment,PYTHONPATH=old['cwd']+':'+str(Path(old['cwd'])/'src')))
    raise RuntimeError('New idle controller failed; prior source restored, learning files retained')
