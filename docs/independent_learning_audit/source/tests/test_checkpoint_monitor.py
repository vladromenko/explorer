from simulator.checkpoint_monitor import eligible_step,paused,diagnostic_busy


def test_only_newer_spaced_saved_checkpoints_are_eligible():
    assert eligible_step(1000,0)==1000
    assert eligible_step(1000,1000) is None
    assert eligible_step(9000,1000) is None
    assert eligible_step(11000,1000)==11000
    assert eligible_step(1000,2000) is None


def test_operator_pause_is_never_removed(tmp_path):
    folder=tmp_path/'diagnostic';folder.mkdir()
    marker=folder/'pause.request';marker.touch()
    assert paused(tmp_path,'owner',folder)
    assert marker.exists()


def test_terminal_diagnostic_still_owns_cleanup_until_process_exits(monkeypatch):
    import psutil
    from types import SimpleNamespace
    def process(pid):
        if pid==456:raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(status=lambda:'running',cmdline=lambda:['python','-m','simulator.live_room_diagnostic','--folder','/retained/trial'])
    monkeypatch.setattr('psutil.Process',process)
    assert not diagnostic_busy(dict(marker=dict(pid=456,folder='/retained/trial'),state=dict(state='running')))
    assert diagnostic_busy(dict(marker=dict(pid=123,folder='/retained/trial'),state=dict(state='complete')))
    assert diagnostic_busy(dict(marker=dict(pid=123,folder='/retained/trial'),state=dict(state='starting_vision')))
    assert not diagnostic_busy(dict(marker=dict(pid=123,folder='/another/trial'),state=dict(state='running')))


def test_failure_before_actor_or_interface_guard_is_not_a_reason_to_wait_for_more_weights():
    from simulator.checkpoint_monitor import failure_scope
    assert failure_scope(dict(success=False,events=[],reason='basket not confirmed'))=='executor_before_policy'
    assert failure_scope(dict(success=False,events=[dict(stage='policy_pick_running')],reason='Недопустимый шаг сустава 3'))=='policy_interface_or_guard'
    assert failure_scope(dict(success=False,events=[dict(stage='policy_pick_running')],reason='not held'))=='policy_outcome'
    assert failure_scope(dict(success=True)) is None


def test_operator_pause_covers_early_evaluation_and_its_monitor(tmp_path):
    import json
    from simulator.checkpoint_monitor import pause_auxiliary
    diagnostics=tmp_path/'diagnostics';diagnostics.mkdir()
    trial=diagnostics/'live-room-example';trial.mkdir()
    monitor=diagnostics/'checkpoint-monitor-example';monitor.mkdir()
    (monitor/'state.json').write_text('{}')
    (trial/'state.json').write_text('{}')
    (diagnostics/'early-room-current.json').write_text(json.dumps(dict(folder=str(trial),pid=0)))
    assert set(pause_auxiliary(tmp_path))=={str(trial),str(monitor)}
    assert (trial/'pause.request').exists() and (monitor/'pause.request').exists()


def test_explicit_auxiliary_resume_waits_for_owner_and_retains_receipt(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import pytest
    from simulator.checkpoint_monitor import resume_auxiliary
    diagnostics=tmp_path/'diagnostics';diagnostics.mkdir()
    trial=diagnostics/'trial';trial.mkdir();(trial/'pause.request').write_text('operator')
    (diagnostics/'early-room-current.json').write_text(json.dumps(dict(folder=str(trial))))
    monkeypatch.setattr('psutil.process_iter',lambda *a:[SimpleNamespace(info=dict(cmdline=['python','-m','simulator.live_room_diagnostic']))])
    with pytest.raises(ValueError,match='still saving'):resume_auxiliary(tmp_path)
    assert (trial/'pause.request').exists()
    monkeypatch.setattr('psutil.process_iter',lambda *a:[])
    assert len(resume_auxiliary(tmp_path))==1
    assert not (trial/'pause.request').exists()
    receipt=next((tmp_path/'maintenance').glob('*.json'))
    assert json.loads(receipt.read_text())['pauses'][0]['contents']=='operator'


def test_training_worker_starts_one_observer_and_retains_upgrade(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from simulator.checkpoint_monitor import start_for_training
    root=tmp_path/'root';training=root/'runs/model';owner=root/'neural-recovery/owner'
    training.mkdir(parents=True);owner.mkdir(parents=True)
    monitor=root/'diagnostics/checkpoint-monitor-model';monitor.mkdir(parents=True)
    (monitor/'state.json').write_text(json.dumps(dict(state='needs_diagnosis',code_revision='old',failure_scope='infrastructure',last_step=1000,last_launched_at=123,results=[dict(path='retained')])))
    monkeypatch.setattr('simulator.checkpoint_monitor.source_revision',lambda:'checked')
    monkeypatch.setattr('psutil.process_iter',lambda *a:[])
    calls=[]
    monkeypatch.setattr('simulator.checkpoint_monitor.subprocess.Popen',lambda args,**kw:calls.append((args,kw)) or 'child')
    assert start_for_training(training,owner,{'EXACT':'env'},'python')=='child'
    saved=json.loads((monitor/'state.json').read_text())
    assert saved['last_step']==0 and saved['results']==[dict(path='retained')]
    assert next(monitor.glob('before-source-upgrade-*.json')).exists()
    assert calls[0][1]['env']=={'EXACT':'env'} and calls[0][0][-1]=='neural-recovery'
    monkeypatch.setattr('psutil.process_iter',lambda *a:[SimpleNamespace(info=dict(cmdline=['python','-m','simulator.checkpoint_monitor']))])
    assert start_for_training(training,owner,{},'python') is None
    assert len(calls)==1


def test_learner_phase_transition_drains_trial_without_model_failure(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    from simulator.checkpoint_monitor import run
    root=tmp_path/'root';training=root/'runs/abcdefabcdef';owner=root/'neural-recovery/123456123456'
    training.mkdir(parents=True);owner.mkdir(parents=True);(root/'diagnostics').mkdir()
    job=dict(state='training',steps_completed=2000,resumed_step=0)
    (training/'job.json').write_text(json.dumps(job))
    (owner/'state.json').write_text(json.dumps(dict(state='running',phase='train',training_model_id=training.name)))
    monkeypatch.setattr(Path,'is_mount',lambda _:True)
    monkeypatch.setattr('simulator.checkpoint_monitor.source_revision',lambda:'checked')
    monkeypatch.setattr('simulator.checkpoint_monitor.numeric_checkpoint',lambda _:training/'model/checkpoints/001000/pretrained_model')
    monkeypatch.setattr('simulator.checkpoint_monitor.current_diagnostic',lambda _:None)
    child=SimpleNamespace(pid=123,done=False)
    child.poll=lambda:0 if child.done else None
    def spawn(*a,**k):
        (training/'job.json').write_text(json.dumps(dict(job,state='validating')));return child
    monkeypatch.setattr('simulator.checkpoint_monitor.subprocess.Popen',spawn)
    monkeypatch.setattr('simulator.checkpoint_monitor.stop_owned',lambda _:setattr(child,'done',True))
    monkeypatch.setattr('simulator.checkpoint_monitor.signal.signal',lambda *a:None)
    run(root,training.name,owner.name,minimum_seconds=0)
    state=json.loads((root/('diagnostics/checkpoint-monitor-'+training.name)/'state.json').read_text())
    assert state['state']=='finished' and not state['results']
    assert len(state['interrupted_trials'])==1 and child.done
    assert state.get('failure_scope') is None
