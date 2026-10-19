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


def test_finished_or_dead_diagnostic_does_not_block_new_eval(monkeypatch):
    monkeypatch.setattr('psutil.pid_exists',lambda pid:pid==123)
    assert not diagnostic_busy(dict(marker=dict(pid=456),state=dict(state='running')))
    assert not diagnostic_busy(dict(marker=dict(pid=123),state=dict(state='complete')))
    assert diagnostic_busy(dict(marker=dict(pid=123),state=dict(state='starting_vision')))


def test_failure_before_actor_or_interface_guard_is_not_a_reason_to_wait_for_more_weights():
    from simulator.checkpoint_monitor import failure_scope
    assert failure_scope(dict(success=False,events=[],reason='basket not confirmed'))=='executor_before_policy'
    assert failure_scope(dict(success=False,events=[dict(stage='policy_pick_running')],reason='Недопустимый шаг сустава 3'))=='policy_interface_or_guard'
    assert failure_scope(dict(success=False,events=[dict(stage='policy_pick_running')],reason='not held'))=='policy_outcome'
    assert failure_scope(dict(success=True)) is None
