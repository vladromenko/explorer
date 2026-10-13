import importlib.util
from pathlib import Path
from unittest.mock import patch

path=Path(__file__).parents[1]/'bin/enable-simulator-parallel.py'
spec=importlib.util.spec_from_file_location('parallel_transition',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_server_owner_accepts_unquoted_macos_path_spaces_only_for_exact_app():
    python='/Users/example/Library/Application Support/ExplorerSimulator/runtime/bin/python'
    assert module.installed_server_command(python+' -m simulator.server --port 8766',python)
    for command in (python+'-other -m simulator.server',
                    python+' -m unrelated.server',
                    '/usr/bin/python -m simulator.server',
                    python+' -m simulator.server-other',
                    python+' -m "simulator.server'):
        assert not module.installed_server_command(command,python)


def snapshot(state='complete',active=False,recording=False):
    return {'night_batches':[{'id':'0a859bcd5239','state':state}],
            'jobs':[{'status':'running'}] if active else [],'recording':recording}


def test_transition_never_interrupts_active_round_or_recording(tmp_path):
    moving=snapshot()
    moving['state']={'command':[90.]*6+[.1,0.,0.]}
    for value in (snapshot('running'),snapshot(active=True),snapshot(recording=True),snapshot('paused'),moving):
        assert module.transition_ready(value,'0a859bcd5239')[0] is False
        with patch.object(module,'get',return_value=value),patch.object(module,'benchmark') as bench,patch.object(module.os,'kill') as kill:
            assert module.run('0a859bcd5239')['state']=='waiting'
        bench.assert_not_called()
        kill.assert_not_called()
    assert module.transition_ready(snapshot(),'0a859bcd5239')[0] is True


def test_lost_launch_response_recovers_without_second_queue(tmp_path):
    module.save(tmp_path/'parallel-transition-0a859bcd5239.json',{'state':'benchmarked','next_plan_id':'123456abcdef'})
    value=snapshot(active=True)
    value['night_batches'].append({'id':'123456abcdef','state':'running'})
    with patch.object(module,'DATA',tmp_path),patch.object(module,'get',return_value=value),patch.object(module,'benchmark') as bench:
        result=module.run('0a859bcd5239')
    assert result['state']=='launched' and result['recovered']
    bench.assert_not_called()
