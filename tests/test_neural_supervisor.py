import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def module():
    spec=importlib.util.spec_from_file_location('neural_supervisor',Path(__file__).parents[1]/'bin/supervise-neural-cycle.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_surviving_pending_checkpoint_writer_blocks_relaunch(monkeypatch):
    supervisor=module()
    cycle={'tasks':[{'pending':{'stage':'train','job_id':'a'*12}}]}
    def process(pid,module,identifier):
        return SimpleNamespace(info={'pid':pid,'cmdline':['python','-m',module,'--config_path=/runs/'+identifier+'/train-config.json']})
    monkeypatch.setattr(supervisor.psutil,'process_iter',lambda fields:[
        process(10,'simulator.train_entry','a'*12),
        process(11,'simulator.train_entry','b'*12),
        process(12,'unrelated.process','a'*12)])
    assert supervisor.workers(cycle)==[10]


def test_progress_is_read_from_live_log_without_mutating_stale_job(tmp_path,monkeypatch):
    supervisor=module();monkeypatch.setattr(supervisor,'DATA',tmp_path)
    identifier='a'*12;folder=tmp_path/'runs'/identifier;folder.mkdir(parents=True)
    saved=json.dumps({'steps':119047,'steps_completed':92411})
    (folder/'job.json').write_text(saved)
    (folder/'train.log').write_text('Training: 90% 107575/119047 [17:49:01]')
    cycle={'tasks':[{'pending':{'stage':'train','job_id':identifier}}]}
    assert supervisor.learner_progress(cycle)==[{'id':identifier,'steps_completed':107575,'total_steps':119047}]
    assert (folder/'job.json').read_text()==saved
