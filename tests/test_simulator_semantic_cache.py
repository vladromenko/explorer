from concurrent.futures import Future
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cold_start_is_separate_from_inference_and_records_no_false_result(tmp_path,monkeypatch):
    from simulator import semantic_cache as cache
    clock=[0.];calls=[]
    class Child:
        def poll(self):return None
    def request(endpoint,message,**kwargs):
        calls.append((message,kwargs['timeout']))
        if clock[0]<121:raise FileNotFoundError()
        return dict(ready=True)
    monkeypatch.setattr(cache,'request',request)
    monkeypatch.setattr(cache.subprocess,'Popen',lambda *a,**k:Child())
    monkeypatch.setattr(cache.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(cache.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+61))
    assert cache.ensure_started(tmp_path).parent==tmp_path/'maintenance'
    assert all(message==dict(kind='ping') and timeout==5 for message,timeout in calls)
    assert not list(tmp_path.rglob('result.json'))


def test_zero_exit_before_readiness_is_a_startup_failure(tmp_path,monkeypatch):
    from simulator import semantic_cache as cache
    monkeypatch.setattr(cache,'request',lambda *a,**k:(_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(cache.subprocess,'Popen',lambda *a,**k:SimpleNamespace(poll=lambda:0))
    monkeypatch.setattr(cache.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='exited before readiness'):cache.ensure_started(tmp_path)


def test_vision_and_compute_failures_are_not_misdiagnosed_as_bad_robot_motion():
    from simulator.failure_repair import diagnose_report,infrastructure_failure
    for reason,kind in [('TimeoutError: Модель зрения не ответила за 60 секунд','vision_timeout'),
                        ('Stage deadline expired: find','compute_deadline'),
                        ('RuntimeError: Vision backend unavailable: reset','vision_backend_failure')]:
        report={'phase':'find','reason':reason}
        assert infrastructure_failure(report)==kind
        assert diagnose_report(report)==(kind,'identified')
    assert infrastructure_failure({'reason':'Предмет остался на полу'}) is None


def test_failed_vision_retries_same_scene_without_poisoning_learning_and_reduces_concurrency(tmp_path,monkeypatch):
    import simulator.skill_training as training
    monkeypatch.setattr('simulator.resource_budget._mac_available',lambda total:None)
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(total=24*2**30,available=16*2**30))
    calls=[]
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,spec):
            calls.append(copy.deepcopy(spec));future=Future()
            value=dict(episode_id=spec['episode_id'],seed=spec['scene_seed'],candidate=spec['candidate'],
                success=False,score=3.,phase_credit={'pick':1.},status='complete',split='train')
            if len(calls)==1:value.update(status='infrastructure_failed',infrastructure_failure='vision_timeout',reason='timeout')
            future.set_result(value);return future
    monkeypatch.setattr(training,'EpisodePool',Pool)
    trainer=training.SkillTraining(tmp_path,'a'*12);state=trainer.create(attempts=8)
    spec=trainer._spec(state,state['incumbent'],0,1234)
    results=trainer._batch(state,[spec],lambda:False,lambda s:None)
    assert len(results)==1 and results[0]['status']=='complete'
    assert state['completed_attempts']==1 and len(state['episodes'])==2
    assert calls[0]['scene_seed']==calls[1]['scene_seed']==1234
    assert calls[0]['parameters']==calls[1]['parameters'] and calls[0]['controller_program']==calls[1]['controller_program']
    assert calls[0]['episode_id']!=calls[1]['episode_id']
    assert state['runtime_recoveries'][0]['excluded_from_learning_reward'] is True
    assert state['effective_workers']==2


def test_semantic_frame_mismatch_is_refused_even_with_a_healthy_cached_worker(tmp_path,monkeypatch):
    import numpy as np
    from simulator.semantic_vision import SemanticDetector
    def infer(root,folder,**kwargs):
        (folder/'result.json').write_text(json.dumps({'frame_id':'another-frame','detections':[]}))
    monkeypatch.setattr('simulator.semantic_cache.infer',infer)
    detector=SemanticDetector(tmp_path,tmp_path/'recognition')
    with pytest.raises(ValueError,match='frame mismatch'):detector.detect(np.zeros((24,24,3),dtype=np.uint8),'sock','fresh-frame')


def test_model_cleanup_ignores_unrelated_python_and_other_storage_roots(tmp_path,monkeypatch):
    from simulator.semantic_cache import stop_servers
    stopped=[]
    class Process:
        def __init__(self,pid,args):self.pid=pid;self.info={'pid':pid,'cmdline':args}
        def terminate(self):stopped.append(self.pid)
    wanted=Process(1,['python','/checked/bin/simulator-grounding-service.py',str(tmp_path),str(tmp_path/'maintenance/x.sock')])
    unrelated=Process(2,['python','user-program.py',str(tmp_path),str(tmp_path/'maintenance/x.sock')])
    another=Process(3,['python','/checked/bin/simulator-grounding-service.py','/different','/different/maintenance/x.sock'])
    monkeypatch.setattr('psutil.process_iter',lambda attrs:[wanted,unrelated,another])
    monkeypatch.setattr('psutil.wait_procs',lambda processes,timeout:(processes,[]))
    assert stop_servers(tmp_path)==[] and stopped==[1]
