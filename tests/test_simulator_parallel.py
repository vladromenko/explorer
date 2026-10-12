import json
from pathlib import Path
from unittest.mock import patch

import pytest

from simulator.parallel_trials import assess,collect
from simulator.curriculum import NightQueue
from simulator.server import JobRequest


def measurements(seconds,peak=256*1024**2):
    return dict(wall_s=seconds,episodes=[dict(seed=i,success=i%2==0,frames=800,error=None)for i in range(4)],
                measurements={str(i):dict(peak_rss_bytes=peak,trajectory_sha256=str(i))for i in range(4)})


def test_capacity_requires_speed_memory_and_identical_physics():
    serial,parallel=measurements(160),measurements(60)
    fit=assess(serial,parallel,24*1024**3,14,30)
    assert fit['fits'] and fit['speedup']>2
    parallel['measurements']['2']['trajectory_sha256']='different'
    assert not assess(serial,parallel,24*1024**3,14,30)['fits']
    parallel=measurements(140)
    assert not assess(serial,parallel,24*1024**3,14,30)['fits']
    assert not assess(serial,measurements(60),24*1024**3,14,10)['fits']
    assert not assess(serial,measurements(60,3*1024**3),24*1024**3,14,30)['fits']


def test_saved_queue_retains_worker_count_on_resume(tmp_path):
    queue=NightQueue(tmp_path)
    plan=queue.create(['sock_basket'],4,10,16,8,100,rounds=1,workers=4)
    seen=[]
    def execute(stage,task,options,cancelled):
        seen.append(options['workers'])
        return dict(model_id='test',episodes=10,successes=0,heldout_independent=True)
    queue.run(plan['id'],execute,lambda:False,lambda text:None)
    assert seen==[4]*4
    assert queue.get(plan['id'])['options']['workers']==4
    assert JobRequest().workers==1
    with pytest.raises(ValueError):JobRequest(workers=3)


def test_parallel_cancellation_stops_only_children_started_by_this_job(tmp_path):
    class Child:
        pid=12345
        def poll(self):return None
        def wait(self,timeout=None):return 0
    calls=0
    def cancelled():
        nonlocal calls
        calls+=1
        return calls>1
    with patch('simulator.parallel_trials.subprocess.Popen',return_value=Child()) as launch, \
         patch('simulator.parallel_trials.os.killpg') as stop:
        with pytest.raises(InterruptedError):
            collect(tmp_path,'cancel-test',4,100,workers=4,cancelled=cancelled)
    assert launch.call_count==4
    assert stop.call_count==4
    assert not list((tmp_path/'demonstrations').glob('*/episode.json'))


def test_recovered_failures_count_and_are_not_repeated(tmp_path):
    recovered={i:dict(seed=100+i,success=False,attempt_completed=True)for i in range(4)}
    with patch('simulator.parallel_trials.subprocess.Popen') as launch:
        result=collect(tmp_path,'recovered',4,100,workers=4,recovered=recovered)
    assert launch.call_count==0
    assert len(result['episodes'])==4
    assert not any(e['success']for e in result['episodes'])
