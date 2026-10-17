import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

from simulator.worker_runtime import WorkerStalled,stalled,terminate_pool


def sleep_child():
    time.sleep(60)


def test_native_hang_teardown_terminates_only_executor_children():
    context=multiprocessing.get_context('spawn')
    owned=context.Process(target=sleep_child);unrelated=context.Process(target=sleep_child)
    owned.start();unrelated.start()
    try:
        started=time.monotonic()
        terminate_pool(SimpleNamespace(_processes={owned.pid:owned}))
        assert not owned.is_alive() and unrelated.is_alive()
        assert time.monotonic()-started<8.
    finally:
        for p in (owned,unrelated):
            if p.is_alive():p.terminate()
            p.join(timeout=5)


def test_watchdog_does_not_confuse_parent_state_updates_with_worker_progress(tmp_path):
    spec=dict(folder=str(tmp_path/'episode'),started_at=100.)
    assert not stalled(spec,now=339.)
    assert stalled(spec,now=341.)
    folder=Path(spec['folder']);folder.mkdir();progress=folder/'progress.json';progress.write_text('{}')
    os.utime(progress,(330.,330.))
    assert not stalled(spec,now=341.)
    assert stalled(spec,now=571.)


def test_native_failure_retries_exact_policy_and_scene_without_reward(tmp_path,monkeypatch):
    from simulator.skill_training import SkillTraining
    trainer=SkillTraining(tmp_path,'ab'*6);state=trainer.create(attempts=8)
    first=trainer._spec(state,state['incumbent'],0,906160004)
    second=trainer._spec(state,state['incumbent'],1,906160014)
    old_id=second['episode_id'];seen=[]
    def batch(saved,specs,*args):
        seen.append([dict(s) for s in specs])
        if len(seen)==1:
            saved['episodes'].append(dict(episode_id=first['episode_id'],status='complete',score=3.,success=False))
            raise WorkerStalled('native cache lock')
        rows=[dict(episode_id=s['episode_id'],status='complete',score=3.,success=False) for s in specs]
        saved['episodes']+=rows;return rows
    monkeypatch.setattr(trainer,'_batch_once',batch)
    rows=trainer._batch(state,[first,second],lambda:False,lambda _:None)
    assert len(rows)==2 and len(seen[1])==1
    retry=seen[1][0]
    assert retry['episode_id']!=old_id
    for key in ('parameters','scene_seed','controller_program','controller_sha256'):
        assert retry.get(key)==seen[0][1].get(key)
    assert state['effective_workers']==1
    failed=[r for r in state['episodes'] if r['status']=='infrastructure_failed']
    assert len(failed)==1 and failed[0]['episode_id']==old_id
    assert not json.loads(Path(failed[0]['report']).read_text())['physics_success_evaluated']
    assert all(r['status']=='complete' for r in rows)


def test_worker_stall_is_reported_as_runtime_problem_not_grasp_failure(tmp_path):
    from simulator.automatic_server import summarized
    from simulator.failure_repair import infrastructure_failure
    root=tmp_path/'skill-training'/('a'*12)/'episodes'/('b'*12);root.mkdir(parents=True)
    os.utime(root,(time.time()-500,time.time()-500))
    row=dict(id='a'*12,state='running',task_id='sock_basket',episodes=[],completed_attempts=1,
        options=dict(attempts=8,hours=1),active_workers=[dict(slot=0,episode_id='b'*12,started_at=time.time()-500)])
    assert summarized(row,tmp_path)['active_workers'][0]['unresponsive'] is True
    assert infrastructure_failure({'reason':'Renderer worker stalled: no frames'})=='renderer_stall'
