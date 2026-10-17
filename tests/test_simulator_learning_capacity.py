"""Resource probes cannot earn skill acceptance or change trajectories."""
import copy
import json
from pathlib import Path

import pytest
from simulator.learning_capacity import assessment,projected_workers,reservation,calibrate,GIB
from simulator.skill_training import SkillTraining


def rows():
    return [dict(seed=100+i%2,candidate=i//2,status='complete',success=False,
        trajectory_sha256=str(i),phase_credit={'pick':8},reason='not yet',split='capacity_benchmark') for i in range(8)]


def test_more_workers_require_ram_and_processor_headroom():
    assert projected_workers(dict(available_bytes=4*GIB,total_bytes=24*GIB),GIB,14)==1
    assert projected_workers(dict(available_bytes=20*GIB,total_bytes=24*GIB),GIB,14)==8
    assert projected_workers(dict(available_bytes=20*GIB,total_bytes=24*GIB),GIB,6)==6
    assert reservation(4*GIB)>5*GIB


@pytest.mark.parametrize('fault',['trajectory','memory','speed','actual_concurrency','missing_trace'])
def test_only_faster_equivalent_complete_replays_admit_more_workers(fault):
    a=dict(results=rows(),wall_s=100)
    b=dict(results=rows(),wall_s=70,minimum_memory_fraction=.2,actual_peak_workers=8)
    assert assessment(a,b,8)['accepted']
    if fault=='trajectory':b['results'][0]['trajectory_sha256']='different'
    if fault=='missing_trace':b['results'][0].pop('trajectory_sha256')
    if fault=='memory':b['minimum_memory_fraction']=.14
    if fault=='speed':b['wall_s']=90
    if fault=='actual_concurrency':b['actual_peak_workers']=4
    assert not assessment(a,b,8)['accepted']


def test_calibration_uses_same_training_inputs_but_separate_uncredited_storage(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'ab'*6);state=trainer.create(attempts=8)
    state['episodes']=[dict(status='complete',peak_rss_bytes=GIB)]
    state['completed_attempts']=8
    specs=[trainer._spec(state,state['incumbent'],i//2,100+i%2) for i in range(8)]
    original=copy.deepcopy(specs);saved=copy.deepcopy(state['episodes']);seen=[]
    monkeypatch.setattr('simulator.learning_capacity.memory_budget',lambda:dict(available_bytes=22*GIB,total_bytes=24*GIB))
    monkeypatch.setattr('simulator.learning_capacity.psutil.cpu_count',lambda **kw:14)
    times=iter([0,100,100,170]);monkeypatch.setattr('simulator.learning_capacity.time.monotonic',lambda:next(times))
    def run(self,bench,inputs,*args):
        seen.append((self.folder,copy.deepcopy(inputs)))
        assert self.folder!=trainer.folder and self.pause==trainer.pause
        assert bench['options']['workers'] in (4,8)
        bench.update(minimum_memory_fraction=.2,actual_peak_workers=bench['options']['workers'])
        return rows()
    monkeypatch.setattr(SkillTraining,'_batch_once',run)
    calibrate(trainer,state,specs,lambda:False,lambda _:None)
    assert state['calibrated_workers']==state['effective_workers']==8
    assert state['completed_attempts']==8 and state['episodes']==saved and specs==original
    assert len(seen)==2
    for _,inputs in seen:
        assert all(x['split']=='capacity_benchmark' for x in inputs)
        assert [(x['candidate'],x['scene_seed'],x['parameters'],x['controller_program']) for x in inputs]==[(x['candidate'],x['scene_seed'],x['parameters'],x['controller_program']) for x in original]
    calibrate(trainer,state,specs,lambda:False,lambda _:None)
    assert len(seen)==2 # once, not a repeated benchmark loop


def test_never_benchmark_holdouts_or_override_serial_or_user_pause(tmp_path,monkeypatch):
    trainer=SkillTraining(tmp_path,'bc'*6);state=trainer.create(attempts=8,workers=1)
    state['episodes']=[dict(status='complete',peak_rss_bytes=GIB)]
    monkeypatch.setattr('simulator.learning_capacity.memory_budget',lambda:pytest.fail('must not measure or launch'))
    specs=[trainer._spec(state,state['incumbent'],i//2,100+i%2) for i in range(8)]
    calibrate(trainer,state,specs,lambda:False,lambda _:None)
    state['options']['workers']=4
    for s in specs:s['split']='validation'
    calibrate(trainer,state,specs,lambda:False,lambda _:None)
    for s in specs:s['split']='train'
    trainer.pause.touch()
    calibrate(trainer,state,specs,lambda:False,lambda _:None)
