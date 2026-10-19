"""Future action queries must not decode future images or change tensors."""
from types import SimpleNamespace
import numpy as np
import torch
from datasets import Dataset
from simulator.train_entry import projected_horizon_query


def test_projected_horizon_preserves_actions_and_omits_unrequested_images():
    from lerobot.datasets.dataset_reader import DatasetReader
    calls=[]
    dataset=Dataset.from_dict(dict(action=[[float(i),i+.5] for i in range(30)],image=['fixture']*30))
    def transform(batch):
        calls.append(list(batch))
        return {key:[torch.tensor(row) if key=='action' else row for row in value] for key,value in batch.items()}
    dataset.set_transform(transform)
    reader=SimpleNamespace(hf_dataset=dataset,_meta=SimpleNamespace(video_keys=[]),_absolute_to_relative_idx=None)
    query=dict(action=list(range(1,21)))
    original=DatasetReader._query_hf_dataset(reader,query)
    assert calls==[['action','image']]
    calls.clear();optimized=projected_horizon_query(reader,query)
    torch.testing.assert_close(original['action'],optimized['action'],rtol=0,atol=0)
    assert calls==[['action']]
    reader._absolute_to_relative_idx={100+i:i for i in range(30)}
    remapped=projected_horizon_query(reader,dict(action=list(range(101,121))))
    torch.testing.assert_close(original['action'],remapped['action'],rtol=0,atol=0)


def test_projection_is_applied_in_spawned_training_process(tmp_path):
    import os
    import subprocess
    import sys
    code='''import os
os.environ['EXPLORER_PROJECTED_HORIZON_READ']='1'
import simulator.train_entry
from lerobot.datasets.dataset_reader import DatasetReader
assert DatasetReader._query_hf_dataset.__module__=='simulator.train_entry'
'''
    subprocess.run([sys.executable,'-c',code],check=True,env=dict(os.environ),timeout=60,capture_output=True)


def test_measurement_observes_real_updates_and_restores_hooks(tmp_path):
    import json
    import torch
    from types import SimpleNamespace
    from simulator.train_entry import measured_updates
    policy=torch.nn.Linear(2,1);optimizer=torch.optim.SGD(policy.parameters(),lr=.1)
    accelerator=SimpleNamespace(backward=lambda loss:loss.backward())
    old_forward=policy.forward;old_backward=accelerator.backward;old_step=optimizer.step
    def update(metrics,model,batch,opt,clip,acc):
        loss=model.forward(batch['state']).square().mean();acc.backward(loss);opt.step();opt.zero_grad();return metrics,loss.item()
    before=policy.weight.detach().clone();path=tmp_path/'measurement.json'
    measured=measured_updates(update,path,warmup=1,samples=2)
    for _ in range(4):measured({},policy,{'state':torch.ones(2,2)},optimizer,1.,accelerator)
    proof=json.loads(path.read_text())
    assert len(proof['samples'])==2 and proof['samples'][0]['batch']['state']['device']=='cpu'
    assert all(row['forward_s']>=0 and row['backward_s']>=0 for row in proof['samples'])
    assert not torch.equal(before,policy.weight)
    assert accelerator.backward is old_backward and optimizer.step==old_step
    assert policy.forward==old_forward
    assert not policy._forward_hooks and not policy._forward_pre_hooks
