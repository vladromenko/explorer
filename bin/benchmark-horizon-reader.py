#!/usr/bin/env python3
"""Read-only exact-tensor comparison on a saved Arrow cache; no learner."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arrow',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Retain prior measurements')
    import numpy as np
    import torch
    from datasets import Dataset
    from lerobot.datasets.dataset_reader import DatasetReader
    from lerobot.datasets.io_utils import hf_transform_to_torch
    from simulator.train_entry import projected_horizon_query
    from simulator.learning import write_json
    torch.set_num_threads(1)
    dataset=Dataset.from_file(str(args.arrow))
    counts=dict(images=0)
    def transform(batch):
        counts['images']+=sum(len(values) for key,values in batch.items() if key.startswith('observation.images.'))
        return hf_transform_to_torch(batch)
    dataset.set_transform(transform)
    reader=SimpleNamespace(hf_dataset=dataset,_meta=SimpleNamespace(video_keys=[]),_absolute_to_relative_idx=None)
    rng=np.random.default_rng(20261019)
    queries=[dict(action=list(range(offset,offset+20))) for offset in rng.integers(0,min(2000,len(dataset))-20,50)]
    results={};outputs={}
    for name,function in [('baseline',DatasetReader._query_hf_dataset),('projected',projected_horizon_query)]:
        function(reader,queries[0]);counts['images']=0;times=[];values=[]
        for query in queries:
            started=time.perf_counter();value=function(reader,query)['action'];times.append(time.perf_counter()-started)
            values.append(value)
        results[name]=dict(queries=len(times),future_rgb_decodes=counts['images'],total_s=sum(times),
            p50_s=float(np.percentile(times,50)),p95_s=float(np.percentile(times,95)))
        outputs[name]=values
    exact=all(torch.equal(a,b) for a,b in zip(outputs['baseline'],outputs['projected']))
    value=dict(kind='real_cached_action_query_benchmark',arrow=str(args.arrow),at=time.time(),
        results=results,action_tensors_bit_identical=exact,source_cache_unchanged=True,
        speedup=results['baseline']['total_s']/results['projected']['total_s'],
        applies_to='future action column queries, not complete training step throughput',
        training_performed=False,simulation_qualified=False,robot_verified=False)
    write_json(args.output,value);print(json.dumps(value))
    if not exact:raise ValueError('Projected reads changed labels; optimization rejected')


if __name__=='__main__':main()
