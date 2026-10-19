#!/usr/bin/env python3
"""Read-only phase-aligned command similarity; never relabels demonstrations."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def signature(rows):
    import numpy as np
    phases={}
    for row in rows:
        if row.get('training_target_valid',True):
            action=row.get('applied_action',row.get('action'))
            phases.setdefault(row.get('phase','unspecified'),[]).append(action[:6])
    chunks=[]
    for phase,values in phases.items():
        values=np.asarray(values,dtype=float)
        samples=np.linspace(0,len(values)-1,32)
        chunks.append(np.stack([np.interp(samples,np.arange(len(values)),values[:,j]) for j in range(6)],axis=1))
    return tuple(phases),np.concatenate(chunks).reshape(-1),sum(map(len,phases.values()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'))
    parser.add_argument('--training',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Retain previous audit')
    import numpy as np
    from simulator.episode_lineage import recording_index,trained_recording
    from simulator.learning import write_json
    index=recording_index(args.root);cache={};records=[]
    for split in ('train','validation'):
        provenance=json.loads((args.root/'runs'/args.training/split/'explorer-provenance.json').read_text())
        for row in provenance['episodes']:
            folder=trained_recording(row,index,cache)
            phases,features,length=signature(json.loads(line) for line in (folder/'samples.jsonl').open())
            records.append(dict(id=row['episode']['id'],split=split,phases=phases,features=features,length=length))
    closest=[];pairs=[]
    for i,left in enumerate(records):
        distances=[]
        for j,right in enumerate(records):
            if j!=i and left['phases']==right['phases']:
                distance=float(np.sqrt(np.mean((left['features']-right['features'])**2)))
                distances.append((distance,j))
                if j>i and distance<=.5 and abs(left['length']-right['length'])/max(left['length'],right['length'])<=.05:
                    pairs.append(dict(left=left['id'],right=right['id'],rms_servo_deg=distance,
                        cross_split=left['split']!=right['split']))
        if distances:
            distance,j=min(distances);closest.append(dict(id=left['id'],split=left['split'],
                nearest=records[j]['id'],nearest_split=records[j]['split'],rms_servo_deg=distance))
    value=dict(at=time.time(),training=args.training,recordings=len(records),nearest=closest,similar_pairs=pairs,
        threshold=dict(phase_aligned_servo_rms_deg=.5,relative_length_difference=.05),
        method='32 interpolated arm/gripper command samples per observed phase, only valid targets; matching phase order',
        limitations='Command similarity is not RGB/scene identity or proof of leakage; no samples removed, no success relabelled',
        files_changed=False,simulation_qualified=False)
    write_json(args.output,value);print(json.dumps(dict(recordings=len(records),similar_pairs=len(pairs),
        cross_split_pairs=sum(p['cross_split'] for p in pairs),output=str(args.output))))


if __name__=='__main__':main()
