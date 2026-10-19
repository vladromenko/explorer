#!/usr/bin/env python3
"""Finite representative recording audit; no actor or training writes."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,numpy as np,time
from simulator.learning import sha256,write_json
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'));parser.add_argument('--training',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
if args.output.exists():raise ValueError('Choose a new audit path')
r=args.root;prov=json.loads((r/'runs'/args.training/'train/explorer-provenance.json').read_text())['episodes']
from simulator.episode_lineage import recording_index,trained_recording
index=recording_index(r);cache={}
selected=[]
for height in [.08,.12,.16,.20,.30]:
 selected.extend([x for x in prov if x['episode']['domain_variation']['parameters']['bin']['height_m']==height][:2])
selected.extend(prov[-4:]);rows=[]
for item in selected:
 e=item['episode'];folder=trained_recording(item,index,cache)
 phases={};count=0;idle=0;above=[0]*6;maximum=np.zeros(6);dts=[];clock_offsets=[];last=None;student=0
 for line in (folder/'samples.jsonl').open():
  row=json.loads(line);state=np.array(row.get('observation_state',row.get('observation')));action=np.array(row.get('applied_action',row.get('action')));delta=abs(action[:6]-state[:6]);maximum=np.maximum(maximum,delta)
  phases[row.get('phase','unspecified')]=phases.get(row.get('phase','unspecified'),0)+1
  count+=1;idle+=int(max(delta)<.1 and max(abs(action[6:]))<.005);above=[n+int(v>2.00001) for n,v in zip(above,delta)]
  if row.get('training_target_valid') is False:student+=1
  if last is not None and 'simulation_time' in row:dts.append(row['simulation_time']-last)
  last=row.get('simulation_time')
  if 'actor_elapsed_s' in row and 'simulation_time' in row:clock_offsets.append(row['actor_elapsed_s']-row['simulation_time'])
 rows.append(dict(id=e['id'],frames=count,phase_counts=phases,idle_command_fraction=idle/count,
  greater_than_room_2deg_step_counts=above,max_recorded_servo_step_deg=maximum.tolist(),student_target_excluded_frames=student,
  simulation_dt_min_max=[min(dts),max(dts)] if dts else None,
  actor_vs_sim_clock_offset_min_max=[min(clock_offsets),max(clock_offsets)] if clock_offsets else None))
value=dict(at=time.time(),sampled_recordings=len(rows),selection='two earliest provenance recordings per basket height plus last four corrections; not an unbiased dataset estimate',
 results=rows,comparison='Room issued servo endpoint limit remains 2 degrees; local demonstration commands can be larger. This is a measured execution-distribution difference, not proof of the old joint refusal cause.',actor_inputs_unchanged=True)
write_json(args.output,value)
print('sampled',len(rows),'maxsteps',np.max([x['max_recorded_servo_step_deg'] for x in rows],axis=0),'idle-range',min(x['idle_command_fraction'] for x in rows),max(x['idle_command_fraction'] for x in rows))
