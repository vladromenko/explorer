#!/usr/bin/env python3
"""Load/inference/clock diagnostic on recorded RGB; never issue a movement."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--frame',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Retain old probes; choose a new output path')
    import numpy as np
    import psutil
    from simulator.learning import load_policy,sha256,write_json
    from simulator.room_policy import issued_command
    from simulator.environment import LOW,HIGH
    memory=psutil.virtual_memory()
    if memory.available/memory.total<.25:raise ValueError('Probe waits for at least 25% available memory')
    import torch
    torch.set_num_threads(2)
    data=np.load(args.frame);image=data['rgb'];recorded_command=data['command']
    command=np.r_[recorded_command,0.,0.,0.] if len(recorded_command)==6 else recorded_command
    started=time.perf_counter();actor=load_policy(args.model,device='cpu');load_s=time.perf_counter()-started
    contexts=[]
    for clock in [0.,6.,12.,24.,43.55]:
        actor.reset();actor.elapsed_s=clock;started=time.perf_counter();prediction=actor(command,image)
        try:
            bounded=issued_command(command,np.clip(prediction,LOW,HIGH),np.zeros(6));admitted=True;reason=None
        except ValueError as error:admitted=False;reason=str(error);bounded=None
        contexts.append(dict(clock_s=clock,prediction=prediction.tolist(),raw_delta_deg=(prediction[:6]-command[:6]).tolist(),
            admitted=admitted,reason=reason,bounded_command=bounded.tolist() if bounded is not None else None,inference_s=time.perf_counter()-started))
    actor.reset();actor.elapsed_s=6.;times=[]
    for index in range(1200):
        started=time.perf_counter();prediction=actor(command,image);elapsed=time.perf_counter()-started
        times.append(dict(elapsed_s=elapsed,queued=actor.last_action_was_queued))
    def timing(queued):
        values=[row['elapsed_s'] for row in times[20:] if row['queued']==queued]
        return dict(count=len(values),p50_s=float(np.percentile(values,50)),p95_s=float(np.percentile(values,95)))
    value=dict(schema='explorer_policy_probe_v1',at=time.time(),model=str(args.model),frame_sha256=sha256(args.frame),
        model_checkpoint_sha256=actor.bundle['checkpoint_sha256'],device=str(next(actor.policy.parameters()).device),
        parameter_count=sum(p.numel() for p in actor.policy.parameters()),eval_mode=not actor.policy.training,
        load_s=load_s,inference_fresh=timing(False),cached_chunk_pop=timing(True),clock_counterfactuals=contexts,
        observation_command=command.tolist(),measured_joint_positions=False,
        base_command_source='zero mandated during room ACT; not measured' if len(recorded_command)==6 else 'recorded command',
        input_shape=list(image.shape),input_dtype=str(image.dtype),actual_motion=False,
        teacher_fallback=False,simulation_qualified=False,robot_verified=False)
    write_json(args.output,value);print(json.dumps(value))


if __name__=='__main__':main()
