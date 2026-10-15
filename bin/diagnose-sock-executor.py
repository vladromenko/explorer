#!/usr/bin/env python3
"""One finite observed simulation attempt; never connects to a real robot.

Recorded frames show actual physics, not an animation of a proposed plan.
This is a diagnostic, not training or independent acceptance qualification.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=887401004)
    parser.add_argument('--scene-profile',choices=['baseline','sock_variants_v1'],default='sock_variants_v1')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from PIL import Image
    from simulator.environment import ExplorerEnv
    from simulator.skill_workshop import trial
    env=ExplorerEnv();last_capture=-1.;last_phase=None
    recording=args.output/'recording';recording.mkdir()
    handle=(recording/'frames.jsonl').open('w')
    def progress(value):
        nonlocal last_capture,last_phase
        now=time.monotonic()
        if env.phase!=last_phase:
            print(json.dumps(dict(phase=env.phase,simulation_time=float(env.data.time))),flush=True)
            last_phase=env.phase
        if now-last_capture<.5:return
        last_capture=now;identifier=f'{env.steps:06d}'
        for camera in ('wrist','overview'):
            Image.fromarray(env.render_rgb(camera)).save(recording/f'{identifier}-{camera}.jpg',quality=85)
        handle.write(json.dumps(dict(frame=identifier,phase=env.phase,
            simulation_time=float(env.data.time),command=env.command.tolist()))+'\n');handle.flush()
    try:
        report=trial(env,args.output,'attempt','sock_basket',args.seed,args.scene_profile,progress=progress)
        print(json.dumps({key:report.get(key) for key in ('state','phase','reason','success','qualification',
            'independent_physics_outcome')},ensure_ascii=False),flush=True)
        return 0 if report.get('success') else 2
    finally:handle.close();env.close()


if __name__=='__main__':raise SystemExit(main())
