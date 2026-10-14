#!/usr/bin/env python3
"""Compare saved physical teacher episodes on paired simulation seeds."""
import argparse
import json
from pathlib import Path

from simulator.environment import ExplorerEnv
from simulator.parallel_trials import memory_free_percent


def run(output,first_seed,count,candidate='planned_grasp_v3'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    env=ExplorerEnv()
    try:
        for seed in range(first_seed,first_seed+count):
            for profile in ('legacy_v1',candidate):
                if memory_free_percent()<25:
                    raise RuntimeError('Less than 25% free memory during teacher comparison')
                folder=output/f'{profile}-{seed}'
                saved=folder/'episode.json'
                episode=json.loads(saved.read_text()) if saved.is_file() else env.make_episode_trial(
                    folder,seed=seed,task_id='sock_basket',scene_profile='sock_variants_v1',teacher_profile=profile)
                if episode.get('seed')!=seed or episode.get('teacher_profile')!=profile or episode.get('attempt_completed') is not True:
                    raise ValueError('Saved physical episode does not match assigned scene')
                row=dict(seed=seed,profile=profile,success=episode['success'],error=episode.get('error'),
                    frames=episode['frames'],outcome=episode.get('outcome'),planning=episode.get('teacher_grasp_planning'))
                rows.append(row)
                (output/'paired.json').write_text(json.dumps(dict(results=rows,diagnostic_only=True,
                    qualification=False,physics_unchanged=True),ensure_ascii=False,indent=2))
                print(json.dumps({key:row[key] for key in ('seed','profile','success','error','frames')},ensure_ascii=False),flush=True)
    finally:env.close()
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--first-seed',type=int,required=True)
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--candidate',choices=('planned_grasp_v3','planned_release_v4'),default='planned_grasp_v3')
    args=parser.parse_args()
    if args.first_seed<0 or args.count<1 or args.count>100:raise ValueError('Invalid paired scene range')
    run(args.output,args.first_seed,args.count,args.candidate)
