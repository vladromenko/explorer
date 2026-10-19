#!/usr/bin/env python3
"""Finite empty-scene regression of command timing; no task success claim."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Keep previous measurement; choose a new output')
    import numpy as np
    from simulator.environment import ExplorerEnv
    from simulator.learning import write_json
    env=ExplorerEnv(render=False,scene='empty');results=[]
    try:
        for legacy in [True,False]:
            env.reset(seed=0);started=float(env.data.time);drift=0.
            for step in range(200):
                action=env.command.copy()
                action[1]=90.+10.*np.sin(step*.015);action[2]=65.+5.*np.cos(step*.015)
                action[:6]=env.command[:6]+np.clip(action[:6]-env.command[:6],-2,2);action[6:]=0
                env.step(action,applied=True)
                if legacy and step%10==0:
                    for _ in range(12):
                        previous=env.command[:6].copy();env.step(np.r_[previous,0.,0.,0.])
                        drift=max(drift,float(max(abs(env.command[:6]-previous))))
            results.append(dict(legacy_extra_ticks=legacy,actor_steps=200,actor_elapsed_s=10.,
                physics_elapsed_s=float(env.data.time)-started,maximum_arm_drift_during_zero_delta_manual_tick_deg=drift,
                finite=bool(np.isfinite(env.data.qpos).all()),warnings=env.data.warning.number.tolist()))
        write_json(args.output,dict(kind='finite_control_cadence_component_probe_not_policy_success',
            results=results,physical_ready=False,simulation_qualified=False))
        if abs(results[0]['physics_elapsed_s']-22)>1e-6 or abs(results[1]['physics_elapsed_s']-10)>1e-6:
            raise ValueError('Control cadence reproduction changed')
        print(results)
    finally:env.close()


if __name__=='__main__':main()
