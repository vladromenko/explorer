"""Isolated ACT feedback candidates and paired diagnostic rollouts.

No service restart, robot connection, training-data mutation or qualification.
The model still predicts its complete chunk; only the observation cadence varies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from .learning import create_bundle, load_policy, read_bundle, write_json


def recorded_failure_summary(folder):
    """Locate the last demonstrated milestone without feeding truth to the actor."""
    results=[]
    for manifest_path in sorted(Path(folder).glob('heldout-*/episode.json')):
        manifest=json.loads(manifest_path.read_text())
        rows=[json.loads(line) for line in (manifest_path.parent/'samples.jsonl').read_text().splitlines() if line.strip()]
        first_grasp=next((index for index,row in enumerate(rows) if row.get('simulated_outcome',{}).get('grasp_contact')),None)
        first_lift=next((index for index,row in enumerate(rows) if row.get('simulated_outcome',{}).get('lifted')),None)
        outcome=manifest.get('outcome',{})
        if manifest.get('error'):
            failure='execution_error'
        elif manifest.get('success') is True:
            failure=None
        elif first_grasp is None:
            failure='no_two_finger_contact'
        elif not outcome.get('lifted'):
            failure='contact_without_verified_lift'
        else:
            failure='lift_without_verified_delivery'
        results.append(dict(id=manifest_path.parent.name,success=manifest.get('success') is True,
                            failure=failure,first_grasp_frame=first_grasp,first_lift_frame=first_lift,
                            outcome=outcome))
    return dict(episodes=len(results),successes=sum(item['success'] for item in results),
                failures={kind:sum(item['failure']==kind for item in results) for kind in
                          ('execution_error','no_two_finger_contact','contact_without_verified_lift','lift_without_verified_delivery')},
                results=results,diagnostic_only=True)


def feedback_candidate(source, destination, execution_steps=5, ensemble=False):
    """Copy a finished checkpoint and invalidate acceptance of its old execution."""
    if type(execution_steps) is not int or not 1<=execution_steps<=20 or execution_steps==1 and not ensemble:
        raise ValueError('Feedback execution steps must be between 2 and 20')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    bundle, checkpoint = read_bundle(source)
    config = json.loads((checkpoint/'config.json').read_text())
    if config.get('chunk_size')!=20 or config.get('temporal_ensemble_coeff') is not None:
        raise ValueError('This candidate requires a bounded 20-step ACT queue')
    state = json.loads((source/'job.json').read_text())
    if state.get('state') not in ('trained_simulation_unvalidated','validated_simulation'):
        raise ValueError('A candidate requires finished training')
    provenance = json.loads((source/'train/explorer-provenance.json').read_text())
    destination.mkdir(parents=True, exist_ok=False)
    target = destination/'model/checkpoints/last/pretrained_model'
    target.mkdir(parents=True)
    for name in bundle['checkpoint_sha256']:
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint/name, path)
    config['n_action_steps'] = execution_steps
    config['temporal_ensemble_coeff'] = .01 if ensemble else None
    write_json(target/'config.json',config)
    # Old offline and physics reports describe different execution settings.
    # Preserve provenance, but require a fresh rollout for this candidate.
    state.update(id=destination.name,state='trained_simulation_unvalidated',
        validation={'heldout_independent':False, 'execution_changed':True},
        checkpoint_relative='model/checkpoints/last/pretrained_model',checkpoint_selection=None,
        feedback_candidate={'source_job':source.name, 'execution_steps':execution_steps,
                            'source_checkpoint_sha256':bundle['checkpoint_sha256']},
        physical_success_verified=False, automatic_execution=False, executed=False)
    write_json(destination/'job.json',state)
    write_json(destination/'train/explorer-provenance.json',provenance)
    result=create_bundle(destination,state)
    if result['checkpoint_sha256']['model.safetensors']!=bundle['checkpoint_sha256']['model.safetensors']:
        raise ValueError('Candidate changed trained weights')
    return result


def paired_audit(model_folder, seeds, destination, modes=('queue20','queue5','ensemble1')):
    """Compare feedback on the same scenes, retaining every result including failures."""
    import numpy as np
    import torch
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    from .environment import ExplorerEnv, LOW, HIGH

    if not seeds or len(set(seeds))!=len(seeds) or any(type(seed) is not int or seed<0 for seed in seeds):
        raise ValueError('Use distinct nonnegative diagnostic scene seeds')
    if not modes or any(mode not in ('queue20','queue10','queue5','ensemble1') for mode in modes):
        raise ValueError('Unknown feedback comparison mode')
    destination=Path(destination)
    if destination.exists():
        raise ValueError('Do not overwrite diagnostic evidence')
    bundle, _=read_bundle(model_folder)
    from .curriculum import task_spec
    frame_limit=4800 if task_spec(bundle['skill_id'])['rooms'] else 1600
    if bundle.get('chunk_size')!=20 or bundle.get('temporal_ensemble_coeff') is not None:
        raise ValueError('Comparison requires a bounded 20-step ACT checkpoint')
    torch.set_num_threads(2)
    report={'model':Path(model_folder).name,'purpose':'diagnostic_tuning_only',
            'qualification':False,'source_checkpoint_sha256':bundle['checkpoint_sha256'],
            'scene_profile':bundle.get('scene_profile','baseline'), 'seeds':seeds,
            'results':[], 'running':True}
    write_json(destination,report)
    actor=load_policy(model_folder,device='cpu')
    env=ExplorerEnv()
    try:
        for seed in seeds:
            for mode in modes:
                actor.policy.config.n_action_steps={'queue20':20,'queue10':10,'queue5':5,'ensemble1':1}[mode]
                actor.policy.config.temporal_ensemble_coeff=.01 if mode=='ensemble1' else None
                if mode=='ensemble1':
                    actor.policy.temporal_ensembler=ACTTemporalEnsembler(.01,20)
                actor.reset()
                env.reset(seed=seed,options={'task_id':bundle.get('skill_id','sock_basket'),
                                           'scene_profile':report['scene_profile']})
                started=time.monotonic()
                preview_folder=destination.parent/'views'/destination.stem
                preview_folder.mkdir(parents=True,exist_ok=True)
                overview_at=-1.
                grasp_frames=0
                first_grasp=None
                first_lift=None
                commands=[]
                error=None
                try:
                    for frame in range(frame_limit):
                        obs=env._observation()
                        from .worker_views import capture_overview
                        overview_at=capture_overview(env,preview_folder,overview_at)
                        command=np.clip(actor(obs['state'],obs['wrist']),LOW,HIGH)
                        _,_,_,_,info=env.step(command,applied=True)
                        commands.append(env.command[:6].copy())
                        grasp_frames+=int(info['task']['grasp_contact'])
                        if first_grasp is None and info['task']['grasp_contact']:
                            first_grasp=frame
                        if first_lift is None and info['task']['lifted']:
                            first_lift=frame
                        if env.task_success or env.last_error:
                            break
                except Exception as exc:
                    error=str(exc)
                acceleration=np.diff(np.array(commands),n=2,axis=0)/.05**2
                result={'seed':seed,'mode':mode,'success':bool(env.task_success),
                    'outcome':env.state()['task'],'frames':len(commands),
                    'first_grasp_frame':first_grasp,'first_lift_frame':first_lift,
                    'grasp_frames':grasp_frames,'error':error or env.last_error,
                    'wall_s':time.monotonic()-started,
                    'max_acceleration_deg_s2':np.max(abs(acceleration),axis=0).tolist() if len(acceleration) else None,
                    'sock_center_m':env.state()['sock_center_m']}
                report['results'].append(result)
                write_json(destination,report)
                print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        env.close()
        report['running']=False
        write_json(destination,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--modes',nargs='+',default=['queue20','queue5','ensemble1'])
    args=parser.parse_args()
    paired_audit(args.model,args.seeds,args.output,args.modes)


if __name__=='__main__':
    main()
