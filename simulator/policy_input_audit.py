"""Offline intervention audit of saved ACT inputs, without robot actuation.

This diagnostic measures sensitivity and action error; it never qualifies a
policy or interprets low imitation error as successful closed-loop behaviour.
"""
import argparse
import json
from pathlib import Path

from .learning import load_policy,write_json


def interventions(inputs,clock_feature):
    """Change one declared input at a time, keeping held-out labels untouched."""
    result={'baseline':inputs}
    image='observation.images.wrist'
    for name,key in [('shuffled_image',image),('shuffled_clock',clock_feature),
                     ('shuffled_state','observation.state')]:
        if key not in inputs:continue
        changed=dict(inputs);changed[key]=inputs[key].roll(max(1,len(inputs[key])//2),dims=0)
        result[name]=changed
    changed=dict(inputs);changed[image]=inputs[image]*0
    result['black_image']=changed
    return result


def audit(model,output,samples=128,device='auto'):
    from src.learning_environment import configure
    configure(model)
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from src.mobile_policy_contract import ACTOR_CLOCK
    model,output=Path(model),Path(output)
    if type(samples) is not int or samples<4:raise ValueError('At least four validation samples required')
    if output.exists():raise ValueError('Preserve existing input audit')
    split=json.loads((model/'split.json').read_text())
    if split.get('heldout_independent') is not True:raise ValueError('Independent validation episodes required')
    actor=load_policy(model,device=device)
    provenance=json.loads((model/'train/explorer-provenance.json').read_text())
    fps=float(provenance['dataset_fps']);horizon=actor.policy.config.chunk_size
    dataset=LeRobotDataset('explorer/simulator-validation',root=model/'validation',video_backend='pyav',
        delta_timestamps={'action':[i/fps for i in range(horizon)]})
    indices=np.unique(np.linspace(0,len(dataset)-1,min(samples,len(dataset)),dtype=int)).tolist()
    if len(indices)<4:raise ValueError('At least four validation samples required')
    observations=('observation.state','observation.images.wrist',ACTOR_CLOCK['feature'])
    scores={};counts=0
    for batch in torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,indices),batch_size=16,num_workers=0):
        inputs={key:batch[key] for key in observations if key in batch}
        expected=batch['action'].numpy().reshape(-1,horizon,9)
        previous=batch['observation.state'].numpy().reshape(-1,9)
        valid=(~batch['action_is_pad']).numpy().reshape(-1,horizon)
        if actor.bundle.get('action_transform'):raise ValueError('Audit requires the declared absolute-action contract')
        actuals={}
        with torch.no_grad():
            for name,values in interventions(inputs,ACTOR_CLOCK['feature']).items():
                actor.reset()
                actuals[name]=actor.post(actor.policy.predict_action_chunk(actor.pre(values))).cpu().numpy().reshape(-1,horizon,9)
        scales=np.array([10.]*6+[.8,.72,1.67])
        for name,actual in actuals.items():
            row=scores.setdefault(name,dict(error=0.,change=0.,commands=0))
            if not np.isfinite(actual).all():raise ValueError('Non-finite policy output')
            row['error']+=float((abs(actual-expected)[valid]/scales).sum())
            row['change']+=float((abs(actual-actuals['baseline'])[valid]/scales).sum())
            row['commands']+=int(valid.sum())
        counts+=len(previous)
    metrics={name:dict(normalized_mae=row['error']/(row['commands']*9),
        prediction_change=row['change']/(row['commands']*9),commands=row['commands']) for name,row in scores.items()}
    result=dict(kind='offline_input_intervention',model_id=model.name,samples=counts,
        checkpoint_sha256=actor.bundle['checkpoint_sha256'],metrics=metrics,
        actor_inputs=list(observations),independent_validation=True,final_task_test_used=False,
        simulation_qualified=False,physical_ready=False,
        limitations=['Input sensitivity and imitation error do not establish closed-loop success.',
                    'Shuffled inputs may be out of distribution; interpret alongside saved physical failures.'])
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=128);p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    a=p.parse_args();print(json.dumps(audit(a.model,a.output,a.samples,a.device),ensure_ascii=False))
