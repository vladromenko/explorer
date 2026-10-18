"""Finite physical checkpoint selection; never grant task qualification.

Intermediate trained weights compete on fresh diagnostic scenes. The separate
final test sets remain unused. No retraining, teacher commands or robot access.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .learning import read_bundle,write_json,create_bundle
from .parallel_trials import memory_free_percent,stop_children


def shortlist(selection,limit=4):
    rows=[r for r in selection['candidates'] if Path(r['checkpoint_relative']).parent.name.isdecimal()]
    if not rows:raise ValueError('No saved intermediate checkpoints')
    final=max(int(Path(r['checkpoint_relative']).parent.name) for r in rows)
    groups={}
    for row in rows:
        step=int(Path(row['checkpoint_relative']).parent.name)
        group=min(limit-1,(step-1)*limit//final)
        if group not in groups or row['normalized_mae']<groups[group]['normalized_mae']:groups[group]=row
    return [groups[k]['checkpoint_relative'] for k in sorted(groups)]


def training_seeds(model):
    result=set()
    for kind in ('train','validation'):
        path=Path(model)/kind/'explorer-provenance.json'
        if not path.is_file():raise ValueError('Missing data lineage: '+str(path))
        provenance=json.loads(path.read_text())
        for row in provenance['episodes']:
            seed=row['episode'].get('seed')
            if type(seed) is not int:raise ValueError('Unknown scene lineage')
            result.add(seed)
    return result


def persisted_seeds(value):
    """Exclude prior diagnostic/final scene identities as well as training."""
    found=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='seed' and type(item) is int:found.add(item)
            elif key in ('seeds','scene_seeds') and isinstance(item,list):
                found.update(seed for seed in item if type(seed) is int)
            else:found.update(persisted_seeds(item))
    elif isinstance(value,list):
        for item in value:found.update(persisted_seeds(item))
    return found


def stage_candidate(model,relative,target):
    model,target=Path(model),Path(target)
    source=(model/relative).resolve()
    if not source.is_relative_to(model.resolve()/'model/checkpoints'):raise ValueError('Invalid checkpoint path')
    target.mkdir(exist_ok=False,parents=True)
    destination=target/'model/checkpoints/last/pretrained_model'
    shutil.copytree(source,destination)
    for kind in ('train','validation'):
        (target/kind).mkdir()
        shutil.copy2(model/kind/'explorer-provenance.json',target/kind/'explorer-provenance.json')
    state=json.loads((model/'job.json').read_text())
    state.update(id=target.name,state='trained_simulation_unvalidated',
        checkpoint_relative='model/checkpoints/last/pretrained_model',checkpoint_selection=None,
        validation=dict(heldout_independent=False,physical_check_required=True),
        source_training_job=model.name,source_checkpoint_relative=relative)
    write_json(target/'job.json',state)
    create_bundle(target,state)
    return target


def worker(request):
    import torch
    from .environment import ExplorerEnv
    from .learning import load_policy
    torch.set_num_threads(2)
    spec=json.loads(Path(request).read_text());model=Path(spec['model']);folder=Path(spec['folder'])
    bundle,_=read_bundle(model)
    actor=load_policy(model,device='cpu');env=ExplorerEnv();rows=[]
    try:
        for i,seed in enumerate(spec['seeds']):
            summary=env.make_episode_trial(folder/f'scene-{i:02d}',seed=seed,policy=actor,
                task_id=bundle['skill_id'],scene_profile=bundle['scene_profile'])
            rows.append({key:summary.get(key) for key in ('seed','success','error','outcome','domain_variation')})
            write_json(folder/'progress.json',dict(completed=len(rows),total=len(spec['seeds']),results=rows))
        write_json(folder/'result.json',dict(model_id=model.name,checkpoint_sha256=bundle['checkpoint_sha256'],
            results=rows,successes=sum(row['success'] is True for row in rows),episodes=len(rows),
            diagnostic_only=True,simulation_qualified=False,teacher_fallback=False))
    finally:env.close()


def run(root,model_id,reference_id,identifier,first_seed,workers=2):
    root=Path(root);model=root/'runs'/model_id;reference=root/'runs'/reference_id
    if not root.parent.is_mount():raise ValueError('External volume missing')
    if type(first_seed) is not int or not 0<=first_seed<2**31-5 or workers not in (1,2,4):
        raise ValueError('Invalid finite experiment budget')
    for name in (model_id,reference_id,identifier):
        if not name or any(c not in '0123456789abcdef-' for c in name):raise ValueError('Invalid experiment identity')
    folder=root/'checkpoint-experiments'/identifier;folder.mkdir(parents=True,exist_ok=False)
    state=dict(id=identifier,state='preparing',model_id=model_id,reference_id=reference_id,results=[],
        independent_final_tests_used=False,diagnostic_only=True,simulation_qualified=False)
    def save(**values):state.update(values,at=time.time());write_json(folder/'state.json',state)
    active={}
    with (root/'skill-training/owner.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            seeds=list(range(first_seed,first_seed+5))
            excluded=training_seeds(model)|training_seeds(reference)
            for store in ('learning-cycles','night-batches'):
                for path in (root/store).glob('*.json'):
                    excluded.update(persisted_seeds(json.loads(path.read_text())))
            if excluded.intersection(seeds):raise ValueError('Diagnostic scenes overlap data or prior tests')
            selection=json.loads((model/'checkpoint-selection.json').read_text())
            selected=shortlist(selection)
            candidates=[reference]
            for index,relative in enumerate(selected):
                candidates.append(stage_candidate(model,relative,folder/'candidates'/f'{index:02d}'))
            tasks=[]
            for i,candidate in enumerate(candidates):
                output=folder/f'candidate-{i:02d}';output.mkdir()
                request=output/'request.json'
                write_json(request,dict(model=str(candidate),folder=str(output),seeds=seeds))
                tasks.append((i,request,output))
            save(state='running',seeds=seeds,selected_checkpoints=selected,total=len(tasks)*len(seeds))
            deadline=time.monotonic()+3600
            while tasks or active:
                if time.monotonic()>deadline:raise TimeoutError('Finite diagnostic budget exhausted; evidence retained')
                if (folder/'pause.request').exists():raise InterruptedError('Experiment paused; all evidence retained')
                if memory_free_percent()<10:raise InterruptedError('Resource reserve exhausted; evidence retained')
                while tasks and len(active)<workers and memory_free_percent()>=25:
                    i,request,output=tasks.pop(0);log=(output/'worker.log').open('ab')
                    child=subprocess.Popen([sys.executable,'-m','simulator.checkpoint_experiment','worker',str(request)],
                        stdout=log,stderr=log,start_new_session=True)
                    active[i]=dict(child=child,log=log,output=output,started=time.monotonic())
                for i,item in list(active.items()):
                    if item['child'].poll() is not None:
                        if item['child'].returncode:raise RuntimeError('Diagnostic worker failed: '+str(item['output']))
                        report=json.loads((item['output']/'result.json').read_text())
                        if any(row.get('error') for row in report['results']):raise RuntimeError('Execution error is not a policy score')
                        state['results'].append(report);item['log'].close();del active[i]
                    elif time.monotonic()-item['started']>1800:raise TimeoutError('Diagnostic worker exceeded 30 minutes')
                save(active_pids=[item['child'].pid for item in active.values()],
                    completed=sum(row['episodes'] for row in state['results']))
                time.sleep(1)
            baseline=next(row for row in state['results'] if row['model_id']==reference_id)
            # Ties retain the reference; selection scenes are never final tests.
            best=max(state['results'],key=lambda row:(row['successes'],row['model_id']==reference_id))
            save(state='complete',selected_model_id=best['model_id'],successes=best['successes'],
                reference_successes=baseline['successes'],improved=best['successes']>baseline['successes'])
        except Exception as error:
            save(state='paused' if isinstance(error,InterruptedError) else 'failed',reason=str(error));raise
        finally:stop_children(active)
    return state


if __name__=='__main__':
    if sys.argv[1:2]==['worker']:worker(sys.argv[2])
    else:
        parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
        parser.add_argument('--model',required=True);parser.add_argument('--reference',required=True)
        parser.add_argument('--id',required=True);parser.add_argument('--first-seed',type=int,required=True)
        args=parser.parse_args();print(json.dumps(run(args.root,args.model,args.reference,args.id,args.first_seed),ensure_ascii=False))
