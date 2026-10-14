"""Paired full-physics diagnostics; never qualification or teacher input."""
import json
import time
from pathlib import Path

from .learning import read_bundle, load_policy, write_json


def paired_result(rows, candidate_id, reference_id):
    sides={name:{row['seed']:row for row in rows if row['model_id']==name}
           for name in (candidate_id,reference_id)}
    candidate,reference=sides[candidate_id],sides[reference_id]
    if not candidate or candidate.keys()!=reference.keys() or len(rows)!=2*len(candidate):
        raise ValueError('Paired comparison requires distinct complete matching scenes')
    if any(row.get('error') for row in rows):
        raise ValueError('Execution errors cannot be ranked as model quality')
    wins=sum(candidate[seed]['success'] and not reference[seed]['success'] for seed in candidate)
    losses=sum(reference[seed]['success'] and not candidate[seed]['success'] for seed in candidate)
    return dict(candidate_model_id=candidate_id,reference_model_id=reference_id,episodes=len(candidate),
        candidate_successes=sum(row['success'] for row in candidate.values()),
        reference_successes=sum(row['success'] for row in reference.values()),
        wins=wins,losses=losses,improved=wins>losses,
        selected_model_id=candidate_id if wins>losses else reference_id,
        diagnostic_only=True,qualification=False,final_task_test_used=False)


def compare_models(candidate,reference,seeds,output,env,cancelled=None,progress=None):
    candidate,reference,output=Path(candidate),Path(reference),Path(output)
    if candidate.name==reference.name or len(seeds)!=len(set(seeds)) or not seeds:
        raise ValueError('Comparison needs different models and distinct scenes')
    bundles={}
    for folder in (candidate,reference):
        bundle,_=read_bundle(folder);bundles[folder.name]=bundle
        provenance=json.loads((folder/'train/explorer-provenance.json').read_text())
        known={row['episode'].get('seed') for row in provenance['episodes']}
        if known.intersection(seeds):raise ValueError('Diagnostic scenes overlap training data')
    a,b=bundles.values()
    if (a.get('skill_id'),a.get('scene_profile'))!=(b.get('skill_id'),b.get('scene_profile')):
        raise ValueError('Compare the same task and scene distribution')
    output.mkdir(parents=True,exist_ok=True);rows=[]
    contract=dict(seeds=seeds,checkpoint_sha256={key:value['checkpoint_sha256'] for key,value in bundles.items()})
    request_path=output/'request.json'
    if request_path.exists() and json.loads(request_path.read_text())!=contract:
        raise ValueError('Preserve the original comparison checkpoint and scene identities')
    write_json(request_path,contract)
    for folder in (reference,candidate):
        actor=None
        for index,seed in enumerate(seeds):
            if cancelled and cancelled():raise InterruptedError('Сравнение моделей приостановлено')
            episode_folder=output/folder.name/f'heldout-{index:04d}'
            saved=episode_folder/'episode.json'
            if saved.is_file():
                episode=json.loads(saved.read_text())
                if episode.get('seed')!=seed:raise ValueError('Diagnostic scene identity changed')
                if not episode.get('attempt_completed'):
                    episode_folder.rename(episode_folder.with_name(episode_folder.name+f'-interrupted-{time.time_ns()}'))
            if not saved.is_file():
                if actor is None:actor=load_policy(folder,device='cpu')
                episode=env.make_episode_trial(episode_folder,seed=seed,policy=actor,
                    task_id=a['skill_id'],scene_profile=a.get('scene_profile','baseline'),
                    progress=(lambda state: check_cancelled(cancelled)))
                check_cancelled(cancelled)
            rows.append(dict(model_id=folder.name,seed=seed,success=episode['success'],
                error=episode.get('error'),outcome=episode.get('outcome')))
            write_json(output/'progress.json',dict(results=rows,diagnostic_only=True))
            if progress:progress(dict(completed=len(rows),total=2*len(seeds),model_id=folder.name,seed=seed))
        del actor
    result=paired_result(rows,candidate.name,reference.name)
    result.update(seeds=seeds,results=rows,checkpoint_sha256={key:value['checkpoint_sha256'] for key,value in bundles.items()})
    write_json(output/'comparison.json',result)
    return result


def check_cancelled(cancelled):
    if cancelled and cancelled():raise InterruptedError('Сравнение моделей приостановлено')
