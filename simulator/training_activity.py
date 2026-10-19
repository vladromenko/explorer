"""Small read-only evidence from the learner, separate from owner heartbeats."""
from datetime import datetime
import json
from pathlib import Path
import re
import socket
import threading
import time

_lock=threading.Lock()
_samples={}


def tail(path,maximum=160000):
    with Path(path).open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-maximum))
        return stream.read().decode(errors='replace')


def measured_rate(text):
    rows=re.findall(r'INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?step:(\d+) ',text)
    rows=rows[-12:]
    if len(rows)<3:return None
    first,last=rows[0],rows[-1]
    seconds=(datetime.fromisoformat(last[0])-datetime.fromisoformat(first[0])).total_seconds()
    updates=int(last[1])-int(first[1])
    return updates/seconds if seconds>0 and updates>=20 else None


def activity(root,training,now=None):
    now=time.time() if now is None else now
    folder=Path(root)/'runs'/training
    try:
        job=json.loads((folder/'job.json').read_text())
        path=folder/'train.log';text=tail(path) if path.exists() else ''
        logged=re.findall(r'step:(\d+) ',text)
        step=max(job.get('steps_completed',0),int(logged[-1]) if logged else 0)
        rate=measured_rate(text)
        age=max(0.,now-path.stat().st_mtime) if path.exists() else None
        phase=job['state']
        if phase=='training':
            phase='updating_weights' if step else 'waiting_for_first_batch' if 'Start offline training' in text else 'loading_policy' if 'Creating policy' in text else 'loading_data' if 'Creating dataset' in text else 'starting_learner'
        if phase=='updating_weights' and age is not None and age>120:phase='progress_not_confirmed'
        key=str(folder.resolve())
        with _lock:
            prior=_samples.get(key);changed=bool(prior and step>prior['step'])
            if not prior or step!=prior['step']:_samples[key]=dict(step=step,observed_at=now)
            last=_samples[key]
        checkpoints=folder/'model/checkpoints'
        names=sorted((p for p in checkpoints.glob('*') if p.is_dir() and p.name.isdecimal()),key=lambda p:int(p.name))
        stable=names[-2] if len(names)>1 else None
        return dict(run_id=training,host=socket.gethostname(),stage=phase,job_state=job['state'],
            steps_completed=step,steps_total=job.get('steps'),sampled_at=now,
            log_modified_at=path.stat().st_mtime if path.exists() else None,log_age_s=age,
            owner_heartbeat_at=job.get('at'),counter_change_observed=changed,
            counter_last_observed_change_at=last['observed_at'],updates_per_second=rate,
            remaining_weight_updates_s=(job['steps']-step)/rate if rate and step<job.get('steps',0) else None,
            eta_excludes='evaluation, checkpoint saving and subsequent correction rounds',
            stable_checkpoint=str(stable) if stable else None,
            newest_checkpoint=str(names[-1]) if names else None,
            checkpoint_rule='Second newest numeric directory only; snapshot hashes must still be verified.',
            simulation_qualified=False,robot_verified=False)
    except (OSError,ValueError,KeyError,TypeError) as error:
        return dict(run_id=training,stage='status_unavailable',sampled_at=now,error=str(error),simulation_qualified=False,robot_verified=False)
