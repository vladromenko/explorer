"""Inspect a learned decision. No ROS import, actuator path, or model promotion."""
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np
from lerobot_bridge import write_json,training_budget

def describe_prediction(start,action):
    start=np.asarray(start,dtype=float);action=np.asarray(action,dtype=float)
    if start.shape!=(6,) or action.shape!=(6,) or not np.isfinite(start).all() or not np.isfinite(action).all():
        raise ValueError('Модель вернула некорректные углы')
    delta=action-start
    inside=bool(np.all(action>=[0,0,0,0,0,30]) and np.all(action<=[180,180,180,180,270,180]))
    return dict(start_deg=start.tolist(),proposed_deg=action.tolist(),delta_deg=delta.tolist(),
                inside_hardware_limits=inside,within_two_degree_envelope=bool(np.max(np.abs(delta))<=2),
                collision_checked=False,executed=False,automatic_execution_allowed=False,
                joint_state_source='commanded_not_measured')

def infer(checkpoint,image,start):
    import torch
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    policy=ACTPolicy.from_pretrained(checkpoint,local_files_only=True);policy.eval();policy.reset()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
    rgb=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2RGB)
    batch={'observation.state':torch.tensor([start],dtype=torch.float32),
           'observation.images.wrist':torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).unsqueeze(0)}
    with torch.no_grad():action=post(policy.select_action(pre(batch))).cpu().numpy().reshape(-1)
    return describe_prediction(start,action)

class PolicyPreview:
    def __init__(self,root,jobs,teaching):
        self.root=Path(root);self.jobs=jobs;self.teaching=teaching;self.lock=threading.Lock()
        self.state=dict(phase='idle',executed=False)

    def status(self):return dict(self.state,busy=self.lock.locked(),automatic_execution=False)

    def start(self,task):
        if not self.lock.acquire(blocking=False):raise ValueError('Дождитесь текущего расчёта')
        try:
            training_budget(self.root)
            jobs=self.jobs.status()['jobs']
            if any(j['state'] in ('queued','exporting','training','validating') for j in jobs):
                raise ValueError('Сначала завершите обучение')
            candidates=[j for j in jobs if j['task']==task and j['state']=='validated_offline']
            if not candidates:raise ValueError('Нет проверенной модели этого навыка: сначала запишите показы и обучите ACT')
            job=candidates[-1]
            if not job.get('validation',{}).get('improves_hold_baseline'):
                raise ValueError('Последняя модель не лучше простого удержания позы; добавьте качественные показы')
            if self.teaching.active:raise ValueError('Завершите запись показа перед проверкой модели')
            pose,image=self.teaching.observation()
            folder=self.root/'data/policy-previews'/uuid.uuid4().hex;folder.mkdir(parents=True)
            cv2.imwrite(str(folder/'image.jpg'),image)
            checkpoint=self.root/'data/learning-jobs'/job['id']/'model/checkpoints/last/pretrained_model'
            write_json(folder/'request.json',dict(checkpoint=str(checkpoint),pose=pose,observed_at=time.time()))
            self.state=dict(phase='predicting',job=job['id'],task=task,executed=False)
            threading.Thread(target=self.run,args=(folder,),daemon=True).start()
            return self.status()
        except Exception:self.lock.release();raise

    def run(self,folder):
        try:
            # Own bounded process. Loading the policy never imports a motor interface.
            with (folder/'log.txt').open('w') as log:
                subprocess.run(['systemd-run','--user','--quiet','--wait','--pipe','--collect',
                    '--unit=explorer-preview-'+folder.name,'--property=MemoryMax=2500M','--property=PartOf=explorer.target',
                    '--property=Nice=15','--property=CPUWeight=10','--property=RuntimeMaxSec=80',
                    str(self.root/'.venv-learning/bin/python'),str(self.root/'bin/policy-preview.py'),str(folder)],
                    check=True,timeout=90,stdout=log,stderr=log)
            self.state.update(phase='ready',result=json.loads((folder/'result.json').read_text()))
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            self.state.update(phase='error',error=str(exc))
        finally:self.lock.release()
