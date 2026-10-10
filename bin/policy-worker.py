#!/usr/bin/env python3
"""Persistent local ACT inference worker. File IPC only; no ROS or actuator IO."""
import json,os,sys,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['OMP_NUM_THREADS']='2'
ROOT=Path('/home/vlad/Explorer');sys.path.insert(0,str(ROOT/'src'))
from learning_environment import configure
configure(ROOT)
import cv2,torch
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors
from lerobot_bridge import write_json
from policy_preview import describe_prediction
folder=Path(sys.argv[1]);config=json.loads((folder/'config.json').read_text())
policy=ACTPolicy.from_pretrained(config['checkpoint'],local_files_only=True);policy.eval();policy.reset()
pre,post=make_pre_post_processors(policy.config,pretrained_path=config['checkpoint'])
write_json(folder/'ready.json',dict(ready=True))
for step in range(30):
    request=folder/f'{step:03d}-request.json';end=time.monotonic()+10
    while not request.exists():
        if time.monotonic()>end:raise TimeoutError('No input from supervisor')
        time.sleep(.025)
    data=json.loads(request.read_text());image=cv2.imread(str(folder/f'{step:03d}.jpg'))
    if image is None:raise ValueError('Missing image')
    rgb=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2RGB)
    batch={'observation.state':torch.tensor([data['pose']],dtype=torch.float32),
           'observation.images.wrist':torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).unsqueeze(0)}
    with torch.no_grad():action=post(policy.select_action(pre(batch))).cpu().numpy().reshape(-1)
    result=describe_prediction(data['pose'],action);result['observed_at']=data['observed_at']
    write_json(folder/f'{step:03d}-result.json',result)
