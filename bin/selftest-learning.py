#!/usr/bin/env python3
"""Synthetic installation test only. Never creates real demonstrations or moves hardware."""
import importlib.util,json,sys,time,subprocess
from pathlib import Path
import cv2,numpy as np
root=Path('/home/vlad/Explorer');sys.path.insert(0,str(root/'src'))
spec=importlib.util.spec_from_file_location('learning_run',root/'bin/learning-run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
folder=root/'data/learning-selftest'/str(int(time.time()));folder.mkdir(parents=True)
episodes=[]
for i in range(3):
 p=folder/'source'/str(i);p.mkdir(parents=True)
 image=np.full((240,320,3),50+i*40,dtype=np.uint8)
 cv2.imwrite(str(p/'frame.jpg'),image)
 episodes.append(dict(id=str(i),name='synthetic installer check',steps=[dict(before_image='frame.jpg',start_deg=[90,90,40,40,90,90],goal_deg=[92,90,40,40,90,90]) for _ in range(3)]))
m.export_dataset(episodes[:2],folder/'source',folder/'train','explorer/train')
m.export_dataset(episodes[2:],folder/'source',folder/'validation','explorer/validation')
config=dict(dataset=dict(repo_id='explorer/train',root=str(folder/'train'),video_backend='pyav'),
 policy=dict(type='act',device='cuda',push_to_hub=False,chunk_size=1,n_action_steps=1,
 pretrained_backbone_weights=None,dim_model=64,n_heads=4,dim_feedforward=128,n_encoder_layers=1,n_vae_encoder_layers=1),
 output_dir=str(folder/'model'),batch_size=1,num_workers=0,steps=1,save_freq=1,log_freq=1,env_eval_freq=0,wandb=dict(enable=False))
(folder/'config.json').write_text(json.dumps(config))
with (folder/'train.log').open('w') as log:
 p=subprocess.run([str(root/'.venv-learning/bin/lerobot-train'),'--config_path='+str(folder/'config.json')],stdout=log,stderr=log)
if p.returncode:
 print((folder/'train.log').read_text()[-6000:]);raise RuntimeError('official CLI selftest failed')
report=m.validate(folder,check_budget=False)
import torch,importlib.metadata
report.update(torch=torch.__version__,lerobot=importlib.metadata.version('lerobot'),test_data='synthetic; not demonstrations; no physical success',folder=str(folder))
(root/'data/learning-selftest-result.json').write_text(json.dumps(report));print(json.dumps(report))
