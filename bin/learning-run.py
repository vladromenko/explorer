#!/usr/bin/env python3
"""Export supervised decision samples, run official LeRobot ACT, validate offline.

An index denotes one completed, operator-observed decision, NOT elapsed seconds.
Only chunk_size=1 is used. Images/true wall times/command provenance are retained.
No serial, ROS, network upload or actuator execution exists in this process.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/home/vlad/Explorer')
os.environ['HF_HOME']=str(ROOT/'data/hf-learning-cache')
os.environ['HF_DATASETS_CACHE']=str(ROOT/'data/hf-learning-cache/datasets')
os.environ['TORCH_HOME']=str(ROOT/'data/torch-learning-cache')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(ROOT/'src'))
from lerobot_bridge import training_budget,write_json

def export_dataset(episodes,source,destination,repo_id):
    import cv2
    import numpy as np
    from lerobot.datasets import LeRobotDataset
    names=['base','shoulder','elbow','wrist_pitch','wrist_roll','gripper']
    features={
        'observation.state':{'dtype':'float32','shape':(6,),'names':names},
        'action':{'dtype':'float32','shape':(6,),'names':names},
        'observation.images.wrist':{'dtype':'image','shape':(240,320,3),'names':['height','width','channels']},
    }
    ds=LeRobotDataset.create(repo_id=repo_id,root=destination,fps=1,features=features,
                            robot_type='rosmaster_m3pro_commanded_discrete',use_videos=False)
    provenance=[]
    for episode in episodes:
        for step in episode['steps']:
            path=source/episode['id']/step['before_image']
            image=cv2.imread(str(path))
            if image is None:raise ValueError('Missing demonstration image')
            image=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2RGB)
            start=np.asarray(step['start_deg'],dtype=np.float32)
            goal=np.asarray(step['goal_deg'],dtype=np.float32)
            if not np.isfinite(start).all() or not np.isfinite(goal).all() or np.max(np.abs(goal-start))>5:
                raise ValueError('Invalid bounded operator decision')
            ds.add_frame({'observation.state':start,'action':goal,
                          'observation.images.wrist':image,'task':episode['name']})
        ds.save_episode()
        provenance.append(episode)
    ds.finalize()
    write_json(destination/'explorer-provenance.json',dict(episodes=provenance,
               joint_state_source='commanded_not_measured',clock='decision_index',
               physical_sample_rate_hz=None,autonomous_replay_allowed=False))

def export_mobile_dataset(episodes,source,destination,repo_id):
    import cv2
    import numpy as np
    from lerobot.datasets import LeRobotDataset
    names=['base','shoulder','elbow','wrist_pitch','wrist_roll','gripper','vx','vy','wz']
    features={'observation.state':{'dtype':'float32','shape':(9,),'names':names},
        'action':{'dtype':'float32','shape':(9,),'names':names},
        'observation.images.wrist':{'dtype':'image','shape':(240,320,3),'names':['height','width','channels']}}
    ds=LeRobotDataset.create(repo_id=repo_id,root=destination,fps=2,features=features,
        robot_type='rosmaster_m3pro_mobile_manipulation_commanded',use_videos=False)
    provenance=[]
    for episode in episodes:
        rows=[json.loads(line) for line in (source/episode['id']/'samples.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)<20:raise ValueError('Mobile episode has too few synchronized samples')
        for current,future in zip(rows,rows[1:]):
            state=np.asarray(current['command'],dtype=np.float32);action=np.asarray(future['command'],dtype=np.float32)
            if state.shape!=(9,) or action.shape!=(9,) or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError('Invalid mobile command sample')
            image=cv2.imread(str(source/episode['id']/current['image']))
            if image is None:raise ValueError('Missing mobile demonstration image')
            image=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2RGB)
            ds.add_frame({'observation.state':state,'action':action,'observation.images.wrist':image,
                'task':episode['name']+' / '+current['stage']})
        ds.save_episode();provenance.append(episode)
    ds.finalize();write_json(destination/'explorer-provenance.json',dict(episodes=provenance,
        format='explorer_mobile_episode_v2',joint_state_source='commanded_not_measured',fps=2,
        automatic_execution_allowed=False))

def run():
    request=json.loads((ROOT/'data/learning-request.json').read_text())
    ident=request['job']
    if any(c not in '0123456789abcdef-' for c in ident):raise ValueError('Invalid job id')
    folder=ROOT/'data/learning-jobs'/ident;state=json.loads((folder/'job.json').read_text())
    child=None
    def stop(signum,frame):
        raise InterruptedError('Обучение остановлено; сохранённые контрольные точки оставлены')
    signal.signal(signal.SIGTERM,stop)
    try:
        training_budget();state['state']='exporting';write_json(folder/'job.json',state)
        mobile=state.get('dataset_kind')=='mobile_manipulation_9dof'
        source=ROOT/('data/mobile-demonstrations' if mobile else 'data/demonstrations')
        episodes=[json.loads((source/e/'episode.json').read_text()) for e in state['episodes']]
        expected_source='operator_mobile_demonstration' if mobile else 'operator_demonstration'
        if len(episodes)<10 or any(e['outcome']!='success' or e['label_source']!='operator' or
                                  e['state']!='complete' or e['name']!=state['task'] or e['source']!=expected_source for e in episodes):
            raise ValueError('Demonstrations changed or were not verified by their operator')
        heldout=max(2,len(episodes)//5)
        exporter=export_mobile_dataset if mobile else export_dataset
        exporter(episodes[:-heldout],source,folder/'train','explorer/train')
        exporter(episodes[-heldout:],source,folder/'validation','explorer/validation')
        write_json(folder/'split.json',dict(train=[e['id'] for e in episodes[:-heldout]],
                   validation=[e['id'] for e in episodes[-heldout:]]))
        config=dict(dataset=dict(repo_id='explorer/train',root=str(folder/'train'),video_backend='pyav'),
            policy=dict(type='act',device='cuda',push_to_hub=False,chunk_size=1,n_action_steps=1,
                        dim_model=256,n_heads=4,dim_feedforward=1024,n_encoder_layers=2,
                        n_vae_encoder_layers=2),output_dir=str(folder/'model'),
            batch_size=4,num_workers=0,steps=state['steps'],save_freq=500,log_freq=25,
            env_eval_freq=0,wandb=dict(enable=False))
        write_json(folder/'train-config.json',config)
        state['state']='training';write_json(folder/'job.json',state)
        env=dict(os.environ,HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',OMP_NUM_THREADS='2')
        with (folder/'train.log').open('a') as log:
            child=subprocess.Popen([str(ROOT/'.venv-learning/bin/lerobot-train'),
                 '--config_path='+str(folder/'train-config.json')],stdout=log,stderr=log,env=env)
            while child.poll() is None:
                training_budget();time.sleep(2)
            if child.returncode:raise RuntimeError('LeRobot завершился с ошибкой; см. журнал обучения')
        state.update(state='validating');write_json(folder/'job.json',state)
        state['validation']=validate(folder)
        state.update(state='validated_offline',finished=time.time(),
                     note='Проверка по отдельным показам завершена. Физическое исполнение не разрешено')
    except (Exception,KeyboardInterrupt) as exc:
        state.update(state='interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',
                     error=str(exc),finished=time.time())
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        write_json(folder/'job.json',state)

def validate(folder,check_budget=True):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    path=folder/'model/checkpoints/last/pretrained_model'
    policy=ACTPolicy.from_pretrained(path,local_files_only=True)
    policy.eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(path))
    ds=LeRobotDataset('explorer/validation',root=folder/'validation',video_backend='pyav')
    errors=[];baseline=[]
    with torch.no_grad():
        for sample in torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
            if check_budget:training_budget()
            expected=sample['action'].clone().cpu().numpy()
            previous=sample['observation.state'].clone().cpu().numpy()
            policy.reset()
            actual=post(policy.select_action(pre(sample))).cpu().numpy()
            errors.extend(np.abs(actual-expected).reshape(-1).tolist())
            baseline.extend(np.abs(previous-expected).reshape(-1).tolist())
    if not errors or not np.isfinite(errors).all():raise ValueError('Проверка не дала корректных результатов')
    mobile=json.loads((folder/'job.json').read_text()).get('dataset_kind')=='mobile_manipulation_9dof'
    split=json.loads((folder/'split.json').read_text())
    total=len(split['train'])+len(split['validation'])
    result=dict(samples=len(ds),held_out_episodes=len(split['validation']),
                held_out_fraction=len(split['validation'])/total,
                mean_absolute_error=float(np.mean(errors)),
                p95_absolute_error=float(np.percentile(errors,95)),
                hold_position_baseline_mae=float(np.mean(baseline)),units='mixed_degrees_and_body_velocity' if mobile else 'degrees',
                improves_hold_baseline=bool(np.mean(errors)<np.mean(baseline)),
                measured_joint_ground_truth=False,physical_success_evaluated=False,
                automatic_execution_allowed=False)
    write_json(folder/'validation.json',result)
    return result

if __name__=='__main__':run()
