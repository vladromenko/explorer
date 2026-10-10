#!/usr/bin/env python3
"""Export supervised decision samples, run official LeRobot ACT, validate offline.

An index denotes one completed, operator-observed decision, NOT elapsed seconds.
Only chunk_size=1 is used. Images/true wall times/command provenance are retained.
No serial, ROS, network upload or actuator execution exists in this process.
"""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from learning_environment import configure,config_notes
CACHE_ENVIRONMENT=configure(ROOT)
from lerobot_bridge import training_budget,write_json,save_job
from mobile_policy_contract import SAMPLE_CONTRACT,CONTRACT_SHA256,read_bundle


class TrainingDeferred(ValueError):
    """Foreground work or power preempted a background learner."""


def budget():
    try:training_budget(ROOT)
    except (OSError,ValueError,KeyError) as exc:raise TrainingDeferred(str(exc)) from exc


def export_once(exporter,episodes,source,destination,repo_id,task=None):
    fingerprint=hashlib.sha256(json.dumps(episodes,sort_keys=True).encode()).hexdigest()
    ready=destination/"explorer-export-ready.json"
    if ready.exists() and json.loads(ready.read_text()).get("input_sha256")==fingerprint:
        return
    if destination.exists():
        destination.rename(destination.with_name(destination.name+"-partial-"+str(time.time_ns())))
    if task is None:exporter(episodes,source,destination,repo_id)
    else:exporter(episodes,source,destination,repo_id,task)
    write_json(ready,{"input_sha256":fingerprint,"finalized":True,"at":time.time()})


def resumable_checkpoint(folder):
    candidates=list((folder/"model/checkpoints").glob("*/pretrained_model/train_config.json"))
    complete=[]
    required=("training_step.json","optimizer_state.safetensors","rng_state.safetensors")
    for config in candidates:
        state=config.parent.parent/"training_state"
        if (config.parent/"model.safetensors").is_file() and all((state/name).is_file() for name in required):
            step=json.loads((state/"training_step.json").read_text())["step"]
            complete.append((int(step),config))
    return max(complete,key=lambda item:item[0]) if complete else None


def create_mobile_bundle(folder,state):
    checkpoint=folder/"model/checkpoints/last/pretrained_model"
    bundle_files={str(path.relative_to(checkpoint)):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in checkpoint.rglob("*") if path.is_file()}
    if "model.safetensors" not in bundle_files or "config.json" not in bundle_files:
        raise ValueError("Checkpoint не содержит обязательные веса и конфигурацию")
    provenance=json.loads((folder/"train/explorer-provenance.json").read_text())
    bundle={"format":"explorer_mobile_act_bundle_v2","created":time.time(),
        "skill_id":state.get("skill_id"),"task":state["task"],"dataset_kind":"mobile_manipulation_9dof",
        "episodes":state["episodes"],"dataset_fingerprint":state.get("dataset_fingerprint"),
        "checkpoint_relative":"model/checkpoints/last/pretrained_model","checkpoint_sha256":bundle_files,
        "framework":"lerobot","policy":"act","observation_order":SAMPLE_CONTRACT["action_order"],
        "action_order":SAMPLE_CONTRACT["action_order"],"units":SAMPLE_CONTRACT["units"],
        "joint_state_source":"command_estimate","camera_feature":"observation.images.wrist",
        "sample_contract":SAMPLE_CONTRACT,"sample_contract_sha256":CONTRACT_SHA256,
        "dataset_fps":provenance["dataset_fps"],"chunk_size":1,"action_steps":1,
        "validation":state["validation"],"physical_success_verified":False,
        "autonomous_motion_on_registration":False}
    write_json(folder/"bundle.json",bundle)
    read_bundle(folder)
    return bundle

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

def export_mobile_dataset(episodes,source,destination,repo_id,task_name=None):
    import cv2
    import numpy as np
    from lerobot.datasets import LeRobotDataset
    names=['base','shoulder','elbow','wrist_pitch','wrist_roll','gripper','vx','vy','wz']
    features={'observation.state':{'dtype':'float32','shape':(9,),'names':names},
        'action':{'dtype':'float32','shape':(9,),'names':names},
        'observation.images.wrist':{'dtype':'image','shape':(240,320,3),'names':['height','width','channels']}}
    rates=[float(episode.get("physical_sample_rate_hz") or 0) for episode in episodes]
    if not rates or min(rates)<1.5:raise ValueError("Нет подтверждённой частоты кадров")
    fps=int(round(float(np.median(rates))))
    if fps<2 or any(abs(rate-fps)/fps>.25 for rate in rates):
        raise ValueError("Показы имеют несовместимые частоты кадров; нужен отдельный набор")
    ds=LeRobotDataset.create(repo_id=repo_id,root=destination,fps=fps,features=features,
        robot_type='rosmaster_m3pro_mobile_manipulation_commanded',use_videos=False)
    provenance=[]
    for episode in episodes:
        rows=[json.loads(line) for line in (source/episode['id']/'samples.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)<12:raise ValueError("Слишком мало синхронизированных кадров")
        boundaries=[0]
        for index in range(1,len(rows)):
            dt=float(rows[index]["image_stamp"])-float(rows[index-1]["image_stamp"])
            if not 0<dt<=max(.75,2/fps):boundaries.append(index)
        boundaries.append(len(rows))
        fragments=[]
        for start,end in zip(boundaries,boundaries[1:]):
            if end-start>=3:
                for current,future in zip(rows[start:end-1],rows[start+1:end]):
                    if current.get("sample_contract_sha256") not in (None,CONTRACT_SHA256):
                        raise ValueError("Несовместимые единицы или семантика команд показа")
                    if current.get("executed_action_source") not in (None,"operator_manual_control"):
                        raise ValueError("Команды модели не используются как успешный показ")
                    state=np.asarray(current.get("observation_state",current["command"]),dtype=np.float32)
                    action=np.asarray(future.get("applied_action",future["command"]),dtype=np.float32)
                    if state.shape!=(9,) or action.shape!=(9,) or not np.isfinite(state).all() or not np.isfinite(action).all():
                        raise ValueError('Invalid mobile command sample')
                    image=cv2.imread(str(source/episode['id']/current['image']))
                    if image is None:raise ValueError('Missing mobile demonstration image')
                    image=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2RGB)
                    ds.add_frame({'observation.state':state,'action':action,'observation.images.wrist':image,
                        'task':task_name or episode['name']})
                ds.save_episode()
                fragments.append({"start_row":start,"end_row_exclusive":end,
                                  "first_image_stamp":rows[start]["image_stamp"],
                                  "last_image_stamp":rows[end-1]["image_stamp"]})
        if not fragments:raise ValueError("После пропусков кадров нет непрерывного фрагмента")
        input_files=[source/episode["id"]/"samples.jsonl"]
        input_files.extend(source/episode["id"]/row["image"] for row in rows)
        inputs={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files}
        provenance.append({"episode":episode,"fragments":fragments,"input_sha256":inputs,
            "dropped_short_fragments":len(boundaries)-1-len(fragments)})
    ds.finalize();write_json(destination/'explorer-provenance.json',dict(episodes=provenance,
        format="explorer_mobile_episode_v3",joint_state_source="commanded_not_measured",
        sample_contract=SAMPLE_CONTRACT,sample_contract_sha256=CONTRACT_SHA256,
        dataset_fps=fps,physical_rates_hz=rates,original_timestamps="samples.jsonl:image_stamp,at,state_stamp",
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
        budget();state.update(state="exporting",at=time.time());save_job(folder/"job.json",state)
        mobile=state.get('dataset_kind')=='mobile_manipulation_9dof'
        source=ROOT/('data/mobile-demonstrations' if mobile else 'data/demonstrations')
        episodes=[json.loads((source/e/'episode.json').read_text()) for e in state['episodes']]
        for identifier,files in state.get("input_sha256",{}).items():
            for name,digest in files.items():
                if hashlib.sha256((source/identifier/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError("Показ изменён после постановки в очередь")
        expected_source='operator_mobile_demonstration' if mobile else 'operator_demonstration'
        if not episodes or any(e['outcome']!='success' or e['label_source']!='operator' or
                                  e['state']!='complete' or (e.get('skill_id')!=state['skill_id'] if state.get('skill_id') else e['name']!=state['task']) or
                                  e['source']!=expected_source for e in episodes):
            raise ValueError('Demonstrations changed or were not verified by their operator')
        heldout=max(1,len(episodes)//5) if len(episodes)>1 else 0
        exporter=export_mobile_dataset if mobile else export_dataset
        training=episodes[:-heldout] if heldout else episodes
        validation=episodes[-heldout:] if heldout else episodes
        if mobile:
            export_once(exporter,training,source,folder/"train","explorer/train",state["task"])
            export_once(exporter,validation,source,folder/"validation","explorer/validation",state["task"])
        else:
            export_once(exporter,training,source,folder/"train","explorer/train")
            export_once(exporter,validation,source,folder/"validation","explorer/validation")
        write_json(folder/'split.json',dict(train=[e['id'] for e in training],
                   validation=[e['id'] for e in validation],heldout_independent=bool(heldout)))
        config=dict(dataset=dict(repo_id='explorer/train',root=str(folder/'train'),video_backend='pyav'),
            policy=dict(type='act',device='cuda',push_to_hub=False,chunk_size=1,n_action_steps=1,
                        pretrained_backbone_weights=None,
                        dim_model=256,n_heads=4,dim_feedforward=1024,n_encoder_layers=2,
                        n_vae_encoder_layers=2),output_dir=str(folder/'model'),
            batch_size=4,num_workers=0,steps=state["steps"],save_freq=25,log_freq=25,
            env_eval_freq=0,wandb=dict(enable=False,notes=config_notes(CACHE_ENVIRONMENT)))
        write_json(folder/'train-config.json',config)
        write_json(folder/"train-config.environment.json",CACHE_ENVIRONMENT)
        checkpoint=resumable_checkpoint(folder)
        args=[str(ROOT/".venv-learning/bin/lerobot-train")]
        if checkpoint:
            args.extend(["--config_path="+str(checkpoint[1]),"--resume=true"])
            state["resumed_step"]=checkpoint[0]
        else:
            if (folder/"model").exists():
                (folder/"model").rename(folder/("model-partial-"+str(time.time_ns())))
            args.append("--config_path="+str(folder/"train-config.json"))
        state.update(state="training",at=time.time());save_job(folder/"job.json",state)
        env=dict(os.environ,HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',OMP_NUM_THREADS='2')
        with (folder/'train.log').open('a') as log:
            child=subprocess.Popen(args,stdout=log,stderr=log,env=env)
            while child.poll() is None:
                if json.loads((folder/"job.json").read_text()).get("cancel_requested"):
                    raise InterruptedError("Обучение отменено оператором")
                budget();state["at"]=time.time();save_job(folder/"job.json",state);time.sleep(2)
            if child.returncode:raise RuntimeError('LeRobot завершился с ошибкой; см. журнал обучения')
        state.update(state="validating",at=time.time());save_job(folder/"job.json",state)
        state['validation']=validate(folder)
        if mobile:
            create_mobile_bundle(folder,state)
            state["bundle"]="bundle.json"
        terminal='validated_offline' if heldout else 'trained_unvalidated'
        note=('Проверка по отдельным показам завершена. Физическое исполнение не разрешено' if heldout else
              'Модель обучена на единственном показе; независимой offline-выборки нет, продвижение запрещено')
        state.update(state=terminal,finished=time.time(),note=note)
    except (Exception,KeyboardInterrupt) as exc:
        interrupted=isinstance(exc,(TrainingDeferred,InterruptedError,KeyboardInterrupt))
        state.update(state="deferred" if interrupted else "failed",
                     error=str(exc),finished=time.time())
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        persisted=json.loads((folder/"job.json").read_text())
        if persisted.get("cancel_requested"):
            state.update(state="cancelled",cancel_requested=True)
        state["at"]=time.time();save_job(folder/"job.json",state,terminal=True)

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
            if check_budget:budget()
            expected=sample['action'].clone().cpu().numpy()
            previous=sample['observation.state'].clone().cpu().numpy()
            policy.reset()
            actual=post(policy.select_action(pre(sample))).cpu().numpy()
            errors.extend(np.abs(actual-expected).reshape(-1,expected.shape[-1]).tolist())
            baseline.extend(np.abs(previous-expected).reshape(-1,expected.shape[-1]).tolist())
    errors=np.asarray(errors,dtype=float);baseline=np.asarray(baseline,dtype=float)
    if not len(errors) or not np.isfinite(errors).all():raise ValueError('Проверка не дала корректных результатов')
    mobile=json.loads((folder/'job.json').read_text()).get('dataset_kind')=='mobile_manipulation_9dof'
    split=json.loads((folder/'split.json').read_text())
    independent=split.get('heldout_independent',True)
    total=len(set(split['train'])|set(split['validation']))
    scales=np.asarray([10.]*6+[.8,.72,1.67] if mobile else [10.]*6)
    weighted_error=float(np.mean(errors/scales))
    weighted_baseline=float(np.mean(baseline/scales))
    result=dict(samples=len(ds),held_out_episodes=len(split['validation']),
                held_out_fraction=len(split['validation'])/total if independent else 0.,
                mean_absolute_error=float(np.mean(errors)),
                p95_absolute_error=float(np.percentile(errors,95)),
                hold_position_baseline_mae=float(np.mean(baseline)),units='mixed_degrees_and_body_velocity' if mobile else 'degrees',
                normalized_mae=weighted_error,normalized_hold_baseline_mae=weighted_baseline,
                joint_mae_deg=float(np.mean(errors[:,:6])),
                joint_hold_baseline_mae_deg=float(np.mean(baseline[:,:6])),
                base_mae_mixed=float(np.mean(errors[:,6:])) if mobile else None,
                base_hold_baseline_mae_mixed=float(np.mean(baseline[:,6:])) if mobile else None,
                improves_hold_baseline=bool(independent and weighted_error<weighted_baseline),
                measured_joint_ground_truth=False,physical_success_evaluated=False,
                automatic_execution_allowed=False,heldout_independent=independent)
    write_json(folder/'validation.json',result)
    return result

if __name__=='__main__':run()
