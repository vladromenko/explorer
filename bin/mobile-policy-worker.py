#!/usr/bin/env python3
"""Inference process isolated from ROS and every actuator."""
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["OMP_NUM_THREADS"]="2"
ROOT=Path("/home/vlad/Explorer")
sys.path.insert(0,str(ROOT/"src"))
from learning_environment import configure
configure(ROOT)

import cv2
import numpy as np
import torch
from lerobot.policies.act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies import make_pre_post_processors
from lerobot_bridge import write_json
from mobile_policy_contract import read_bundle,decode_policy_action,clock_features


def run(folder):
    folder=Path(folder)
    config=json.loads((folder/"config.json").read_text())
    bundle,checkpoint=read_bundle(config["job_folder"])
    model_config=ACTConfig.from_pretrained(checkpoint,local_files_only=True)
    model_config.pretrained_backbone_weights=None
    policy=ACTPolicy.from_pretrained(checkpoint,config=model_config,local_files_only=True)
    policy.eval()
    policy.reset()
    preprocess,postprocess=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
    policy_started=None
    write_json(folder/"ready.json",{"ready":True,"skill_id":bundle.get("skill_id"),
                                    "action_order":bundle["action_order"]})
    for step in range(300):
        request=folder/f"{step:04d}-request.json"
        deadline=time.monotonic()+10
        while not request.exists() and time.monotonic()<deadline:time.sleep(.02)
        if not request.exists():return
        data=json.loads(request.read_text())
        if data.get("reset_history") is True:
            policy.reset()
            policy_started=None
        image=cv2.imread(str(folder/f"{step:04d}.jpg"))
        if image is None:raise ValueError("Нет кадра для политики")
        state=np.asarray(data["state"],dtype=np.float32)
        if state.shape!=(9,) or not np.isfinite(state).all():raise ValueError("Неверное состояние робота")
        rgb=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2RGB)
        observation={"observation.state":torch.from_numpy(state.copy()).unsqueeze(0),
                     "observation.images.wrist":torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).unsqueeze(0)}
        context=bundle.get("actor_context")
        if context:
            if policy_started is None:policy_started=time.monotonic()
            observation[context["feature"]]=torch.tensor([clock_features(context,time.monotonic()-policy_started)],dtype=torch.float32)
        started=time.monotonic()
        with torch.no_grad():action=postprocess(policy.select_action(preprocess(observation))).cpu().numpy().reshape(-1)
        if action.shape!=(9,) or not np.isfinite(action).all():raise ValueError("Политика выдала неверную команду")
        action=decode_policy_action(state,action,bundle)
        write_json(folder/f"{step:04d}-result.json",{
            "observed_at":data["observed_at"],"state":data["state"],
            "action":action,"inference_s":time.monotonic()-started})


if __name__=="__main__":run(sys.argv[1])
