#!/usr/bin/env python3
"""Synthetic technical ACT 9D chain. No demonstrations or actuator commands."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default="/home/vlad/Explorer")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    root=Path(args.root).resolve()
    sys.path.insert(0,str(root/"src"))
    from learning_environment import configure,config_notes
    cache_environment=configure(root)
    os.environ["OMP_NUM_THREADS"]="2"
    import cv2
    import numpy as np
    import torch
    from lerobot_bridge import write_json
    from mobile_policy_contract import SAMPLE_CONTRACT,CONTRACT_SHA256,read_bundle
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    spec=importlib.util.spec_from_file_location("mobile_learning_selftest",root/"bin/learning-run.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=root/"data/learning-selftest"/("mobile-"+str(time.time_ns()))
    folder.mkdir(parents=True)
    episodes=[]
    for number in range(2):
        identifier="synthetic-"+str(number)
        source=folder/"source"/identifier;source.mkdir(parents=True)
        rows=[]
        for index in range(32):
            filename=f"{index:06d}.jpg"
            image=np.full((240,320,3),60+number*30,dtype=np.uint8)
            cv2.circle(image,(40+index*4,120),12,(20,150,230),-1)
            if not cv2.imwrite(str(source/filename),image):raise ValueError("Synthetic frame write failed")
            command=[90+index%7,90,90,90,90,90+index%3,.04 if index%4 else 0,0,0]
            rows.append({"image":filename,"image_stamp":number*10+index/4,"at":number*10+index/4,
                "state_stamp":number*10+index/4,"command":command,"applied_action":command,
                "observation_state":command,"sample_contract_sha256":CONTRACT_SHA256,
                "executed_action_source":"operator_manual_control",
                "test_data":"synthetic; never ingest as an operator demonstration"})
        (source/"samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        episodes.append({"id":identifier,"name":"Synthetic technical test","physical_sample_rate_hz":4,
                         "test_data":"synthetic; no physical outcome"})
    module.export_mobile_dataset(episodes[:1],folder/"source",folder/"train","explorer/train")
    module.export_mobile_dataset(episodes[1:],folder/"source",folder/"validation","explorer/validation")
    state={"dataset_kind":"mobile_manipulation_9dof","episodes":[e["id"] for e in episodes],
           "task":"Synthetic technical test","skill_id":None,"test_data":"synthetic"}
    write_json(folder/"job.json",state)
    write_json(folder/"split.json",{"train":[episodes[0]["id"]],"validation":[episodes[1]["id"]],
                                   "heldout_independent":True})
    config={"dataset":{"repo_id":"explorer/train","root":str(folder/"train"),"video_backend":"pyav"},
        "policy":{"type":"act","device":args.device,"push_to_hub":False,"chunk_size":1,"n_action_steps":1,
                  "pretrained_backbone_weights":None,"dim_model":64,"n_heads":4,"dim_feedforward":128,
                  "n_encoder_layers":1,"n_vae_encoder_layers":1},
        "output_dir":str(folder/"model"),"batch_size":1,"num_workers":0,"steps":2,
        "save_freq":1,"log_freq":1,"env_eval_freq":0,
        "wandb":{"enable":False,"notes":config_notes(cache_environment)}}
    write_json(folder/"config.json",config)
    write_json(folder/"train-config.environment.json",cache_environment)
    started=time.monotonic()
    with (folder/"train.log").open("w") as log:
        child=subprocess.run([str(root/".venv-learning/bin/lerobot-train"),
            "--config_path="+str(folder/"config.json")],stdout=log,stderr=log,timeout=180,env=dict(os.environ))
    if child.returncode:
        raise RuntimeError("Official ACT train failed: "+(folder/"train.log").read_text()[-3000:])
    checkpoint=module.resumable_checkpoint(folder)
    if checkpoint is None or checkpoint[0]!=2:raise ValueError("Complete optimizer checkpoint is missing")
    saved=sorted((folder/"model/checkpoints").glob("[0-9]*/pretrained_model/model.safetensors"))
    hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in saved]
    if len(set(hashes))<2:raise ValueError("Training did not change saved weights")
    # Exercise the installed CLI's optimizer/RNG resume path, rather than only
    # claiming that a checkpoint directory can be found.
    with (folder/"resume.log").open("w") as log:
        resumed=subprocess.run([str(root/".venv-learning/bin/lerobot-train"),
            "--config_path="+str(saved[0].parent/"train_config.json"),"--resume=true"],
            stdout=log,stderr=log,timeout=180,env=dict(os.environ))
    if resumed.returncode:
        raise RuntimeError("Official ACT resume failed: "+(folder/"resume.log").read_text()[-3000:])
    checkpoint=module.resumable_checkpoint(folder)
    if checkpoint is None or checkpoint[0]!=2:raise ValueError("Optimizer resume did not finish step two")
    state["validation"]=module.validate(folder,check_budget=False)
    module.create_mobile_bundle(folder,state)
    bundle,pretrained=read_bundle(folder)
    policy=ACTPolicy.from_pretrained(pretrained,local_files_only=True)
    policy.eval();policy.reset()
    preprocess,postprocess=make_pre_post_processors(policy.config,pretrained_path=str(pretrained))
    frame=torch.zeros((1,3,240,320),dtype=torch.float32)
    with torch.no_grad():
        prediction=postprocess(policy.select_action(preprocess({"observation.state":torch.tensor(
            [[90]*6+[0,0,0]],dtype=torch.float32),"observation.images.wrist":frame}))).cpu().numpy()
    if prediction.shape!=(1,9) or not np.isfinite(prediction).all():raise ValueError("Compatible inference failed")
    report={"passed":True,"technical_chain":["mobile_export","official_ACT_weight_update", "optimizer_checkpoint",
        "official_optimizer_resume","bundle_checksum_and_contract","compatible_policy_reload","finite_9D_inference"],
        "folder":str(folder),"device":args.device,"elapsed_s":round(time.monotonic()-started,3),
        "checkpoint_step":checkpoint[0],"weight_sha256":hashes,"contract_sha256":CONTRACT_SHA256,
        "dataset_fps":bundle["dataset_fps"],"action_shape":list(prediction.shape),
        "cache_environment":cache_environment,
        "test_data":"synthetic; no real demonstrations existed in current installation",
        "physical_success_verified":False,"actuator_commands_sent":0}
    write_json(folder/"result.json",report)
    write_json(root/"data/mobile-learning-selftest-result.json",report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":main()
