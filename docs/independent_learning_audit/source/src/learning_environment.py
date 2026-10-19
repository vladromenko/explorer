"""Writable local caches established before importing Torch/HF/LeRobot."""
import json
import os
from pathlib import Path


def configure(root):
    data=Path(root).expanduser().resolve()/"data"
    home=data/"hf-learning-cache"
    paths={"HF_HOME":home,"HF_DATASETS_CACHE":home/"datasets",
           "HF_HUB_CACHE":home/"hub","HUGGINGFACE_HUB_CACHE":home/"hub",
           "HF_ASSETS_CACHE":home/"assets","HF_LEROBOT_HOME":data/"lerobot-cache",
           "TORCH_HOME":data/"torch-learning-cache"}
    for name,path in paths.items():
        path.mkdir(parents=True,exist_ok=True)
        os.environ[name]=str(path)
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HUB_OFFLINE"]="1"
    return {name:str(path) for name,path in paths.items()}


def config_notes(environment):
    # TrainPipelineConfig is strict. Its supported disabled-W&B notes field
    # carries runtime provenance into the official saved train_config.json.
    return json.dumps({"explorer_cache_environment":environment},sort_keys=True)
