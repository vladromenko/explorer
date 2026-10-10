#!/usr/bin/env python3
"""One local ACT inference, without sending a command to the robot."""
import json
import os
from pathlib import Path
import sys
import cv2
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['OMP_NUM_THREADS']='2'
ROOT=Path('/home/vlad/Explorer')
sys.path.insert(0,str(ROOT/'src'))
from learning_environment import configure
configure(ROOT)
from policy_preview import infer
from lerobot_bridge import write_json
folder=Path(sys.argv[1]);request=json.loads((folder/'request.json').read_text())
image=cv2.imread(str(folder/'image.jpg'))
if image is None:raise ValueError('Нет изображения для проверки')
result=infer(Path(request['checkpoint']),image,request['pose'])
result['observed_at']=request['observed_at']
write_json(folder/'result.json',result)
