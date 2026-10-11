#!/usr/bin/env python3
"""Local open-vocabulary grounding. Camera coordinates only; never commands motion."""
import os
os.environ['HF_HOME']='/home/vlad/Explorer/data/hf-learning-cache'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['OMP_NUM_THREADS']='2'
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
ROOT=Path('/home/vlad/Explorer');sys.path.insert(0,str(ROOT/'src'))
from surface_foreground import locate
from detection_candidates import unique_candidates
folder=Path(sys.argv[1]);request=json.loads((folder/'request.json').read_text())
sample=np.load(folder/'rgbd.npz',allow_pickle=False)
image=Image.fromarray(cv2.cvtColor(sample['rgb'],cv2.COLOR_BGR2RGB));started=time.monotonic()
model_path=ROOT/'models/grounding-dino-tiny'
torch.set_num_threads(2)
processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
# Bound attention tensors on the shared 8 GB Orin. Keep boxes in original
# image coordinates via the processor's target_sizes below.
processor.image_processor.size={"shortest_edge":640,"longest_edge":960}
model=AutoModelForZeroShotObjectDetection.from_pretrained(model_path,local_files_only=True).to(
    device="cuda",dtype=torch.float16).eval()
model.config.disable_custom_kernels=True
inputs=processor(images=image,text=request['english_label']+'.',return_tensors='pt').to(
    device="cuda",dtype=torch.float16)
with torch.inference_mode():outputs=model(**inputs)
result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=.35,text_threshold=.25,target_sizes=[image.size[::-1]])[0]
objects=[];annotated=sample['rgb'].copy()
labels=result.get('text_labels',result.get('labels',[]))
for box,score,label in zip(result['boxes'],result['scores'],labels):
    bbox=box.cpu().tolist();geometry=locate(sample,bbox)
    objects.append(dict(label=str(label),requested_label=request['label'],bbox=bbox,confidence=float(score),
                        position=geometry['position'],depth_evidence=geometry))
raw_objects = list(objects)
objects = unique_candidates(raw_objects)
for item in objects:
    x1, y1, x2, y2 = map(int, item["bbox"])
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 230, 140), 2)
    cv2.putText(annotated, request["english_label"] + " " + str(round(item["confidence"], 2)),
        (x1, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 230, 140), 1)
cv2.imwrite(str(folder/'result.jpg'),annotated)
record=dict(objects=objects,raw_objects=raw_objects,duplicate_filter="same_label_iou_nms_0.5",
            image_stamp=float(sample['stamp']),frame=str(sample['frame']),
            processing_seconds=time.monotonic()-started,model='GroundingDINO-tiny',
            runtime={"device":"cuda","dtype":"float16","shortest_edge":640,"longest_edge":960},
            world_coordinates_validated=False,executed=False,identity_verified=False)
(folder/'result.json').write_text(json.dumps(record,allow_nan=False))
