#!/usr/bin/env python3
"""Isolated GroundingDINO inference on one recorded RGB frame; no robot imports."""
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor

folder=Path(sys.argv[1]);request=json.loads((folder/'request.json').read_text())
lock=json.loads((Path(__file__).resolve().parents[1]/'config/grounding-model.json').read_text())
model_path=Path(request['model'])
for name,digest in lock['files'].items():
    with (model_path/name).open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual!=digest:raise ValueError('Grounding model checksum mismatch: '+name)
torch.set_num_threads(2)
processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
processor.image_processor.size={'shortest_edge':640,'longest_edge':960}
model=AutoModelForZeroShotObjectDetection.from_pretrained(model_path,local_files_only=True).eval()
model.config.disable_custom_kernels=True
image=Image.open(folder/'image.png').convert('RGB')
inputs=processor(images=image,text=request['query'].strip().rstrip('.')+'.',return_tensors='pt')
with torch.inference_mode():output=model(**inputs)
result=processor.post_process_grounded_object_detection(output,inputs.input_ids,threshold=.35,
    text_threshold=.25,target_sizes=[image.size[::-1]])[0]
labels=result.get('text_labels',result.get('labels',[]))
rows=[dict(bbox=box.tolist(),confidence=float(score),label=str(label))
    for box,score,label in zip(result['boxes'],result['scores'],labels)]
(folder/'result.json').write_text(json.dumps(dict(frame_id=request['frame_id'],detections=rows,
    model=lock['repository'],revision=lock['revision'],input='recorded_RGB_only',identity_proven=False)))
