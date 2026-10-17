#!/usr/bin/env python3
"""Read-only CPU/MPS benchmark on recorded RGB; never submits robot trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from PIL import Image
    import numpy as np
    import torch
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    from simulator.resource_budget import memory_budget
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recording',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=dict(kind='recorded_RGB_performance_only',physical_ready=False,
        enabled_in_production=False,torch=torch.__version__,devices={})
    lock=json.loads((ROOT/'config/grounding-model.json').read_text());model_path=args.root/'models/grounding-dino-tiny'
    def guard():
        m=memory_budget()
        if m['available_bytes']/m['total_bytes']<.20:raise MemoryError('Benchmark stopped to retain 20% system headroom')
    try:
        guard()
        for name,digest in lock['files'].items():
            with (model_path/name).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=digest:raise ValueError('Model checksum mismatch')
        torch.set_num_threads(2)
        processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        processor.image_processor.size={'shortest_edge':640,'longest_edge':960}
        model=AutoModelForZeroShotObjectDetection.from_pretrained(model_path,local_files_only=True).eval()
        model.config.disable_custom_kernels=True
        inputs=[]
        for folder in args.recording:
            request=json.loads((folder/'request.json').read_text());im=Image.open(folder/'image.png').convert('RGB')
            inputs.append((processor(images=im,text=request['query'].strip().rstrip('.')+'.',return_tensors='pt'),im.size[::-1]))
        for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
            guard();model.to(device);measurements=[];rows=[]
            for index,(batch,size) in enumerate([inputs[0]]+inputs):
                guard();batch={k:v.to(device) for k,v in batch.items()}
                if device=='mps':torch.mps.synchronize()
                started=time.perf_counter()
                with torch.inference_mode():output=model(**batch)
                if device=='mps':torch.mps.synchronize()
                elapsed=time.perf_counter()-started
                result=processor.post_process_grounded_object_detection(output,batch['input_ids'],threshold=.35,
                    text_threshold=.25,target_sizes=[size])[0]
                labels=result.get('text_labels',result.get('labels',[]))
                if index:
                    measurements.append(elapsed)
                    rows.append([dict(bbox=b.cpu().tolist(),confidence=float(s.cpu()),label=str(l)) for b,s,l in zip(result['boxes'],result['scores'],labels)])
            report['devices'][device]=dict(warm_inference_s=measurements,detections=rows)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
        if 'mps' in report['devices']:
            a,b=[report['devices'][d] for d in ['cpu','mps']]
            equivalent=True;max_box=0.;max_score=0.
            for x,y in zip(a['detections'],b['detections']):
                if len(x)!=len(y) or [z['label'] for z in x]!=[z['label'] for z in y]:equivalent=False;continue
                for u,v in zip(x,y):
                    max_box=max(max_box,float(np.max(np.abs(np.array(u['bbox'])-v['bbox']))))
                    max_score=max(max_score,abs(u['confidence']-v['confidence']))
            report.update(speedup=sum(a['warm_inference_s'])/sum(b['warm_inference_s']),
                same_detection_decisions=equivalent,max_box_difference_px=max_box,max_confidence_difference=max_score,
                note='Recorded RGB only; production backend unchanged. Full actor requalification required before changing numerics.')
    except Exception as e:report['error']=type(e).__name__+': '+str(e)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='devices'}))


if __name__=='__main__':main()
