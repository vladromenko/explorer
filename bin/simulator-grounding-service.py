#!/usr/bin/env python3
"""One pinned RGB-only model shared serially by simulator workers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('socket',type=Path)
    args=parser.parse_args();root=args.root.resolve();endpoint=args.socket
    lock_path=endpoint.with_suffix('.lock')
    with lock_path.open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        from PIL import Image
        import torch
        from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
        lock=json.loads((Path(__file__).resolve().parents[1]/'config/grounding-model.json').read_text())
        model_path=root/'models/grounding-dino-tiny'
        for name,digest in lock['files'].items():
            with (model_path/name).open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=digest:raise ValueError('Grounding model checksum mismatch: '+name)
        torch.set_num_threads(2)
        processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        processor.image_processor.size={'shortest_edge':640,'longest_edge':960}
        model=AutoModelForZeroShotObjectDetection.from_pretrained(model_path,local_files_only=True).eval()
        model.config.disable_custom_kernels=True
        from simulator.vision_backend import PinnedDetectorBackend
        backend=PinnedDetectorBackend(torch,model,processor)
        endpoint.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(endpoint));os.chmod(endpoint,0o600);server.listen(16);server.settimeout(20.)
            last=time.monotonic()
            try:
                while time.monotonic()-last<300:
                    try:connection,_=server.accept()
                    except socket.timeout:continue
                    with connection:
                        connection.settimeout(5.)
                        try:
                            data=b''
                            while b'\n' not in data and len(data)<16384:
                                block=connection.recv(4096)
                                if not block:break
                                data+=block
                            message=json.loads(data)
                            if message.get('kind')=='ping':reply=dict(ready=True,pid=os.getpid(),revision=lock['revision'])
                            else:
                                folder=Path(message['folder']).resolve()
                                if not folder.is_relative_to(root):raise ValueError('Inference input outside simulator storage')
                                request=json.loads((folder/'request.json').read_text())
                                if Path(request['model']).resolve()!=model_path:raise ValueError('Different model requested')
                                image=Image.open(folder/'image.png').convert('RGB')
                                inputs=processor(images=image,text=request['query'].strip().rstrip('.')+'.',return_tensors='pt')
                                result=backend.infer(inputs,image.size[::-1])
                                (root/'maintenance/vision-backend.json').write_text(json.dumps(backend.measurement))
                                labels=result.get('text_labels',result.get('labels',[]))
                                rows=[dict(bbox=box.tolist(),confidence=float(score),label=str(label))
                                    for box,score,label in zip(result['boxes'],result['scores'],labels)]
                                value=dict(frame_id=request['frame_id'],detections=rows,model=lock['repository'],
                                    revision=lock['revision'],input='recorded_RGB_only',identity_proven=False,
                                    backend='persistent_pinned_model',worker_pid=os.getpid(),runtime=backend.measurement)
                                temp=folder/'result.pending.json';temp.write_text(json.dumps(value));temp.replace(folder/'result.json')
                                reply=dict(ready=True,frame_id=request['frame_id'],pid=os.getpid())
                            last=time.monotonic()
                        except Exception as error:reply=dict(error=type(error).__name__+': '+str(error))
                        try:connection.sendall(json.dumps(reply).encode()+b'\n')
                        except (OSError,BrokenPipeError):pass
            finally:endpoint.unlink(missing_ok=True)


if __name__=='__main__':main()
