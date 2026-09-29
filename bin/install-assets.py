#!/usr/bin/env python3
"""Restore large third-party runtime assets from a local export and verify them."""
import argparse,hashlib,json,shutil
from pathlib import Path

ROOT=Path('/home/vlad/Explorer')
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True,help='directory produced by export-assets.sh')
args=parser.parse_args();source=args.source.resolve()
for name in ('models','vendor'):
    if not (source/name).is_dir():raise SystemExit('В экспорте нет папки '+name)
    shutil.copytree(source/name,ROOT/name,dirs_exist_ok=True,symlinks=True)
manifest=json.loads((ROOT/'config/components.json').read_text())
required=dict(manifest['models'])
grounding=json.loads((ROOT/'config/grounding-model.json').read_text())
required.update({'models/grounding-dino-tiny/'+name:value for name,value in grounding['files'].items()})
required.update({
    'models/voice/ggml-base.bin':None,
    'models/voice/ru_RU-irina-medium.onnx':None,
    'models/voice/ru_RU-irina-medium.onnx.json':None,
    'vendor/ros/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent':None,
    'vendor/ros/orbbec_camera/share/orbbec_camera/launch/dabai_dcw2.launch.py':None,
    'vendor/llama/llama-server':None,
    'vendor/tailscale/tailscale':None,
    'vendor/tailscale/tailscaled':None,
    'vendor/bin/awgproxy':None,
})
missing=[]
for relative,expected in required.items():
    dst=ROOT/relative
    if not dst.is_file():missing.append(relative);continue
    if expected and hashlib.sha256(dst.read_bytes()).hexdigest()!=expected:raise SystemExit('SHA256 не совпал: '+relative)
if missing:raise SystemExit('В архиве отсутствуют:\n'+'\n'.join(missing))
print('Большие модели и исполняемые зависимости восстановлены и проверены.')
