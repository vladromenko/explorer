#!/usr/bin/env python3
"""Analyze an already captured factory hand-eye validation without motion."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from handeye import visual_pose
from validate_handeye_factory_support import transform_error

ROOT=Path('/home/vlad/Explorer')
parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
reference=json.loads((ROOT/'config/handeye-reference.json').read_text())
transform=np.asarray(reference['parent_T_child'],dtype=float)
paths=sorted(args.folder.glob('*.npz'));samples=[dict(np.load(path,allow_pickle=False)) for path in paths]
if len(samples)!=9:raise ValueError('Factory validation requires nine captured poses')
base=np.asarray([sample['base_pose'] for sample in samples])
if np.max(np.abs(base-base[0]))>.005:raise ValueError('Base moved during hand-eye validation')
comparisons=[]
for index in (1,3,5,7):
    observed,quality=visual_pose(samples[index-1],samples[index])
    predicted=np.linalg.inv(samples[index]['base_mount']@transform)@(samples[index-1]['base_mount']@transform)
    translation,rotation=transform_error(predicted,observed)
    comparisons.append(dict(index=index,joint=(index+1)//2,translation_error_m=translation,
                            rotation_error_deg=rotation,visual_quality=quality))
report=dict(at=time.time(),hardware_executed=True,simulation=False,outcome='passed',
    kind='handeye_factory_validation',joint_state_source='command_estimate',reference=str(ROOT/'config/handeye-reference.json'),
    samples=[str(path) for path in paths],comparisons=comparisons,
    max_translation_error_m=max(c['translation_error_m'] for c in comparisons),
    max_rotation_error_deg=max(c['rotation_error_deg'] for c in comparisons),
    camera_to_mount_reference=transform.tolist(),reference_mount='arm4',
    limits=dict(translation_m=.018,rotation_deg=3.),execution_authorized=False)
if report['max_translation_error_m']>=.018 or report['max_rotation_error_deg']>=3.:
    report.update(outcome='failed',reason='Fresh optical motion disagrees with saved transform')
else:report['execution_authorized']=True
(args.folder/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
