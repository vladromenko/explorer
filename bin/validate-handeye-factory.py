#!/usr/bin/env python3
"""Validate the saved arm4-to-camera transform with fresh bounded robotio moves."""
import json
import math
from pathlib import Path
import time
import urllib.request
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from arm_commissioning import stationary_status
from arm_model import ArmModel
from handeye import visual_pose

ROOT=Path('/home/vlad/Explorer')
token=(ROOT/'config/access_token').read_text().strip()

def api(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)

def read(name):return json.loads((ROOT/'data'/name).read_text())

def capture(folder,index,goal,model):
    time.sleep(1.0)
    status=read('status.json');stationary_status(status,time.time())
    arm=read('arm-state.json')
    if arm.get('servo_deg')!=goal or arm.get('phase')!='command_elapsed_observation_required':
        raise ValueError('Arm command estimate does not match capture pose')
    with np.load(ROOT/'data/rgbd-snapshot.npz',allow_pickle=False) as source:sample={key:source[key] for key in source.files}
    if not 0<=time.time()-float(sample['stamp'])<1.5 or float(sample['stamp'])<arm['at']+.3:
        raise ValueError('RGB-D frame is not fresh after the arm settled')
    model.set_state(goal[:5],-.3)
    path=folder/f'{index:02d}.npz'
    np.savez_compressed(path,**sample,servo_deg=goal,
        base_mount=model.state.get_global_link_transform('arm4'),mount_frame='arm4',
        base_pose=np.array([status['raw_pose'][key] for key in ('x','y','yaw')]),
        measured_joint_positions=False,joint_state_source='command_estimate')
    return path

def transform_error(predicted,observed):
    error=np.linalg.inv(predicted)@observed
    return float(np.linalg.norm(error[:3,3])),float(np.degrees(Rotation.from_matrix(error[:3,:3]).magnitude()))

def main():
    reference=json.loads((ROOT/'config/handeye-reference.json').read_text())
    transform=np.asarray(reference['parent_T_child'],dtype=float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Invalid reference transform')
    start=read('arm-state.json').get('servo_deg')
    allowed=([90]*6,[90,125,3,0,90,30])
    if start not in allowed:raise ValueError('Fresh validation starts only from an observed accepted pose')
    deltas=(10,-10,10,10) if start==allowed[1] else (10,10,10,10)
    poses=[list(start)]
    for joint,offset in enumerate(deltas):
        moved=list(start);moved[joint]+=offset;poses.extend([moved,list(start)])
    model=ArmModel()
    previous=poses[0]
    for goal in poses[1:]:
        for shape in (0.,-.2,-.4,-.6,-.8):
            if not model.path(previous[:5],goal[:5],shape)['valid']:raise ValueError('Validation path collides')
        previous=goal
    folder=ROOT/'data/handeye-validations'/time.strftime('%Y%m%d-%H%M%S');folder.mkdir(parents=True)
    paths=[];current=list(start)
    try:
        paths.append(capture(folder,0,current,model))
        for index,goal in enumerate(poses[1:],1):
            changed=[i for i,(a,b) in enumerate(zip(current,goal)) if a!=b]
            if len(changed)!=1 or abs(goal[changed[0]]-current[changed[0]])!=10:raise ValueError('Non-bounded validation transition')
            result=api('arm/jog',dict(joint=changed[0]+1,delta=goal[changed[0]]-current[changed[0]],observing=True))
            if result.get('servo_deg')!=goal:raise ValueError('Arm command receipt mismatch')
            current=list(goal);paths.append(capture(folder,index,current,model))
        samples=[dict(np.load(path,allow_pickle=False)) for path in paths]
        base=np.asarray([sample['base_pose'] for sample in samples])
        if np.max(np.abs(base-base[0]))>.005:raise ValueError('Base moved during hand-eye validation')
        comparisons=[]
        for index in (1,3,5,7):
            observed,quality=visual_pose(samples[index-1],samples[index])
            reference_camera=samples[index-1]['base_mount']@transform
            current_camera=samples[index]['base_mount']@transform
            predicted=np.linalg.inv(current_camera)@reference_camera
            translation,rotation=transform_error(predicted,observed)
            comparisons.append(dict(index=index,joint=(index+1)//2,
                translation_error_m=translation,rotation_error_deg=rotation,visual_quality=quality))
        report=dict(at=time.time(),hardware_executed=True,simulation=False,outcome='passed',
            kind='handeye_factory_validation',joint_state_source='command_estimate',
            reference=str(ROOT/'config/handeye-reference.json'),samples=[str(path) for path in paths],
            comparisons=comparisons,max_translation_error_m=max(c['translation_error_m'] for c in comparisons),
            max_rotation_error_deg=max(c['rotation_error_deg'] for c in comparisons),
            camera_to_mount_reference=transform.tolist(),reference_mount='arm4',
            limits=dict(translation_m=.018,rotation_deg=3.0),execution_authorized=False)
        if report['max_translation_error_m']>=.018 or report['max_rotation_error_deg']>=3.0:
            report.update(outcome='failed',reason='Fresh optical motion disagrees with saved transform')
        else:report['execution_authorized']=True
    except Exception as exc:
        report=dict(at=time.time(),hardware_executed=bool(paths),simulation=False,outcome='failed',
                    kind='handeye_factory_validation',reason=str(exc),execution_authorized=False)
    finally:
        api('control',dict(op='stop'))
        (folder/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
