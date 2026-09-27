#!/usr/bin/env python3
"""Bounded, explicitly observed commissioning; not a boot task or learned skill."""
import argparse,json,time,urllib.request
from pathlib import Path
import numpy as np
from arm_commissioning import stationary_status
from arm_model import ArmModel
from handeye import fit
ROOT=Path('/home/vlad/Explorer')
args=argparse.ArgumentParser();args.add_argument('--execute-observed',action='store_true');opts=args.parse_args()
if not opts.execute_observed:raise SystemExit('Requires explicit observer and clear near-home envelope')
key=(ROOT/'config/access_token').read_text().strip()
def api(path,body):
    req=urllib.request.Request('http://127.0.0.1:8080/api/'+path,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
start=json.loads((ROOT/'data/arm-state.json').read_text())['servo_deg']
if start!=[92,125,3,0,90,30]:raise ValueError('This bounded survey requires its exact observed starting command')
poses=[[92,125,3,0],[102,125,3,0],[82,125,3,0],[92,115,3,0],[102,115,3,0],[82,115,3,0],
       [92,125,13,0],[102,125,13,0],[82,125,13,0],[92,125,3,10],[102,115,13,10],[82,119,9,6],[96,117,7,8]]
poses=[p+[90,30] for p in poses]
model=ArmModel();previous=start
for pose in poses+[start]:
    for shape in (0.,-.2,-.4,-.6,-.8):
        if not model.path(previous[:5],pose[:5],shape)['valid']:raise ValueError('Survey path collides')
    previous=pose
folder=ROOT/'data/handeye-surveys'/time.strftime('%Y%m%d-%H%M%S');folder.mkdir(parents=True)
paths=[];pose=list(start)
def capture(index):
    time.sleep(1.5)
    status=json.loads((ROOT/'data/status.json').read_text());stationary_status(status,time.time())
    state=json.loads((ROOT/'data/arm-state.json').read_text())
    if state['servo_deg']!=pose:raise ValueError('Pose changed outside survey')
    sample=dict(np.load(ROOT/'data/rgbd-snapshot.npz',allow_pickle=False))
    if not 0<=time.time()-float(sample['stamp'])<1.5 or float(sample['stamp'])<state['at']+.4:
        raise ValueError('Snapshot not fresh after settling')
    model.set_state(pose[:5],-.3)
    path=folder/f'{index:02d}.npz'
    np.savez_compressed(path,**sample,servo_deg=pose,base_tool=model.state.get_global_link_transform('Gripping'),
                        base_pose=np.array([status['raw_pose'][k] for k in ('x','y','yaw')]))
    paths.append(path);print(json.dumps(dict(captured=index,pose=pose)),flush=True)
def move(goal):
    for joint in range(4):
        while pose[joint]!=goal[joint]:
            delta=max(-2,min(2,goal[joint]-pose[joint]))
            if delta not in (-2,2):raise ValueError('Survey requires integer two-degree steps')
            result=api('arm/jog',dict(joint=joint+1,delta=delta,observing=True))
            pose[joint]+=delta
            if result['servo_deg']!=pose:raise ValueError('Command receipt mismatch')
            time.sleep(.15)
try:
    for i,goal in enumerate(poses):move(goal);capture(i)
    move(start)
    report=fit(paths)
except Exception as exc:
    report=dict(consistent=False,error=str(exc),execution_authorized=False)
    # An error never sends a blind recovery command or continues the sequence.
finally:
    api('control',dict(op='stop'))
report['survey']=str(folder);(folder/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
