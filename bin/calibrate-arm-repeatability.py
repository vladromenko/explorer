#!/usr/bin/env python3
"""Observed finite arm audit. Saves measurements; never changes calibration flags."""
import argparse,json,time,urllib.request
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from arm_commissioning import HOME,stationary_status
from arm_repeatability import plan
from arm_model import ArmModel
from handeye import visual_pose

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--execute-observed',action='store_true')
parser.add_argument('--joints',type=int,nargs='+',default=[1,2,3,4])
parser.add_argument('--approach-from-above',action='store_true',
                    help='Test a consistent final approach using two additional 2-degree steps')
parser.add_argument('--approach-degrees',type=int,choices=(2,4),default=2)
args=parser.parse_args()
if args.approach_degrees!=2 and not args.approach_from_above:parser.error('--approach-degrees requires --approach-from-above')
root=Path('/home/vlad/Explorer')
start=json.loads((root/'data/arm-state.json').read_text())['servo_deg']
if start!=HOME:raise SystemExit('Start from the observed near-home pose first')
steps=plan(start,args.joints,args.approach_from_above,args.approach_degrees)
if not args.execute_observed:
    print(json.dumps(dict(executed=False,steps=steps)));raise SystemExit(0)
base=json.loads((root/'data/status.json').read_text());stationary_status(base,time.time())
key=(root/'config/access_token').read_text().strip()
model=ArmModel();previous=start
for step in steps:
    for shape in (0.,-.2,-.4,-.6,-.8):
        if not model.path(previous[:5],step['pose'][:5],shape)['valid']:
            raise SystemExit('Planned step collides')
    previous=step['pose']
folder=root/'data/arm-repeatability'/time.strftime('%Y%m%d-%H%M%S')
folder.mkdir(parents=True,exist_ok=False)
records=[]
def api(path,body):
    request=urllib.request.Request('http://127.0.0.1:8080/api/'+path,data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(request,timeout=15))
def capture(step):
    # Wait for a new settled RGB-D exposure, not only for command acceptance.
    ready_at=time.time()+1.;deadline=ready_at+3
    frame=None
    while time.time()<deadline:
        s=json.loads((root/'data/status.json').read_text());stationary_status(s,time.time())
        if max(abs(s['raw_pose'][k]-base['raw_pose'][k]) for k in ('x','y','yaw'))>.001:
            raise ValueError('Base moved during arm audit')
        a=json.loads((root/'data/arm-state.json').read_text())
        if a['servo_deg']!=step['pose'] or a['phase']!='command_elapsed_observation_required':
            raise ValueError('Arm command changed outside audit')
        frame=dict(np.load(root/'data/rgbd-snapshot.npz',allow_pickle=False))
        if ready_at<=float(frame['stamp'])<=time.time():break
        time.sleep(.05)
    if frame is None or float(frame['stamp'])<ready_at:raise ValueError('No settled camera exposure')
    np.savez_compressed(folder/(step['capture']+'.npz'),**frame)
    records.append(dict(label=step['capture'],servo_deg=step['pose'],image_stamp=float(frame['stamp'])))
    print(json.dumps(records[-1]),flush=True)
report=dict(execution_authorized=False,calibration_applied=False,measured_joint_angles=False,
            consistent_approach_test=args.approach_from_above,
            approach_degrees=args.approach_degrees if args.approach_from_above else 0)
try:
    for step in steps:
        if 'delta' in step:
            result=api('arm/jog',dict(joint=step['joint'],delta=step['delta'],observing=True))
            if result['servo_deg']!=step['pose']:raise ValueError('Command receipt mismatch')
        else:capture(step)
    results=[]
    for joint in args.joints:
        for a,b in (('initial','from_below'),('initial','from_above'),('from_below','from_above')):
            left=dict(np.load(folder/f'j{joint}_{a}.npz',allow_pickle=False))
            right=dict(np.load(folder/f'j{joint}_{b}.npz',allow_pickle=False))
            transform,quality=visual_pose(left,right)
            results.append(dict(joint=joint,before=a,after=b,
                camera_translation_m=float(np.linalg.norm(transform[:3,3])),
                camera_rotation_deg=float(np.degrees(Rotation.from_matrix(transform[:3,:3]).magnitude())),
                quality=quality))
    report.update(results=results,command_returned_to_start=True)
except Exception as exc:report['error']=str(exc)
finally:
    report['records']=records
    (folder/'result.json').write_text(json.dumps(report,indent=2))
    api('control',dict(op='stop'))
print(json.dumps(dict(folder=str(folder),**report)),flush=True)
