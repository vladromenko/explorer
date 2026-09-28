#!/usr/bin/env python3
"""Read-only pose-graph audit of a saved hand-eye survey; no actuator access."""
import argparse
import json,time
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from handeye import visual_pose,mount_poses
cv2.setNumThreads(2)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('survey',type=Path)
root=parser.parse_args().survey
files=sorted(root.glob('*.npz'));samples=[dict(np.load(p,allow_pickle=False)) for p in files]
n=len(samples);train=n-2
if n<8:raise ValueError('At least eight poses required')
base=np.asarray([s['base_pose'] for s in samples])
if np.max(np.abs(base-base[0]))>.005:raise ValueError('Base moved during survey')
edges=[];initial=[np.eye(4)]
for i in range(1,n):
 candidates=sorted(set([0]+list(range(max(0,i-3),i))+[j for j in range(i) if np.max(np.abs(samples[j]['servo_deg']-samples[i]['servo_deg']))<=2]))
 matched=[]
 for j in candidates:
  try:
   h,q=visual_pose(samples[j],samples[i]);edges.append((j,i,h,q));matched.append((j,h,q))
  except ValueError:pass
 if not matched:raise ValueError('No edges for '+str(i))
 j,h,q=min(matched,key=lambda v:v[2]['depth_median_error_m'])
 initial.append(h@initial[j])
 print(json.dumps(dict(frame=i,edges=len(matched))),flush=True)
def pack(m):return np.r_[Rotation.from_matrix(m[:3,:3]).as_rotvec(),m[:3,3]]
def unpack(v):
 m=np.eye(4);m[:3,:3]=Rotation.from_rotvec(v[:3]).as_matrix();m[:3,3]=v[3:];return m
def error(h,a,b):
 e=np.linalg.inv(h)@b@np.linalg.inv(a)
 return np.r_[Rotation.from_matrix(e[:3,:3]).as_rotvec()/.01,e[:3,3]/.003]
training=[e for e in edges if e[1]<train]
def residual(x):
 poses=[np.eye(4)]+[unpack(v) for v in x.reshape(-1,6)]
 return np.concatenate([error(h,poses[a],poses[b]) for a,b,h,q in training])
solution=least_squares(residual,np.concatenate([pack(p) for p in initial[1:train]]),loss='soft_l1',max_nfev=100)
visual=[np.eye(4)]+[unpack(v) for v in solution.x.reshape(-1,6)]
for i in range(train,n):
 held=[e for e in edges if e[1]==i and e[0]<train]
 if not held:raise ValueError('Held-out sample has no training-frame reference')
 def held_residual(x):return np.concatenate([error(h,visual[a],unpack(x)) for a,b,h,q in held])
 result=least_squares(held_residual,pack(initial[i]),loss='soft_l1',max_nfev=100)
 visual.append(unpack(result.x))
g,mount_frame=mount_poses(samples)
def solve(method):
 r,t=cv2.calibrateHandEye([v[:3,:3] for v in g[:train]],[v[:3,3] for v in g[:train]],
 [v[:3,:3] for v in visual[:train]],[v[:3,3] for v in visual[:train]],method=method)
 h=np.eye(4);h[:3,:3]=r;h[:3,3]=t[:,0];return h
hand=solve(cv2.CALIB_HAND_EYE_PARK);second=solve(cv2.CALIB_HAND_EYE_TSAI)
world=[a@hand@b for a,b in zip(g,visual)];diff=[np.linalg.inv(world[0])@w for w in world]
r=dict(at=time.time(),offline_only=True,execution_authorized=False,measured_joint_positions=False,
 source=str(root),training_frames=train,held_out_indices=[train,train+1],training_edges=len(training),edges=len(edges),optimizer_success=bool(solution.success),
 camera_to_mount_reference=hand.tolist(),reference_mount=mount_frame,
 translation_residual_m=[float(np.linalg.norm(v[:3,3])) for v in diff],
 rotation_residual_deg=[float(np.degrees(Rotation.from_matrix(v[:3,:3]).magnitude())) for v in diff],
 method_disagreement_translation_m=float(np.linalg.norm((np.linalg.inv(hand)@second)[:3,3])))
r['residuals_within_limits']=max(r['translation_residual_m'])<.01 and max(r['rotation_residual_deg'])<2
r['note']='Visual graph audit only; no absolute joint calibration or execution admission'
out=root/'posegraph-audit.json';out.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
