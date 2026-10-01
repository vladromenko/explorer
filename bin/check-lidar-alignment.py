#!/usr/bin/env python3
"""Read-only overlap check of the two installed scanner transforms."""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

root=Path('/home/vlad/Explorer')
config=json.loads((root/'config/lidar_geometry.json').read_text())
rclpy.init();node=rclpy.create_node('explorer_lidar_alignment_check');scans={}
def receive(key,m):
    r=np.array(m.ranges);theta=m.angle_min+np.arange(len(r))*m.angle_increment
    valid=np.isfinite(r)&(r>.35)&(r<4.)
    xy=np.column_stack([r[valid]*np.cos(theta[valid]),r[valid]*np.sin(theta[valid])])
    scans[key]=(xy+np.array(config['frames'][m.header.frame_id][:2]),time.monotonic())
for key in ('scan0','scan1'):node.create_subscription(LaserScan,'/'+key,lambda m,k=key:receive(k,m),qos_profile_sensor_data)
end=time.monotonic()+8
while time.monotonic()<end and len(scans)<2:rclpy.spin_once(node,timeout_sec=.1)
if len(scans)!=2:raise SystemExit('Missing scanner')
a=scans['scan0'][0];b=scans['scan1'][0];tree=cKDTree(b)
dist,_=tree.query(a);initial=dist[dist<.15]
p=a.copy();R=np.eye(2);t=np.zeros(2)
for _ in range(25):
    d,ix=tree.query(p);keep=d<min(.15,float(np.quantile(d,.65)))
    if keep.sum()<80:raise SystemExit('Insufficient overlap')
    x=p[keep];y=b[ix[keep]];cx=x.mean(0);cy=y.mean(0)
    u,s,vt=np.linalg.svd((x-cx).T@(y-cy));rot=vt.T@u.T
    if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
    shift=cy-rot@cx;p=p@rot.T+shift;R=rot@R;t=rot@t+shift
d,_=tree.query(p);f=d[d<.15]
result=dict(at=time.time(),initial_overlap_fraction=float(len(initial)/len(a)),
            initial_median_m=float(np.median(initial)),fitted_median_m=float(np.median(f)),
            residual_translation_m=t.tolist(),residual_yaw_deg=float(np.degrees(np.arctan2(R[1,0],R[0,0]))),
            method='trimmed ICP of independently received raw scans; no config mutation',
            note='Relative alignment only; absolute body-frame orientation needs observed motion',
            raw_base_points={'scan0':a.tolist(),'scan1':b.tolist()})
p=root/'data/factory-acceptance-20260929'/('lidar-overlap-'+time.strftime('%H%M%S')+'.json')
p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result))
result.pop('raw_base_points');print(json.dumps(result))
node.destroy_node();rclpy.shutdown()
