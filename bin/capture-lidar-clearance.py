#!/usr/bin/env python3
"""Read-only close-return evidence; does not mask or discard obstacles."""
import json,math,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
root=Path('/home/vlad/Explorer');frames=json.loads((root/'config/lidar_geometry.json').read_text())['frames']
rclpy.init();node=Node('explorer_clearance_audit');rows=[]
def receive(key,m):
    points=[]
    for i,r in enumerate(m.ranges):
        if math.isfinite(r) and m.range_min<r<.30:
            a=m.angle_min+i*m.angle_increment;offset=frames[m.header.frame_id]
            points.append(dict(range=r,angle_deg=math.degrees(a),base_xy_nominal=[offset[0]+r*math.cos(a),offset[1]+r*math.sin(a)]))
    rows.append(dict(at=time.time(),topic=key,points=points))
subscriptions=[]
for key in ('scan0','scan1'):
    subscriptions.append(node.create_subscription(LaserScan,'/'+key,lambda m,k=key:receive(k,m),qos_profile_sensor_data))
end=time.monotonic()+15
while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
out=root/'data'/('lidar-clearance-'+time.strftime('%Y%m%d-%H%M%S')+'.json');out.write_text(json.dumps(rows))
for key in ('scan0','scan1'):
    hits=[r for r in rows if r['topic']==key and r['points']]
    print(json.dumps(dict(topic=key,scans=sum(r['topic']==key for r in rows),close_scans=len(hits),examples=hits[:3])))
print(out);node.destroy_node();rclpy.shutdown()
