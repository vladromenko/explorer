#!/usr/bin/env python3
"""Execute ONE physically supervised bounded pulse and capture its telemetry.

Does not set any commissioning completion flag. Observer confirmation is required.
"""
import argparse
import json
import math
import os
from pathlib import Path
import secrets
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

root=Path('/home/vlad/Explorer')
parser=argparse.ArgumentParser()
parser.add_argument('axis',choices=['forward','left','ccw'])
args=parser.parse_args()
velocity={'forward':[.04,0.,0.],'left':[0.,.04,0.],'ccw':[0.,0.,.15]}[args.axis]
duration=.6
rclpy.init();node=Node('explorer_commission_base');pub=node.create_publisher(String,'/explorer/request',10)
samples=[];started=time.monotonic();command_started=None
def odom(m):
 p=m.pose.pose.position;q=m.pose.pose.orientation;t=m.twist.twist
 samples.append(dict(t=time.monotonic()-started,kind='odom',x=p.x,y=p.y,yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),vx=t.linear.x,vy=t.linear.y,wz=t.angular.z))
def imu(m):samples.append(dict(t=time.monotonic()-started,kind='imu',wz=m.angular_velocity.z))
def cmd(m):samples.append(dict(t=time.monotonic()-started,kind='command',vx=m.linear.x,vy=m.linear.y,wz=m.angular.z))
node.create_subscription(Odometry,'/odom_raw',odom,qos_profile_sensor_data)
node.create_subscription(Imu,'/imu/data_raw',imu,qos_profile_sensor_data)
node.create_subscription(Twist,'/cmd_vel',cmd,10)
def spin(seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.01)
def request(op,**data):
 request_id=str(uuid.uuid4())
 pub.publish(String(data=json.dumps(dict(op=op,id=request_id,at=time.monotonic(),**data))))
 return request_id
token=secrets.token_urlsafe(24);permit=root/'data/commissioning-permit.json'
result={}
try:
 discovery_deadline=time.monotonic()+10
 while not pub.get_subscription_count() and time.monotonic()<discovery_deadline:spin(.05)
 if not pub.get_subscription_count():raise RuntimeError('Controller request subscription unavailable')
 sensor_deadline=time.monotonic()+10
 while time.monotonic()<sensor_deadline:
  spin(.05)
  od=[s for s in samples if s['kind']=='odom']
  if len(od)>=5:break
 state=json.loads((root/'data/status.json').read_text())
 if time.time()-state['at']>1 or not state['stop_latched']:raise RuntimeError('Expected fresh, stop-latched controller')
 if any(state['sensor_age'].get(k,99)>limit for k,limit in [('imu',.3),('odom',.3),('scan0',.4),('scan1',.4),('battery',2)]):raise RuntimeError('Sensors are stale')
 if any(v.get('nearest',0) is None or v['nearest']<.30 for v in state['lidar'].values()):raise RuntimeError('Insufficient scanner clearance')
 od=[s for s in samples if s['kind']=='odom']
 if len(od)<5:raise RuntimeError('Probe odometry subscription has no fresh baseline')
 if any(abs(v)> .02 for s in od[-5:] for v in (s['vx'],s['vy'],s['wz'])):raise RuntimeError('Base is not stationary')
 clear_id=request('clear_stop');ack_deadline=time.monotonic()+2
 while time.monotonic()<ack_deadline:
  spin(.05)
  state=json.loads((root/'data/status.json').read_text())
  if state.get('last_request',{}).get('id')==clear_id:break
 if state.get('last_request',{}).get('id')!=clear_id or not state['last_request'].get('ok') or state['stop_latched']:
  raise RuntimeError('Controller did not acknowledge clearing stop: '+str(state.get('last_request')))
 fd=os.open(permit,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
 with os.fdopen(fd,'w') as f:json.dump(dict(token=token,expires=time.monotonic()+2),f)
 command_started=time.monotonic()
 request('commission_pulse',token=token,velocity=velocity,duration=duration)
 while time.monotonic()-command_started<duration:
  spin(.045)
  request('commission_lease',token=token)
 request('stop');spin(1.2)
 before=[s for s in samples if s['kind']=='odom' and s['t']<command_started-started][-1]
 after=[s for s in samples if s['kind']=='odom'][-1]
 commands=[s for s in samples if s['kind']=='command' and s['t']>=command_started-started]
 result=dict(axis=args.axis,requested=velocity,duration=duration,delta={k:after[k]-before[k] for k in ('x','y','yaw')},
             peak={k:max(abs(s[k]) for s in commands) for k in ('vx','vy','wz')},
             final_velocity={k:after[k] for k in ('vx','vy','wz')},physical_direction_confirmed=False)
 print(json.dumps(result))
finally:
 request('stop');spin(.2)
 permit.unlink(missing_ok=True)
 path=root/'data'/('base-probe-'+args.axis+'-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
 path.write_text(json.dumps(dict(result=result,samples=samples),indent=2))
 node.destroy_node();rclpy.shutdown()
