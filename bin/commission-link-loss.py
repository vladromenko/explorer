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
import signal
import subprocess
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
parser.add_argument('--power-cut-observer',action='store_true',required=True)
parser.set_defaults(axis='forward')
parser.add_argument('--extended-reverse',action='store_true')
parser.add_argument('--settled-speed',action='store_true',help='Wait for a full second of commands before pausing transport')
parser.set_defaults(timeout_test=False)
args=parser.parse_args()
if args.timeout_test and args.axis!='forward':parser.error('Timeout test only permits forward')
velocity={'forward':[.04,0.,0.],'left':[0.,.04,0.],'ccw':[0.,0.,.15]}[args.axis]
if args.extended_reverse:velocity=[-.025,0.,0.]
restore_seconds=2.6 if args.extended_reverse else 1.4
duration=.6
freeze_after=.35
if args.settled_speed:
 freeze_after=1.0
 duration=1.3
 restore_seconds=2.2
pid=int(subprocess.check_output(['systemctl','--user','show','explorer-mcu.service','-p','MainPID','--value']))
if pid<=1 or 'micro_ros_agent' not in str(Path(f'/proc/{pid}/exe').resolve()) or Path(f'/proc/{pid}').stat().st_uid!=os.getuid():
 raise RuntimeError('Expected the running user-owned MCU agent')
frozen_at=None;resume_at=None;rescue=None
quiet_seconds=.8 if args.timeout_test else 0.
rclpy.init();node=Node('explorer_commission_base');pub=node.create_publisher(String,'/explorer/request',10)
samples=[];started=time.monotonic();command_started=None
def odom(m):
 p=m.pose.pose.position;q=m.pose.pose.orientation;t=m.twist.twist
 stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
 samples.append(dict(t=time.monotonic()-started,at=time.time(),source_stamp=stamp,source_age=time.time()-stamp,kind='odom',x=p.x,y=p.y,yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),vx=t.linear.x,vy=t.linear.y,wz=t.angular.z))
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
 # Independent session restores transport even if the SSH probe process dies.
 rescue_code='import os,signal,sys,time\nprint("ready",flush=True)\ntime.sleep(max(0,float(sys.argv[2])-time.monotonic()))\ntry: os.kill(int(sys.argv[1]),signal.SIGCONT)\nexcept ProcessLookupError: pass'
 rescue=subprocess.Popen(['/usr/bin/python3','-c',rescue_code,str(pid),str(command_started+restore_seconds)],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,text=True)
 if rescue.stdout.readline().strip()!='ready':raise RuntimeError('Recovery timer unavailable')
 request('commission_pulse',token=token,velocity=velocity,duration=duration,quiet_seconds=quiet_seconds)
 while time.monotonic()-command_started<restore_seconds+.25:
  spin(.025)
  elapsed=time.monotonic()-command_started
  if frozen_at is None and elapsed>=freeze_after:
   if elapsed>freeze_after+.15:raise RuntimeError('Probe timing slipped before transport interruption')
   frozen_at=time.monotonic();os.kill(pid,signal.SIGSTOP)
   print(json.dumps(dict(event='transport_paused',at=time.time(),elapsed=elapsed)),flush=True)
  if frozen_at is not None and elapsed>=restore_seconds and resume_at is None:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass
   resume_at=time.monotonic()
   print(json.dumps(dict(event='transport_restored',at=time.time(),elapsed=elapsed)),flush=True)
  request('commission_lease',token=token)
 request('stop');spin(8.0 if args.settled_speed else 1.2)
 before=[s for s in samples if s['kind']=='odom' and s['t']<command_started-started][-1]
 after=[s for s in samples if s['kind']=='odom'][-1]
 final_age=time.time()-after['source_stamp']
 final_fresh=0<=final_age<.5
 commands=[s for s in samples if s['kind']=='command' and s['t']>=command_started-started]
 result=dict(axis=args.axis,requested=velocity,duration=duration,quiet_seconds=quiet_seconds,delta={k:after[k]-before[k] for k in ('x','y','yaw')},
             peak={k:max(abs(s[k]) for s in commands) for k in ('vx','vy','wz')},
             final_velocity={k:after[k] for k in ('vx','vy','wz')} if final_fresh else None,final_telemetry_age_s=final_age,final_telemetry_fresh=final_fresh,physical_direction_confirmed=False,transport_pause_s=resume_at-frozen_at,extended_reverse=args.extended_reverse,hardware_stop_verified=False,observer_analysis_required=True)
 result['agent_pid_before']=pid
 result['agent_pid_after']=int(subprocess.check_output(['systemctl','--user','show','explorer-mcu.service','-p','MainPID','--value']))
 print(json.dumps(result))
finally:
 try:os.kill(pid,signal.SIGCONT)
 except ProcessLookupError:pass
 request('stop');spin(.2)
 permit.unlink(missing_ok=True)
 path=root/'data'/('link-loss-probe-'+('reverse-extended' if args.extended_reverse else args.axis)+'-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
 path.write_text(json.dumps(dict(result=result,samples=samples),indent=2))
 node.destroy_node();rclpy.shutdown()
