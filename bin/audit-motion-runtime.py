#!/usr/bin/env python3
"""Read-only graph and timing capture. No publishers, services or actuator IO."""
import argparse
import collections
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32, UInt64
from arm_msgs.msg import ArmJoints

parser=argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=20)
parser.add_argument('--output', required=True)
args=parser.parse_args()
if not 5 <= args.seconds <= 300:raise ValueError('Capture must be 5..300 seconds')
rclpy.init();node=Node('explorer_readonly_motion_audit')
samples=collections.defaultdict(list);subscriptions=[]
types={'/cmd_vel':Twist,'/explorer/nav_cmd_vel':TwistStamped,'/odom_raw':Odometry,
       '/imu/data_raw':Imu,'/scan0':LaserScan,'/scan1':LaserScan,'/battery':Float32,
       '/explorer/control_heartbeat':UInt64,'/arm6_feedback':ArmJoints}
def receive(topic, msg, info):
    sample=dict(monotonic=time.monotonic(),publisher_gid=bytes(info.get('publisher_gid',[])).hex() or 'unavailable_in_rclpy')
    if isinstance(msg,Twist):sample['velocity']=[msg.linear.x,msg.linear.y,msg.angular.z]
    if isinstance(msg,TwistStamped):sample['velocity']=[msg.twist.linear.x,msg.twist.linear.y,msg.twist.angular.z]
    if hasattr(msg,'header'):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        sample['stamp']=stamp;sample['stamp_age_s']=time.time()-stamp
    if isinstance(msg,ArmJoints):sample['joints']=[getattr(msg,'joint'+str(i)) for i in range(1,7)]
    samples[topic].append(sample)
def callback(topic):
    def cb(msg, info):receive(topic,msg,info)
    return cb
for topic,typ in types.items():
    subscriptions.append(node.create_subscription(typ,topic,callback(topic),qos_profile_sensor_data))
started=time.monotonic()
while time.monotonic()-started<args.seconds:rclpy.spin_once(node,timeout_sec=.02)
graph={};summary={}
for topic in types:
    graph[topic]=[dict(node=e.node_namespace.rstrip('/')+'/'+e.node_name,
        gid=bytes(e.endpoint_gid).hex(),qos=str(e.qos_profile)) for e in node.get_publishers_info_by_topic(topic)]
    values=samples[topic];gaps=[b['monotonic']-a['monotonic'] for a,b in zip(values,values[1:])]
    summary[topic]=dict(count=len(values),max_gap_s=max(gaps,default=None),
        publishers=dict(collections.Counter(v['publisher_gid'] for v in values)),
        nonzero=sum(any(v.get('velocity',[])) for v in values))
nodes={ns.rstrip('/')+'/'+name:node.get_publisher_names_and_types_by_node(name,ns)
       for name,ns in node.get_node_names_and_namespaces()}
result=dict(at=time.time(),duration_s=time.monotonic()-started,graph=graph,nodes=nodes,summary=summary,samples=samples)
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result))
print(json.dumps(dict(summary=summary,graph=graph),indent=2))
node.destroy_node();rclpy.shutdown()
