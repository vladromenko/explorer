"""Reference chassis TF and time-aligned dual-lidar endpoint projection.

The virtual scan is an approximation: its ray origin differs from each real
scanner by about 15 cm. Keep raw scans for obstacle clearing in costmaps.
Navigation extrinsics are accepted only when the configuration contains the
physical motion evidence.  Precision manipulation remains a separate scope.
"""
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster, TransformException

ROOT=Path('/home/vlad/Explorer')

class Geometry(Node):
    def __init__(self):
        super().__init__('explorer_lidar_geometry')
        self.config=json.loads((ROOT/'config/lidar_geometry.json').read_text())
        self.buffer=Buffer(cache_time=Duration(seconds=5))
        self.listener=TransformListener(self.buffer,self)
        self.static=StaticTransformBroadcaster(self)
        transforms=[]
        frames={'base_link':[0.,0.,0.], 'base_scan':[0.,0.,.1253], **self.config['frames']}
        for child,xyz in frames.items():
            t=TransformStamped();t.header.stamp=self.get_clock().now().to_msg()
            t.header.frame_id='base_footprint';t.child_frame_id=child
            t.transform.translation.x,t.transform.translation.y,t.transform.translation.z=xyz
            t.transform.rotation.w=1.;transforms.append(t)
        self.static.sendTransform(transforms)
        self.latest={};self.used=None
        self.pub=self.create_publisher(LaserScan,'/explorer/scan',qos_profile_sensor_data)
        for key in ('scan0','scan1'):
            self.create_subscription(LaserScan,'/'+key,lambda m,k=key:self.receive(k,m),qos_profile_sensor_data)
        self.create_timer(.02,self.merge)

    def receive(self,key,m):self.latest[key]=(m,time.monotonic())

    def merge(self):
        if len(self.latest)!=2:return
        scans=[self.latest[k][0] for k in ('scan0','scan1')]
        if any(time.monotonic()-v[1]>.4 for v in self.latest.values()):return
        stamps=[Time.from_msg(s.header.stamp) for s in scans]
        key=tuple(t.nanoseconds for t in stamps)
        if key==self.used or abs(key[0]-key[1])>80_000_000:return
        target=stamps[key.index(max(key))]
        if abs(self.get_clock().now().nanoseconds-target.nanoseconds)>500_000_000:return
        ranges=np.full(720,np.inf)
        try:
            for scan,stamp in zip(scans,stamps):
                tf=self.buffer.lookup_transform_full('base_scan',target,scan.header.frame_id,stamp,'odom')
                q=tf.transform.rotation
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                r=np.asarray(scan.ranges);angles=scan.angle_min+np.arange(len(r))*scan.angle_increment+yaw
                valid=np.isfinite(r)&(r>scan.range_min)&(r<min(scan.range_max,8.))
                x=r[valid]*np.cos(angles[valid])+tf.transform.translation.x
                y=r[valid]*np.sin(angles[valid])+tf.transform.translation.y
                radius=np.hypot(x,y);bearing=np.arctan2(y,x)
                bins=np.floor((bearing+math.pi)/(2*math.pi)*720+.5).astype(int)%720
                np.minimum.at(ranges,bins,radius)
        except TransformException:return
        self.used=key
        out=LaserScan();out.header.stamp=target.to_msg();out.header.frame_id='base_scan'
        out.angle_min=-math.pi;out.angle_increment=2*math.pi/720;out.angle_max=out.angle_min+719*out.angle_increment
        out.scan_time=1/7.;out.time_increment=0.;out.range_min=.05;out.range_max=8.
        # Unknown angular bins must not become free-space clearing rays.
        ranges[~np.isfinite(ranges)]=np.nan
        out.ranges=ranges.tolist();self.pub.publish(out)
        navigation_validated=(self.config.get('navigation_validated') is True and
                              self.config.get('physically_calibrated') is True and
                              bool(self.config.get('validation',{}).get('evidence')))
        status=dict(at=time.time(),provisional=not navigation_validated,
                    navigation_validated=navigation_validated,
                    precision_manipulation_validated=self.config.get('precision_manipulation_validated') is True,
                    validation_scope=self.config.get('validation',{}).get('scope'),
                    paired_stamp_delta_ms=abs(key[0]-key[1])/1e6,valid_bins=int(np.isfinite(ranges).sum()))
        tmp=ROOT/'data/lidar_geometry.tmp';tmp.write_text(json.dumps(status));tmp.replace(ROOT/'data/lidar_geometry.json')

def main():
    rclpy.init();node=Geometry()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
