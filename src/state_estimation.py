"""Planar sensor adapter. Raw inputs remain available for calibration.

Only wheel vx/vy and gravity-aligned gyro z are fused; magnetometer-derived
absolute heading and linear acceleration are deliberately not fused.
"""
import json
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

class Adapter(Node):
    def __init__(self):
        super().__init__('explorer_state_adapter')
        bias=json.loads(Path('/home/vlad/Explorer/config/imu_bias.json').read_text())
        self.bias=bias['gyro_z_bias_rad_s']
        self.variance=max(.0004,bias['gyro_z_std_rad_s']**2)
        self.imu=self.create_publisher(Imu,'/explorer/imu_planar',10)
        self.odom=self.create_publisher(Odometry,'/explorer/wheel_odometry',10)
        self.create_subscription(Imu,'/imu/data_raw',self.imu_cb,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/odom_raw',self.odom_cb,qos_profile_sensor_data)
    def imu_cb(self,m):
        out=Imu();out.header=m.header;out.header.frame_id='base_footprint'
        out.orientation_covariance[0]=-1.;out.linear_acceleration_covariance[0]=-1.
        out.angular_velocity.z=m.angular_velocity.z-self.bias
        out.angular_velocity_covariance=[1e6,0.,0.,0.,1e6,0.,0.,0.,self.variance]
        self.imu.publish(out)
    def odom_cb(self,m):
        m.header.frame_id='odom';m.child_frame_id='base_footprint'
        # Initial conservative wheel covariance, pending measured scale/slip tests.
        m.twist.covariance=[0.]*36
        for i,v in zip([0,7,14,21,28,35],[.0025,.0064,1e6,1e6,1e6,.01]):m.twist.covariance[i]=v
        self.odom.publish(m)

rclpy.init();node=Adapter()
try:rclpy.spin(node)
finally:node.destroy_node();rclpy.shutdown()
