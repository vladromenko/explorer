"""Planar sensor adapter. Raw inputs remain available for calibration.

Only wheel vx/vy and gravity-aligned gyro z are fused; magnetometer-derived
absolute heading and linear acceleration are deliberately not fused.
"""
import json
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from stationary_gyro import StationaryGyro

class Adapter(Node):
    def __init__(self):
        super().__init__('explorer_state_adapter')
        bias=json.loads(Path('/home/vlad/Explorer/config/imu_bias.json').read_text())
        self.bias=bias['gyro_z_bias_rad_s']
        self.stationary=StationaryGyro(self.bias)
        self.variance=max(.0004,bias['gyro_z_std_rad_s']**2)
        self.imu=self.create_publisher(Imu,'/explorer/imu_planar',10)
        self.odom=self.create_publisher(Odometry,'/explorer/wheel_odometry',10)
        self.create_subscription(Imu,'/imu/data_raw',self.imu_cb,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/odom_raw',self.odom_cb,qos_profile_sensor_data)
        self.create_subscription(Twist,'/cmd_vel',lambda m:self.stationary.command([m.linear.x,m.linear.y,m.angular.z],time.monotonic()),1)
        self.create_timer(.05,self.controller_state)

    def controller_state(self):
        try:
            status=json.loads(Path("/home/vlad/Explorer/data/status.json").read_text())
            self.stationary.controller(status,time.monotonic())
        except (OSError,ValueError,TypeError):pass
    def imu_cb(self,m):
        g=m.angular_velocity;a=m.linear_acceleration
        wz,still=self.stationary.correct([g.x,g.y,g.z],[a.x,a.y,a.z],time.monotonic())
        if wz is None:return
        out=Imu();out.header=m.header;out.header.frame_id='base_footprint'
        out.orientation_covariance[0]=-1.;out.linear_acceleration_covariance[0]=-1.
        out.angular_velocity.z=wz
        out.angular_velocity_covariance=[1e6,0.,0.,0.,1e6,0.,0.,0.,self.variance]
        self.imu.publish(out)
    def odom_cb(self,m):
        v=m.twist.twist
        self.stationary.wheel([v.linear.x,v.linear.y,v.angular.z],time.monotonic())
        m.header.frame_id='odom';m.child_frame_id='base_footprint'
        # Initial conservative wheel covariance, pending measured scale/slip tests.
        m.twist.covariance=[0.]*36
        for i,v in zip([0,7,14,21,28,35],[.0025,.0064,1e6,1e6,1e6,.01]):m.twist.covariance[i]=v
        self.odom.publish(m)

rclpy.init();node=Adapter()
try:rclpy.spin(node)
except (KeyboardInterrupt,ExternalShutdownException):pass
finally:node.destroy_node();rclpy.try_shutdown()
