"""Explorer actuator owner. No model, image processing, or HTTP in this process."""
import json
import math
import os
from pathlib import Path
import time
import signal
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan, Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String, UInt64
from arm_msgs.msg import ArmJoints
from safety import SafetyGate
from commissioning import Pulse
import secrets

ROOT = Path(os.environ.get('EXPLORER_ROOT', '/home/vlad/Explorer'))


class Core(Node):
    def __init__(self):
        super().__init__('explorer_control')
        self.config = json.loads((ROOT/'config/commissioning.json').read_text())
        self.gate = SafetyGate(self.config)
        self.seen = {}
        self.battery = None
        self.pose = None
        self.scans = {}
        self.arm_feedback = None
        self.reason = 'STARTING'
        self.last_result = {}
        self.probe = None
        self.probe_token = None
        self.autonomy_lease = -1e9
        self.pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.arm_pub = self.create_publisher(ArmJoints, '/arm6_joints', 1)
        self.heartbeat = self.create_publisher(UInt64, '/explorer/control_heartbeat', 1)
        self.create_subscription(Float32, '/battery', self.battery_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu/data_raw', lambda m:self.touch('imu'), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom_raw', self.odom_cb, qos_profile_sensor_data)
        for name in ('scan0', 'scan1'):
            self.create_subscription(LaserScan, '/'+name, lambda m,n=name:self.scan_cb(n,m), qos_profile_sensor_data)
        self.create_subscription(ArmJoints, '/arm6_feedback', self.arm_cb, qos_profile_sensor_data)
        self.create_subscription(String, '/explorer/request', self.request, 1)
        self.create_subscription(Twist, '/explorer/nav_cmd_vel', self.nav_cb, 1)
        self.joy_held = False
        self.create_subscription(Joy, '/joy', self.joy_cb, qos_profile_sensor_data)
        self.last_tick = time.monotonic()
        self.create_timer(.02, self.tick)
        self.create_timer(.5, self.write_status)

    def touch(self, key):
        self.seen[key] = time.monotonic()

    def battery_cb(self, msg):
        self.touch('battery')
        self.battery = msg.data if math.isfinite(msg.data) else None

    def odom_cb(self, msg):
        self.touch('odom')
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = dict(x=p.x, y=p.y, yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)), frame=msg.header.frame_id)

    def scan_cb(self, name, msg):
        self.touch(name)
        valid = [r for r in msg.ranges if math.isfinite(r) and msg.range_min < r < msg.range_max]
        self.scans[name] = dict(frame=msg.header.frame_id, valid=len(valid), nearest=min(valid) if valid else None)

    def arm_cb(self, msg):
        self.touch('arm')
        self.arm_feedback = [getattr(msg, 'joint'+str(i)) for i in range(1,7)]

    def joy_cb(self, msg):
        self.touch('gamepad')
        if len(msg.buttons)<10 or len(msg.axes)<3:return
        if msg.buttons[1]:
            self.gate.stop()
            self.joy_held=False
            return
        # SDL game_controller_node standard mapping; ROS axes are positive left/forward.
        if msg.buttons[9]:
            if self.probe:self.probe.finished=True
            self.joy_held=True
            limits=self.config['max_velocity']
            self.gate.submit([msg.axes[1]*limits[0],msg.axes[0]*limits[1],msg.axes[2]*limits[2]],'manual',time.monotonic())
        elif self.joy_held:
            self.joy_held=False
            self.gate.submit([0.,0.,0.],'manual',time.monotonic())

    def nav_cb(self,msg):
        now=time.monotonic()
        if self.gate.mode=='AUTONOMOUS' and not self.gate.estop and now-self.autonomy_lease<.25:
            try:self.gate.submit([msg.linear.x,msg.linear.y,msg.angular.z],'autonomy',now)
            except ValueError:self.gate.stop()

    def request(self, msg):
        req = {}
        try:
            parsed = json.loads(msg.data)
            if not isinstance(parsed,dict):raise ValueError('Request must be an object')
            req = parsed
            op = req['op']
            if op == 'stop':
                self.gate.stop()
                if self.probe:self.probe.finished=True
            else:
                age = time.monotonic() - float(req['at'])
                if not math.isfinite(age) or age < 0 or age > .25:
                    raise ValueError('Expired request')
                if op == 'clear_stop':
                    self.gate.command_at = -1e9
                    self.gate.estop = False
                elif op == 'commission_pulse':
                    permit_path=ROOT/'data/commissioning-permit.json'
                    permit=json.loads(permit_path.read_text())
                    if time.monotonic()>permit['expires'] or not secrets.compare_digest(req['token'],permit['token']):
                        raise ValueError('No current local commissioning permit')
                    if self.gate.estop or self.probe:
                        raise ValueError('Stop is latched or a probe is active')
                    self.probe=Pulse(req['velocity'],req['duration'],time.monotonic(),req.get('quiet_seconds',0.))
                    self.probe_token=req['token']
                    permit_path.unlink()
                elif op == 'commission_lease':
                    if not self.probe or not secrets.compare_digest(req['token'],self.probe_token):
                        raise ValueError('No matching probe')
                    self.probe.last_lease=time.monotonic()
                elif op == 'autonomy_lease':
                    if self.gate.estop or self.gate.mode!='AUTONOMOUS' or not all(self.config.get(k,False) for k in ('base_commissioned','mcu_watchdog_verified','lidar_tf_validated','localization_verified')):
                        raise ValueError('Autonomy prerequisites not met')
                    self.autonomy_lease=time.monotonic()
                elif op == 'mode':
                    if req['mode'] not in ('MANUAL','ASSISTED','AUTONOMOUS'):
                        raise ValueError('Invalid mode')
                    self.gate.command_at = -1e9
                    if self.probe:self.probe.finished=True
                    self.gate.mode = req['mode']
                elif op == 'drive':
                    if self.probe:self.probe.finished=True
                    if not self.gate.submit(req['velocity'], req.get('source','manual'), time.monotonic()):
                        raise ValueError('Autonomy not selected')
                elif op == 'arm':
                    raise ValueError('Arm motion unavailable until feedback, calibrated geometry and collision checking are verified')
                else:
                    raise ValueError('Unknown operation')
            self.last_result = dict(id=req.get('id'), ok=True, op=op)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.last_result = dict(id=req.get('id'), ok=False, error=str(exc))

    def tick(self):
        now = time.monotonic()
        healthy = all(now-self.seen.get(k,-1e9) < ttl for k,ttl in [('odom',.5),('imu',.5),('scan0',.6),('scan1',.6),('battery',3.)])
        healthy = healthy and self.battery is not None and self.battery > self.config['battery_stop_voltage']
        # Conservative all-direction guard until the measured scanner extrinsics are installed.
        collision = any(s['nearest'] is None or s['nearest'] < .30 for s in self.scans.values())
        if self.gate.source=='autonomy' and now-self.autonomy_lease>.25:self.gate.command_at=-1e9
        if self.probe:
            velocity,self.reason=self.probe.tick(now,now-self.last_tick,self.gate.estop,healthy,collision)
            if velocity is not None:self.gate.output=list(velocity)
            if self.probe.finished:
                self.gate.stop()
                self.probe=None
                self.probe_token=None
        else:
            healthy = healthy and self.config['lidar_tf_validated']
            velocity, self.reason = self.gate.tick(now, now-self.last_tick, healthy, collision)
        self.last_tick = now
        if velocity is not None:
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = velocity
            self.pub.publish(msg)
        self.heartbeat.publish(UInt64(data=time.monotonic_ns()))

    def write_status(self):
        now = time.monotonic()
        status = dict(at=time.time(), mode=self.gate.mode, stop_latched=self.gate.estop,
                      reason=self.reason, battery=self.battery, raw_pose=self.pose,
                      velocity=self.gate.output, sensor_age={k:round(now-v,3) for k,v in self.seen.items()},
                      lidar=self.scans, arm_feedback=self.arm_feedback, commissioning=self.config,
                      last_request=self.last_result)
        temp=ROOT/'data/status.tmp'
        temp.write_text(json.dumps(status, allow_nan=False))
        temp.replace(ROOT/'data/status.json')


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def shutdown_signal(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,shutdown_signal)
    signal.signal(signal.SIGINT,shutdown_signal)
    node=Core()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            node.pub.publish(Twist())
            time.sleep(.02)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
