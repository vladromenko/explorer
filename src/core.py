"""Explorer actuator owner. No model, image processing, or HTTP in this process."""
import json
import math
import os
from pathlib import Path
import time
import signal
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan, Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String, UInt64, ColorRGBA
from arm_msgs.msg import ArmJoints
from safety import SafetyGate
from commissioning import Pulse
from battery_gauge import battery_summary
from power_policy import PowerPolicy
import secrets

ROOT = Path(os.environ.get('EXPLORER_ROOT', '/home/vlad/Explorer'))


class Core(Node):
    def __init__(self):
        super().__init__('explorer_control')
        self.config = json.loads((ROOT/'config/commissioning.json').read_text())
        self.gate = SafetyGate(self.config)
        self.power_policy = PowerPolicy(yaml.safe_load((ROOT/'config/power.yaml').read_text()))
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.critical_saved = False
        try:
            latch=json.loads((ROOT/'data/power-critical-latch.json').read_text())
            self.power_policy.critical=latch.get('boot_id')==self.boot_id
            self.critical_saved=self.power_policy.critical
        except FileNotFoundError:pass
        except (OSError,ValueError):self.power_policy.critical=True
        self.power_state = self.power_policy.evaluate(time.monotonic())
        self.charging = None
        self.create_timer(.5, self.read_power_interlock)
        self.seen = {}
        self.battery = None
        self.pose = None
        self.scans = {}
        self.arm_feedback = None
        self.arm_active_until = 0.
        self.reason = 'STARTING'
        self.last_result = {}
        self.probe = None
        self.probe_token = None
        self.autonomy_lease = -1e9
        self.pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.arm_pub = self.create_publisher(ArmJoints, '/arm6_joints', 1)
        self.rgb_pub = self.create_publisher(ColorRGBA, '/rgb', 1)
        self.heartbeat = self.create_publisher(UInt64, '/explorer/control_heartbeat', 1)
        self.request_ack = self.create_publisher(String, '/explorer/request_ack', 10)
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
        # Vendor ColorRGBA.a selects an effect: 100 = off, 101 = changing colours.
        self.create_timer(10., self.refresh_lights)

    def refresh_lights(self):
        try:
            effect=json.loads((ROOT/'config/appearance.json').read_text())['rgb_effect']
            if type(effect) is not int or not 100<=effect<=107:effect=100
        except (OSError,ValueError,KeyError,TypeError):effect=100
        self.rgb_pub.publish(ColorRGBA(a=float(effect)))

    def touch(self, key):
        self.seen[key] = time.monotonic()

    def read_power_interlock(self):
        try:
            self.charging = json.loads((ROOT/'data/charging-interlock.json').read_text()).get('connected')
            if self.charging is not None and type(self.charging) is not bool:self.charging=True
        except FileNotFoundError:self.charging = None
        except (OSError, ValueError):self.charging = True  # malformed interlock fails closed

    def battery_cb(self, msg):
        self.touch('battery')
        self.battery = msg.data if math.isfinite(msg.data) else None
        self.power_policy.sample(self.battery, time.monotonic())

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
        if not self.config.get('gamepad_commissioned',False):return
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
                self.pub.publish(Twist())
            else:
                age = time.monotonic() - float(req['at'])
                if not math.isfinite(age) or age < 0 or age > .25:
                    raise ValueError('Expired request')
                if op == 'clear_stop':
                    if not self.power_state['motion_allowed']:raise ValueError('Power state prohibits arming')
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
                    raise ValueError('General arm execution is not commissioned. Supervised near-home probes use bin/commission-arm.py; servo feedback is unavailable.')
                else:
                    raise ValueError('Unknown operation')
            self.last_result = dict(id=req.get('id'), ok=True, op=op)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.last_result = dict(id=req.get('id'), ok=False, error=str(exc))
        self.request_ack.publish(String(data=json.dumps(dict(self.last_result,
            handled_monotonic=time.monotonic(),mcu_acknowledged=False))))

    def tick(self):
        now = time.monotonic()
        arm_link_ok=all(now-self.seen.get(k,-1e9)<ttl for k,ttl in [('odom',.5),('battery',2.)])
        if getattr(self,'arm_link_was_healthy',False) and not arm_link_ok:
            try:
                fault=ROOT/'data/arm-telemetry-fault.tmp'
                fault.write_text(json.dumps(dict(at=time.time(),reason='MCU telemetry interrupted')))
                fault.replace(ROOT/'data/arm-telemetry-fault.json')
            except OSError:self.gate.stop()
        self.arm_link_was_healthy=arm_link_ok
        self.power_state = self.power_policy.evaluate(now, active=now<self.arm_active_until or any(abs(v)>.001 for v in self.gate.output), charging=self.charging)
        if not self.power_state['motion_allowed']:
            self.gate.stop()
            if self.probe:self.probe.finished=True
        scale = self.power_state['speed_scale']
        self.gate.command = [max(-m*scale, min(m*scale, v)) for m,v in zip(self.config['max_velocity'],self.gate.command)]
        healthy = all(now-self.seen.get(k,-1e9) < ttl for k,ttl in [('odom',.5),('imu',.5),('scan0',.6),('scan1',.6),('battery',3.)])
        healthy = healthy and self.power_state['motion_allowed'] and self.battery is not None and self.battery > self.config['battery_stop_voltage']
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
        try:
            arm_state=json.loads((ROOT/'data/arm-state.json').read_text())
            if arm_state.get('boot_id')!=self.boot_id:arm_state=dict(phase='UNKNOWN_AFTER_REBOOT',measured=False)
        except (OSError,ValueError):arm_state=dict(phase='UNKNOWN',measured=False)
        if arm_state.get('phase')=='command_in_progress':
            self.arm_active_until=min(now+5.,float(arm_state.get('ends_monotonic',0.)))
        if self.power_policy.critical and not self.critical_saved:
            latch=ROOT/'data/power-critical-latch.tmp'
            with latch.open('w') as stream:
                json.dump(dict(boot_id=self.boot_id,at=time.time()),stream)
                stream.flush();os.fsync(stream.fileno())
            latch.replace(ROOT/'data/power-critical-latch.json')
            self.critical_saved=True
        status = dict(at=time.time(), mode=self.gate.mode, stop_latched=self.gate.estop,
                      reason=self.reason, battery=self.battery, raw_pose=self.pose,
                      battery_gauge=battery_summary(self.power_policy.filtered,now-self.seen.get('battery',-1e9)),
                      power=self.power_state,
                      velocity=self.gate.output, sensor_age={k:round(now-v,3) for k,v in self.seen.items()},
                      lidar=self.scans, arm_feedback=self.arm_feedback, commissioning=self.config,
                      arm_command_state=arm_state,
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
