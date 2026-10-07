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
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from rclpy.duration import Duration
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Imu, LaserScan, Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String, UInt64, ColorRGBA
from arm_msgs.msg import ArmJoints
from safety import SafetyGate
from commissioning import Pulse
from battery_gauge import battery_summary
from power_policy import PowerPolicy
import secrets
import hashlib
from motion_session import MotionSession
from transition_log import TransitionLog
from source_freshness import SourceFreshness
from lidar_observation import summarize as summarize_scan
from controller_control import BaseTransport
from factory_zero_cadence import StationaryZeroCadence
from appearance import read as read_appearance,messages as appearance_messages

ROOT = Path(os.environ.get('EXPLORER_ROOT', '/home/vlad/Explorer'))


def publish_stop(node):
    """Stop both transports, including when the ROS executor is shutting down."""
    native = getattr(node, 'native', None)
    if native:
        native.stop(time.monotonic_ns())
    node.pub.publish(Twist())


def publish_hold(node):
    """Stop base motion immediately while preserving an active mission session."""
    native = getattr(node, 'native', None)
    if native:
        now_ns = time.monotonic_ns()
        native.velocity([0., 0., 0.], now_ns, now_ns)
    else:
        node.pub.publish(Twist())


class Core(Node):
    def __init__(self):
        super().__init__('explorer_control')
        self.config = json.loads((ROOT/'config/commissioning.json').read_text())
        self.gate = SafetyGate(self.config)
        self.config_sha256=hashlib.sha256((ROOT/'config/commissioning.json').read_bytes()).hexdigest()
        self.create_timer(2.,self.refresh_graduated_capabilities)
        self.events=TransitionLog(ROOT/'data/motion-transitions.jsonl')
        self.session=MotionSession()
        self.nav_not_before=0.
        self.nav_last_stamp=0.
        self.hold_requested=False
        self.stationary_since=None
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
        self.source_freshness=SourceFreshness()
        self.battery = None
        self.pose = None
        self.odom_velocity = None
        self.scans = {}
        self.arm_feedback = None
        self.arm_measurements = None
        self.arm_feedback_transport=dict(publishers=None,checked_monotonic=None)
        self.create_timer(5.,self.inspect_arm_transport)
        self.arm_active_until = 0.
        self.reason = 'STARTING'
        self.last_result = {}
        self.probe = None
        self.probe_id=None
        self.last_probe_result=None
        self.probe_token = None
        self.autonomy_lease = -1e9
        self.pub = self.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=1,lifespan=Duration(seconds=.15)))
        self.native = None
        self.factory_zero_cadence = None
        try:
            factory=json.loads((ROOT/'config/factory-runtime.json').read_text())
            if factory.get('idle_zero_cadence') is True and not (ROOT/'config/controller-profile.json').exists():
                self.factory_zero_cadence=StationaryZeroCadence()
        except (OSError,ValueError):pass
        if (ROOT/'config/controller-profile.json').exists():
            self.native_pub = self.create_publisher(String, '/explorer/controller_request', 10)
            self.native = BaseTransport(lambda request:self.native_pub.publish(String(data=json.dumps(request))))
            self.create_subscription(String, '/explorer/controller_state',
                lambda message:self.native.observe(json.loads(message.data), time.monotonic_ns()), 10)
        self.arm_pub = self.create_publisher(ArmJoints, '/arm6_joints', 1)
        self.rgb_pub = self.create_publisher(ColorRGBA, '/rgb', 1)
        self.heartbeat = self.create_publisher(UInt64, '/explorer/control_heartbeat', 1)
        self.request_ack = self.create_publisher(String, '/explorer/request_ack', 10)
        self.create_subscription(Float32, '/battery', self.battery_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu/data_raw', lambda m:self.accept_sensor('imu',m,.5), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom_raw', self.odom_cb, qos_profile_sensor_data)
        for name in ('scan0', 'scan1'):
            self.create_subscription(LaserScan, '/'+name, lambda m,n=name:self.scan_cb(n,m), qos_profile_sensor_data)
        self.create_subscription(ArmJoints, '/arm6_feedback', self.arm_cb, qos_profile_sensor_data)
        self.create_subscription(String, '/explorer/arm_measurements', self.arm_measurement_cb, 10)
        self.create_subscription(String, '/explorer/request', self.request, 10)
        self.create_subscription(TwistStamped, '/explorer/nav_cmd_vel', self.nav_cb, 1)
        self.joy_held = False
        self.create_subscription(Joy, '/joy', self.joy_cb, qos_profile_sensor_data)
        self.last_tick = time.monotonic()
        self.create_timer(.02, self.tick)
        self.create_timer(.5, self.write_status)
        # Core is the only /rgb publisher. Vendor a=100..106 selects effects;
        # a=255 applies an RGB colour to every WS2812 LED.
        self.last_light_key=None;self.last_light_refresh=-1e9
        self.create_timer(.25,self.refresh_lights)

    def refresh_graduated_capabilities(self):
        """Hot-apply only evidence-derived capability flags.

        All motion limits and hardware commissioning fields remain immutable for
        this process.  This prevents the curriculum service from becoming a
        general configuration bypass.
        """
        keys=('localization_verified','gripper_calibrated','visual_closed_loop_arm_verified',
              'learned_policy_verified','autonomous_delivery_verified')
        try:
            raw=(ROOT/'config/commissioning.json').read_bytes();candidate=json.loads(raw)
            old_other={k:v for k,v in self.config.items() if k not in keys}
            new_other={k:v for k,v in candidate.items() if k not in keys}
            if old_other!=new_other:return
            for key in keys:self.config[key]=candidate.get(key,False) is True
            self.config_sha256=hashlib.sha256(raw).hexdigest()
        except (OSError,ValueError,TypeError):pass

    def snapshot(self):
        return dict(mode=self.gate.mode,stop_latched=self.gate.estop,
                    mission=self.session.mission,held=self.session.held,velocity=list(self.gate.output),
                    command_source=self.gate.source,command_at=self.gate.command_at,
                    lease_at=self.session.lease_at)

    def event(self, initiator, reason, before):
        now=time.monotonic()
        self.events.emit(initiator,reason,before,self.snapshot(),
            command_source=self.gate.source,command_age_s=now-self.gate.command_at,
            lease_age_s=now-self.session.lease_at,mission_id=self.session.mission,
            sensor_age={k:now-v for k,v in self.seen.items()},observed_velocity=self.odom_velocity,
            power=self.power_state,obstacle=self.scans,config_sha256=self.config_sha256)

    def emergency(self, initiator, reason):
        before=self.snapshot()
        native = getattr(self, 'native', None)
        if native and not self.gate.estop:
            native.stop(time.monotonic_ns())
        self.gate.stop();self.session.end();self.autonomy_lease=-1e9;self.joy_held=False
        self.hold_requested=False
        if self.probe:self.probe.cancel(reason)
        if before!=self.snapshot():self.event(initiator,reason,before)

    def refresh_lights(self):
        now=time.monotonic();config=read_appearance(ROOT)
        status=dict(power_state=self.power_state.get('state','UNKNOWN'),velocity=self.gate.output,
                    mission=self.session.mission,arm_active=now<self.arm_active_until)
        mode,frames=appearance_messages(config,status,now)
        key=(mode,tuple(tuple(sorted(frame.items())) for frame in frames))
        if key==self.last_light_key and now-self.last_light_refresh<8:return
        for frame in frames:self.rgb_pub.publish(ColorRGBA(**frame))
        self.last_light_key=key;self.last_light_refresh=now

    def touch(self, key):
        self.seen[key] = time.monotonic()

    def accept_sensor(self,key,msg,max_age):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if self.source_freshness.accept(key,stamp,self.get_clock().now().nanoseconds/1e9,max_age):
            self.touch(key)
            return True
        return False

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
        if not self.accept_sensor('odom',msg,.5):return
        t=msg.twist.twist
        self.odom_velocity=[t.linear.x,t.linear.y,t.angular.z]
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = dict(x=p.x, y=p.y, yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)), frame=msg.header.frame_id)

    def scan_cb(self, name, msg):
        if not self.accept_sensor(name,msg,.6):return
        self.scans[name] = summarize_scan(msg)

    def arm_cb(self, msg):
        self.touch('arm')
        self.arm_feedback = [getattr(msg, 'joint'+str(i)) for i in range(1,7)]
        self.arm_feedback_transport.update(received_monotonic=time.monotonic(),time_field=msg.time,
                                          acquisition_time_known=False)

    def inspect_arm_transport(self):
        publishers=self.get_publishers_info_by_topic('/arm6_feedback')
        self.arm_feedback_transport.update(publishers=[dict(node=p.node_namespace.rstrip('/')+'/'+p.node_name,
            type=p.topic_type,qos=str(p.qos_profile)) for p in publishers],checked_monotonic=time.monotonic())

    def arm_measurement_cb(self, msg):
        data = json.loads(msg.data)
        self.arm_measurements = data
        if data.get('all_fresh') or data.get('reference_valid') and data.get('estimated'):
            self.touch('arm')
        # Preserve signed measurements and raw evidence separately from the
        # legacy command-angle array; never make old callers infer calibration.

    def joy_cb(self, msg):
        self.touch('gamepad')
        if not self.config.get('gamepad_commissioned',False):return
        if len(msg.buttons)<10 or len(msg.axes)<3:return
        if msg.buttons[1]:
            self.emergency('gamepad','OPERATOR_ESTOP')
            return
        # SDL game_controller_node standard mapping; ROS axes are positive left/forward.
        if msg.buttons[9]:
            if self.session.mission:
                before=self.snapshot();self.session.end();self.event('gamepad','MANUAL_TAKEOVER',before)
            if self.probe:self.probe.cancel('MANUAL_TAKEOVER')
            self.joy_held=True
            limits=self.config['max_velocity']
            self.gate.submit([msg.axes[1]*limits[0],msg.axes[0]*limits[1],msg.axes[2]*limits[2]],'manual',time.monotonic())
        elif self.joy_held:
            self.joy_held=False
            self.gate.submit([0.,0.,0.],'manual',time.monotonic())

    def nav_cb(self,msg):
        now=time.monotonic()
        if getattr(self,"policy_mission",None)==self.session.mission and self.session.mission is not None:return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        age=self.get_clock().now().nanoseconds/1e9-stamp
        if stamp<self.nav_not_before or stamp<=self.nav_last_stamp or not -.02<=age<.25:return
        self.nav_last_stamp=stamp
        if self.gate.mode=='AUTONOMOUS' and not self.gate.estop and self.session.permits(now):
            try:self.gate.submit([msg.twist.linear.x,msg.twist.linear.y,msg.twist.angular.z],'autonomy',now-max(0.,age))
            except ValueError:self.emergency('nav2','INVALID_VELOCITY')

    def request(self, msg):
        req = {}
        try:
            parsed = json.loads(msg.data)
            if not isinstance(parsed,dict):raise ValueError('Request must be an object')
            req = parsed
            op = req['op']
            before=self.snapshot()
            if op in ('stop','estop'):
                publish_stop(self)
                self.emergency(req.get('initiator','request'),'OPERATOR_ESTOP')
            else:
                age = time.monotonic() - float(req['at'])
                if not math.isfinite(age) or age < 0 or age > .25:
                    raise ValueError('Expired request')
                if op == 'clear_stop':
                    if not self.power_state['motion_allowed']:raise ValueError('Power state prohibits arming')
                    native = getattr(self, 'native', None)
                    if native:native.start(time.monotonic_ns())
                    self.gate.command_at = -1e9
                    self.gate.estop = False
                elif op == 'commission_pulse':
                    if self.gate.mode!='MANUAL' or self.session.mission:
                        raise ValueError('Observed finite motion requires manual mode without a mission')
                    permit_path=ROOT/'data/commissioning-permit.json'
                    permit=json.loads(permit_path.read_text())
                    if time.monotonic()>permit['expires'] or not secrets.compare_digest(req['token'],permit['token']):
                        raise ValueError('No current local commissioning permit')
                    if self.gate.estop or self.probe:
                        raise ValueError('Stop is latched or a probe is active')
                    self.probe=Pulse(req['velocity'],req['duration'],float(req['at']),req.get('quiet_seconds',0.))
                    self.probe_id=req.get('id')
                    self.probe_token=req['token']
                    permit_path.unlink()
                elif op == 'commission_lease':
                    if not self.probe or not secrets.compare_digest(req['token'],self.probe_token):
                        raise ValueError('No matching probe')
                    self.probe.last_lease=float(req['at'])
                elif op in ('begin_mission','autonomy_lease','resume_base'):
                    if self.gate.estop or self.gate.mode!='AUTONOMOUS' or not all(self.config.get(k,False) for k in ('base_commissioned','mcu_watchdog_verified','lidar_tf_validated','localization_verified')):
                        raise ValueError('Autonomy prerequisites not met')
                    if op=='begin_mission':
                        self.session.begin(req['mission'],time.monotonic())
                        self.policy_mission=req['mission'] if req.get('kind')=='mobile_policy' else None
                    elif op=='resume_base':
                        self.session.resume(req['mission'],time.monotonic());self.hold_requested=False
                        self.nav_not_before=self.get_clock().now().nanoseconds/1e9
                    else:self.session.renew(req['mission'],time.monotonic())
                    self.autonomy_lease=time.monotonic()
                elif op in ('hold_base','cancel_mission','finish_mission'):
                    mission=req.get('mission')
                    if mission is not None:
                        if op=='hold_base':self.session.hold(mission)
                        else:self.session.end(mission)
                    elif self.session.mission:
                        raise ValueError('Mission identity required')
                    self.gate.hold()
                    if not self.hold_requested:self.stationary_since=None
                    self.hold_requested=True
                    if self.probe:self.probe.cancel('HOLD_REQUESTED')
                    publish_hold(self)
                elif op=='manual_release':
                    # A disconnected/idle panel cannot cancel local autonomy.
                    if self.gate.mode=='MANUAL' and not self.session.mission:
                        self.gate.hold();self.joy_held=False
                        # Manual base/arm operation is sequential: releasing the
                        # chassis establishes the stationary hold needed by the arm.
                        self.hold_requested=True;self.stationary_since=None
                        publish_hold(self)
                elif op == 'mode':
                    if req['mode'] not in ('MANUAL','ASSISTED','AUTONOMOUS'):
                        raise ValueError('Invalid mode')
                    self.session.end();self.gate.hold();self.hold_requested=False
                    self.joy_held=False
                    if self.probe:self.probe.cancel('MODE_CHANGED')
                    self.gate.mode = req['mode']
                elif op == 'drive':
                    if req.get('source','manual')!='manual':
                        raise ValueError('Autonomy velocity must come from the local navigator')
                    self.session.end();self.hold_requested=False
                    if self.probe:self.probe.cancel('MANUAL_TAKEOVER')
                    if not self.gate.submit(req['velocity'], req.get('source','manual'), float(req['at'])):
                        raise ValueError('Autonomy not selected')
                elif op == 'policy_drive':
                    now=time.monotonic()
                    if self.gate.mode!="AUTONOMOUS" or self.gate.estop or not self.config.get("localization_verified"):
                        raise ValueError("Автономная локализация не принята")
                    if req.get("mission")!=getattr(self,"policy_mission",None) or not self.session.permits(now):
                        raise ValueError("Нет действующей миссии обученной политики")
                    values=req.get("velocity")
                    if not isinstance(values,list) or len(values)!=3 or any(
                            not isinstance(value,(float,int)) or not math.isfinite(value) or abs(value)>limit
                            for value,limit in zip(values,(.12,.12,.35))):
                        raise ValueError("Скорость политики вне проверяемого диапазона")
                    self.session.renew(req["mission"],now)
                    if not self.gate.submit(values,"autonomy",float(req["at"])):
                        raise ValueError("Миссия не допущена")
                    self.autonomy_lease=now
                elif op == 'arm':
                    raise ValueError('General arm execution is not commissioned. Supervised near-home probes use bin/commission-arm.py; servo feedback is unavailable.')
                else:
                    raise ValueError('Unknown operation')
                if op not in ('drive','autonomy_lease','commission_lease') and (before!=self.snapshot() or op in ('hold_base','cancel_mission','finish_mission','clear_stop')):
                    self.event(req.get('initiator','request'),op.upper(),before)
            self.last_result = dict(id=req.get('id'), ok=True, op=op)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.last_result = dict(id=req.get('id'), ok=False, error=str(exc))
        self.request_ack.publish(String(data=json.dumps(dict(self.last_result,
            handled_monotonic=time.monotonic(),mcu_acknowledged=False))))

    def tick(self):
        now = time.monotonic()
        arm_controller = self.native.state.get('controller', {}) if self.native else {}
        arm_execution_pending = bool(arm_controller.get('arm_enabled') or arm_controller.get('arm_cancel_pending'))
        arm_link_ok=all(now-self.seen.get(k,-1e9)<ttl for k,ttl in [('odom',.5),('battery',2.)])
        if not arm_link_ok:
            source_faults={k:v for k,v in self.source_freshness.diagnostics.items() if v['reason']}
            self.emergency('telemetry','SENSOR_ACQUISITION_INVALID' if source_faults else 'MCU_TELEMETRY_LOST')
        if getattr(self,'arm_link_was_healthy',False) and not arm_link_ok:
            try:
                fault=ROOT/'data/arm-telemetry-fault.tmp'
                fault.write_text(json.dumps(dict(at=time.time(),reason='MCU telemetry interrupted')))
                fault.replace(ROOT/'data/arm-telemetry-fault.json')
            except OSError:self.emergency('telemetry','FAULT_RECORD_FAILED')
        self.arm_link_was_healthy=arm_link_ok
        self.power_state = self.power_policy.evaluate(now, active=arm_execution_pending or now<self.arm_active_until or any(abs(v)>.001 for v in self.gate.output), charging=self.charging)
        if not self.power_state['motion_allowed']:
            self.emergency('power',self.power_state['state'])
        scale = self.power_state['speed_scale']
        self.gate.command = [max(-m*scale, min(m*scale, v)) for m,v in zip(self.config['max_velocity'],self.gate.command)]
        healthy = all(now-self.seen.get(k,-1e9) < ttl for k,ttl in [('odom',.5),('imu',.5),('scan0',.6),('scan1',.6),('battery',3.)])
        healthy = healthy and self.power_state['motion_allowed'] and self.battery is not None and self.battery > self.config['battery_stop_voltage']
        # Conservative all-direction guard until the measured scanner extrinsics are installed.
        # The fixed lidars have legitimate near-field returns from the robot itself.
        # Until a direction-aware footprint filter is available, obstacle stops belong
        # to autonomous navigation.  A present operator keeps the independent STOP,
        # command-expiry, power and sensor gates, but must be able to manoeuvre away.
        collision = self.gate.mode == 'AUTONOMOUS' and any(
            s['nearest'] is None or s['nearest'] < .30 for s in self.scans.values())
        if self.session.mission and now-self.session.lease_at>.25:
            before=self.snapshot();self.session.end();self.gate.hold();self.hold_requested=True
            self.event('mission','MISSION_LEASE_EXPIRED',before)
        before=self.snapshot()
        source_at = self.gate.command_at
        if self.probe:
            source_at = self.probe.last_lease
            velocity,self.reason=self.probe.tick(now,now-self.last_tick,self.gate.estop,healthy,collision)
            if velocity is not None:self.gate.output=list(velocity)
            if self.probe.finished:
                self.last_probe_result=dict(id=self.probe_id,reason=self.reason,
                    completed=self.reason=='PROBE COMPLETE',ended_monotonic=now,
                    elapsed_s=now-self.probe.started,requested=self.probe.target)
                self.gate.hold();self.hold_requested=True
                self.probe=None
                self.probe_token=None
        else:
            healthy = healthy and self.config['lidar_tf_validated']
            velocity, self.reason = self.gate.tick(now, now-self.last_tick, healthy, collision)
        command_arm=self.native and (self.native.state.get('arm') or {}).get('estimated_only') is True
        if arm_execution_pending and not command_arm:
            if self.probe:self.probe.cancel('ARM EXECUTION ACTIVE')
            self.gate.hold()
            velocity = [0., 0., 0.]
            self.reason = 'ARM EXECUTION ACTIVE'
        elif arm_execution_pending and command_arm and velocity is not None:
            velocity=[max(-limit,min(limit,value)) for value,limit in zip(velocity,[.05,.05,.15])]
            self.gate.output=list(velocity)
        signature=(self.reason,self.gate.mode,self.gate.estop,self.session.mission)
        if signature!=getattr(self,'last_signature',None):
            self.event('velocity_gate',self.reason,before);self.last_signature=signature
        observed=self.odom_velocity
        stationary=arm_link_ok and observed is not None and len(observed)==3 and all(
            math.isfinite(v) and abs(v)<limit for v,limit in zip(observed,[.005,.005,.02])) and not any(self.gate.output)
        self.stationary_since=(self.stationary_since or now) if stationary else None
        self.last_tick = now
        if velocity is not None:
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = velocity
            if self.native:
                self.native.velocity(velocity, int(source_at*1e9), time.monotonic_ns())
            else:
                cadence=getattr(self,'factory_zero_cadence',None)
                if cadence is None or cadence.publish_due(velocity,self.stationary_since,now):
                    self.pub.publish(msg)
        self.heartbeat.publish(UInt64(data=time.monotonic_ns()))

    def write_status(self):
        now = time.monotonic()
        try:
            arm_state=json.loads((ROOT/'data/arm-state.json').read_text())
            if arm_state.get('boot_id')!=self.boot_id:arm_state=dict(phase='UNKNOWN_AFTER_REBOOT',measured=False)
            controller_boot = arm_state.get('controller_boot_id')
            if controller_boot is not None:
                current_boot = ((self.native.state.get('identity') or {}).get('boot')
                                if self.native else None)
                if controller_boot != current_boot:
                    arm_state=dict(phase='UNKNOWN_AFTER_CONTROLLER_REBOOT',measured=False)
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
                      monotonic=now,boot_id=self.boot_id,config_sha256=self.config_sha256,
                      mission=self.session.mission,mission_held=self.session.held,
                      base_hold_confirmed=bool(self.hold_requested and self.stationary_since and now-self.stationary_since>=.25),
                      last_motion_event=self.events.last,motion_log_dropped=self.events.dropped,
                      last_probe_result=self.last_probe_result,
                      sensor_source_time=self.source_freshness.diagnostics,
                      commissioning_config_matches_disk=hashlib.sha256((ROOT/'config/commissioning.json').read_bytes()).hexdigest()==self.config_sha256,
                      odom_velocity=self.odom_velocity,odom_received_monotonic=self.seen.get('odom'),
                      reason=self.reason, battery=self.battery, raw_pose=self.pose,
                      battery_gauge=battery_summary(self.power_policy.filtered,now-self.seen.get('battery',-1e9)),
                      power=self.power_state,
                      velocity=self.gate.output, sensor_age={k:round(now-v,3) for k,v in self.seen.items()},
                      lidar=self.scans, arm_feedback=self.arm_feedback, commissioning=self.config,
                      arm_measurements=self.arm_measurements,
                      arm_state=self.arm_measurements if self.native else None,
                      arm_feedback_transport=self.arm_feedback_transport,
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
            publish_stop(node)
            time.sleep(.02)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
