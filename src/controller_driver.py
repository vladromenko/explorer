"""ROS 2 bridge for the new, separately approved STM32 image.

Opt-in replacement for the serial micro-ROS agent. Legacy /cmd_vel and
/arm6_joints are NEVER subscribed, so they cannot bypass session/deadline checks.
No motion is enabled by connecting or reconnecting this node.
"""
import json
import hashlib
import math
import os
from pathlib import Path
import secrets
import signal
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState, LaserScan, MagneticField
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from builtin_interfaces.msg import Time
import serial

from controller_protocol import ClockMapping, Kind, Parser, Session, arm_payload, base_payload, encode
from controller_feedback import ArmFeedback, ScanAssembler, decode_status, load_calibration
from controller_requests import PendingRequests

ROOT = Path(os.environ.get('EXPLORER_ROOT', '/home/vlad/Explorer'))


class ControllerDriver(Node):
    def __init__(self):
        super().__init__('explorer_controller_driver')
        self.profile = json.loads((ROOT/'config/controller-profile.json').read_text())
        if self.profile.get('transport') != 'controller_v1':
            raise ValueError('controller v1 profile is not selected')
        self.expected_source = self.profile['firmware_source_sha256']
        if len(self.expected_source) != 64:
            raise ValueError('exact approved source identity is required')
        self.serial = serial.Serial(port=None, baudrate=2000000, timeout=0, write_timeout=0,
                                    exclusive=True)
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.port = self.profile.get('device', '/dev/explorer_mcu')
        self.serial.open()
        calibration_path = ROOT/'config/controller-calibration.json'
        if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != self.profile['calibration_sha256']:
            raise ValueError('calibration changed since profile approval')
        self.calibration = load_calibration(calibration_path)
        self.parser, self.session, self.arm = Parser(), Session(), ArmFeedback(self.calibration)
        self.pending_requests = PendingRequests()
        self.scans = [ScanAssembler(), ScanAssembler()]
        self.challenges = {}
        self.identity = None
        self.state = {}
        self.last_status_ns = None
        self.last_odom_us = None
        self.pose = [0., 0., 0.]
        self.fault = None
        self.sensor_health = {}
        self.source_sha256 = None
        self.odom_pub = self.create_publisher(Odometry, '/odom_raw', qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', qos_profile_sensor_data)
        self.mag_pub = self.create_publisher(MagneticField, '/imu/mag', qos_profile_sensor_data)
        self.battery_pub = self.create_publisher(Float32, '/battery', qos_profile_sensor_data)
        self.scan_pub = [self.create_publisher(LaserScan, '/scan'+str(i), qos_profile_sensor_data) for i in range(2)]
        self.joint_pub = self.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        self.state_pub = self.create_publisher(String, '/explorer/controller_state', 10)
        self.result_pub = self.create_publisher(String, '/explorer/controller_result', 10)
        self.arm_pub = self.create_publisher(String, '/explorer/arm_measurements', 10)
        self.servo_sample_pub = self.create_publisher(String, '/explorer/servo_sample', 100)
        self.create_subscription(String, '/explorer/controller_request', self.command, 10)
        self.create_timer(.005, self.poll)
        self.create_timer(.5, self.handshake)
        self.create_timer(.1, self.publish_state)
        self.handshake()

    def send(self, packet):
        if self.serial.write(packet) != len(packet):
            raise IOError('partial serial write: command will not be retried')

    def handshake(self):
        now = time.monotonic_ns()
        self.challenges = {key: value for key, value in self.challenges.items() if now-value < 100_000_000}
        nonce = secrets.randbits(64)
        self.challenges[nonce] = now
        try:
            self.send(encode(Kind.HELLO, struct.pack('<Q', nonce)))
        except (OSError, serial.SerialException) as exc:
            self.disconnect(str(exc))

    def disconnect(self, reason):
        self.pending_requests.clear()
        self.session.disconnected()
        self.fault = reason
        self.arm = ArmFeedback(self.calibration)
        self.last_odom_us = None
        self.scans = [ScanAssembler(), ScanAssembler()]

    def stamp(self, mcu_us, ttl_ns=500_000_000):
        now = time.monotonic_ns()
        acquired = self.session.clock.source_host_ns(mcu_us, now, ttl_ns)
        # Epoch is used only when publishing ROS messages. A late NTP correction
        # never changes the monotonic command lease or source sample age.
        epoch = self.get_clock().now().nanoseconds - (now-acquired)
        return Time(sec=epoch//1_000_000_000, nanosec=epoch%1_000_000_000)

    def command(self, message):
        request = {}
        try:
            request = json.loads(message.data)
            kind = Kind[request['operation']]
            if kind == Kind.ESTOP:
                self.send(encode(Kind.ESTOP))
                self.pending_requests.clear()
                self.session.state = 'fault'
                return
            if self.profile.get('telemetry_only', True):
                raise ValueError('controller commissioning: telemetry only')
            now = time.monotonic_ns()
            if self.last_status_ns is None or now-self.last_status_ns > 200_000_000:
                raise ValueError('controller telemetry unavailable')
            data = b''
            if kind == Kind.BASE:
                data = base_payload(request['velocity'], request.get('finite_us', 0))
            elif kind in (Kind.ARM, Kind.ARM_RECOVER):
                position = request['position_rad']
                if len(position) != 6:
                    raise ValueError('all six actuator positions required')
                if kind == Kind.ARM:
                    position = [calibration.target_position(q)
                                for calibration, q in zip(self.arm.calibration, position)]
                data = arm_payload(position)
            elif kind == Kind.CALIBRATION:
                data = b''.join(cal.payload() for cal in self.arm.calibration)
            elif kind == Kind.RGB:
                data = bytes(request['rgb'])
                if len(data) != 3:
                    raise ValueError('RGB requires three bytes')
            elif kind == Kind.BEEP:
                if type(request['enabled']) is not bool:
                    raise ValueError('beeper state must be boolean')
                data = bytes([request['enabled']])
            elif kind not in (Kind.OPEN, Kind.HOLD, Kind.CANCEL, Kind.CLEAR, Kind.ARM_ENABLE, Kind.ARM_CANCEL, Kind.RECOVERY_ENABLE):
                raise ValueError('unsupported command')
            packet = self.session.prepare(kind, request['source_monotonic_ns'], request['expires_monotonic_ns'],
                                          now, data, source_id=request['source_id'],
                                          source_sequence=request['source_sequence'])
            self.pending_requests.remember(boot=self.session.clock.boot, session=self.session.session,
                sequence=self.session.sequence, operation=kind, request=request, now_ns=now,
                earliest_result_us=self.session.clock.interval(now, now)[0])
            self.send(packet)
        except (ValueError, KeyError, TypeError, OverflowError, OSError, serial.SerialException) as exc:
            self.result_pub.publish(String(data=json.dumps(dict(
                source_id=request.get('source_id'), source_sequence=request.get('source_sequence'),
                accepted=False, reached=False, reason=str(exc)))))
            if isinstance(exc, (OSError, serial.SerialException)):
                self.disconnect(str(exc))

    def poll(self):
        try:
            now = time.monotonic_ns()
            for kind, payload in self.parser.feed(self.serial.read(8192)):
                self.receive(kind, payload, now)
            if self.last_status_ns is not None and now-self.last_status_ns > 250_000_000:
                self.disconnect('controller status deadline missed')
                self.last_status_ns = None
        except (ValueError, struct.error) as exc:
            self.fault = str(exc)
        except (OSError, serial.SerialException) as exc:
            self.disconnect(str(exc))

    def receive(self, kind, payload, now):
        if kind == Kind.IDENTITY:
            if len(payload) != 88:
                raise ValueError('incompatible controller identity')
            nonce, mcu_us, boot, device, revision, flash_kib = struct.unpack_from('<QQQIIH', payload)
            sent = self.challenges.pop(nonce, None)
            if sent is None:
                raise ValueError('unsolicited/replayed clock observation')
            source = payload[49:81].hex()
            if source != self.expected_source or payload[48] != 1 or device != 0x450:
                self.disconnect('controller build/protocol/chip identity mismatch')
                return
            clock = ClockMapping.observation(boot, sent, now, mcu_us)
            if self.session.clock is None or self.session.clock.boot != boot:
                self.pending_requests.clear()
            self.session.synchronize(clock, self.state.get('highest_session', 0))
            self.source_sha256 = source
            self.identity = dict(boot=boot, device=device, revision=revision, flash_kib=flash_kib,
                                 source_sha256=source, reset_flags=struct.unpack_from('<I', payload, 84)[0])
            self.fault = None
        elif self.session.clock is not None:
            if kind == Kind.SENSOR_DIAGNOSTICS:
                if len(payload) != 64:
                    raise ValueError('bad sensor diagnostics record')
                self.sensor_health['startup'] = dict(received_ns=now,
                    imu_stage=payload[8], imu_id=payload[9], gyro_config=payload[10],
                    accel_config=payload[11], spi_status=payload[12],
                    lidar=[dict(zip(('rx_bytes','valid_packets','parser_errors','overruns','uart_errors','start_attempts'),
                        struct.unpack_from('<6I', payload, 16+24*i))) for i in range(2)])
            elif kind == Kind.STATUS:
                state = decode_status(payload)
                if state['boot'] != self.session.clock.boot:
                    self.disconnect('MCU rebooted')
                    return
                self.stamp(state['acquired_us'])
                self.last_status_ns = now
                self.state = state
                self.session.highest = max(self.session.highest, state['highest_session'])
                if state['mode'] == 2:
                    self.session.state = 'fault'
                if state['encoder_measurement_valid']:
                    self.publish_odom(state)
                else:
                    self.last_odom_us = None
                    self.fault = 'encoder measurement invalid; odometry not published'
            elif kind == Kind.RESULT:
                if len(payload) != 20:
                    raise ValueError('bad result record')
                acquired, sequence, operation, result, mode, fault = struct.unpack('<QQBBBB', payload)
                self.stamp(acquired)
                source = self.pending_requests.match(boot=self.session.clock.boot,
                    session=self.session.session, sequence=sequence, operation=operation,
                    acquired_us=acquired, now_ns=now)
                if source is not None:
                    self.session.result(Kind(operation), sequence, result)
                self.result_pub.publish(String(data=json.dumps(dict(sequence=sequence, operation=operation,
                    accepted=result == 0, error=result, mode=mode, fault=fault, reached=False,
                    matched_request=source is not None,
                    **(source or dict(source_id=None, source_sequence=None))))))
            elif kind == Kind.SERVO:
                sample = self.arm.consume(payload, self.session.clock, now)
                self.servo_sample_pub.publish(String(data=json.dumps(dict(sample, boot_id=self.session.clock.boot),allow_nan=False)))
                if sample['position_valid'] and sample['joint'] <= 5:
                    msg = JointState()
                    msg.header.stamp = self.stamp(struct.unpack_from('<Q', payload, 13)[0], 250_000_000)
                    msg.name = ['arm'+str(sample['joint'])+'_Joint']
                    msg.position = [sample['position_rad']]
                    self.joint_pub.publish(msg)
            elif kind == Kind.BATTERY:
                if len(payload) != 15:
                    raise ValueError('bad battery record')
                acquired, error, raw, volts = struct.unpack('<QB Hf', payload)
                self.sensor_health['battery'] = dict(error=error, received_ns=now)
                self.stamp(acquired, 2_000_000_000)
                if error == 0 and 0 < raw < 4096 and math.isfinite(volts) and 0 < volts < 20:
                    self.battery_pub.publish(Float32(data=volts))
            elif kind == Kind.IMU:
                self.sensor_health['imu'] = dict(error=payload[8], mag_error=payload[9], received_ns=now)
                self.publish_imu(payload)
            elif kind in (Kind.LIDAR0, Kind.LIDAR1):
                index = int(kind)-int(Kind.LIDAR0)
                self.sensor_health['lidar'+str(index)] = dict(received_ns=now)
                complete = self.scans[index].packet(payload)
                if complete is not None:
                    self.publish_scan(index, complete)

    def publish_odom(self, state):
        sample = state['acquired_us']
        vx, vy, wz = state['velocity']
        if self.last_odom_us is not None:
            dt = (sample-self.last_odom_us)/1e6
            if not 0 < dt < .2:
                self.last_odom_us = sample
                raise ValueError('odometry sample gap; no extrapolated travel')
            yaw = self.pose[2] + wz*dt/2
            self.pose[0] += (vx*math.cos(yaw)-vy*math.sin(yaw))*dt
            self.pose[1] += (vx*math.sin(yaw)+vy*math.cos(yaw))*dt
            self.pose[2] += wz*dt
        self.last_odom_us = sample
        msg = Odometry()
        msg.header.stamp = self.stamp(sample)
        msg.header.frame_id, msg.child_frame_id = 'odom', 'base_footprint'
        msg.pose.pose.position.x, msg.pose.pose.position.y = self.pose[:2]
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(self.pose[2]/2), math.cos(self.pose[2]/2)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z = vx, vy, wz
        for i in (0, 7, 35):
            msg.pose.covariance[i], msg.twist.covariance[i] = .05, .02
        self.odom_pub.publish(msg)

    def publish_imu(self, payload):
        if len(payload) != 44:
            raise ValueError('bad IMU record')
        acquired = struct.unpack_from('<Q', payload)[0]
        stamp = self.stamp(acquired)
        if payload[8] == 0:
            raw = struct.unpack_from('>7h', payload, 10)
            msg = Imu()
            msg.header.stamp, msg.header.frame_id = stamp, 'imu_frame'
            msg.orientation_covariance[0] = -1.0  # orientation is not measured here
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = [v*9.80665/2048 for v in raw[:3]]
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = [math.radians(v/16.4) for v in raw[3:6]]
            for i in (0, 4, 8):
                msg.angular_velocity_covariance[i], msg.linear_acceleration_covariance[i] = .01, .1
            self.imu_pub.publish(msg)
        if payload[9] == 0:
            mag = struct.unpack_from('<3h', payload, 25)
            msg = MagneticField()
            msg.header.stamp, msg.header.frame_id = self.stamp(struct.unpack_from('<Q', payload, 36)[0]), 'imu_frame'
            msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z = [v*.15e-6 for v in mag]
            self.mag_pub.publish(msg)

    def publish_scan(self, index, scan):
        msg = LaserScan()
        msg.header.stamp = self.stamp(scan['acquired_us'], 600_000_000)
        msg.header.frame_id = 'laser'+str(index)+'_frame'
        msg.angle_min, msg.angle_increment = 0., 2*math.pi/666
        msg.angle_max = 665*msg.angle_increment
        msg.scan_time = (scan['ended_us']-scan['acquired_us'])/1e6
        msg.time_increment = msg.scan_time/666
        msg.range_min, msg.range_max = .02, 12.
        msg.ranges, msg.intensities = scan['ranges'], scan['intensities']
        self.scan_pub[index].publish(msg)

    def publish_state(self):
        now = time.monotonic_ns()
        self.pending_requests.expire(now)
        arm = self.arm.snapshot(now)
        self.arm_pub.publish(String(data=json.dumps(arm, allow_nan=False)))
        state = dict(at=time.time(), monotonic_ns=now, identity=self.identity,
                     telemetry_only=self.profile.get('telemetry_only', True), sensors=self.sensor_health,
                     session_state=self.session.state, fault=self.fault, controller=self.state,
                     telemetry_fresh=self.last_status_ns is not None and now-self.last_status_ns < 250_000_000,
                     parser_errors=self.parser.errors, arm=arm)
        self.state_pub.publish(String(data=json.dumps(state, allow_nan=False)))
        target = ROOT/'data/controller-state.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, allow_nan=False))
        temporary.replace(target)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    driver = ControllerDriver()
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            driver.send(encode(Kind.ESTOP))
        except (OSError, serial.SerialException):
            pass
        driver.serial.close()
        driver.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
