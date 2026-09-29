"""Exercise the actual ROS driver class against a serial/ROS software stand.

No serial device or ROS service is opened. The parser/session/request tracking
are production code; only transports, ROS publishers and the clock are mocked.
"""
import ast
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import secrets
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from controller_feedback import ArmFeedback, ScanAssembler, load_calibration, vendor_calibration
from controller_protocol import ClockMapping, Kind, Parser, Session, arm_payload, base_payload, decode, encode, RESULT_NAMES
from controller_requests import PendingRequests
from controller_release import select_controller_profile
from controller_arm_commissioning import arm_commissioning_allowed


class FakeNode:
    def __init__(self, name):
        self.name = name

    def create_publisher(self, *args):
        return SimpleNamespace(publish=Mock())

    def create_subscription(self, *args):
        return None

    def create_timer(self, *args):
        return None

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_700_000_000_000_000_000))


class FakePort:
    def __init__(self, available):
        self.available = available
        self.is_open = False
        self.closed = False
        self.writes = []
        self.input = b''
        self.partial = False
        self.read_error = None
        self.output_resets = self.input_resets = 0

    def open(self):
        if not self.available:
            raise OSError('USB device absent')
        self.is_open = True

    def close(self):
        self.closed, self.is_open = True, False

    def reset_input_buffer(self):
        self.input_resets += 1
        self.input = b''

    def reset_output_buffer(self):
        self.output_resets += 1

    def read(self, size):
        if self.read_error:
            raise self.read_error
        value, self.input = self.input[:size], self.input[size:]
        return value

    def write(self, packet):
        if not self.is_open:
            raise OSError('closed port')
        self.writes.append(packet)
        return len(packet)-1 if self.partial else len(packet)


class DriverReconnectTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000_000
        self.available = True
        self.ports = []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'config').mkdir()
        calibration = json.dumps(dict(schema=1, joints=[asdict(c) for c in vendor_calibration()])).encode()
        (self.root/'config/controller-calibration.json').write_bytes(calibration)
        (self.root/'config/controller-profile.json').write_text(json.dumps(dict(
            transport='controller_v1', firmware_source_sha256='ab'*32,
            calibration_sha256=hashlib.sha256(calibration).hexdigest(), telemetry_only=False)))
        def factory(**kwargs):
            self.assertEqual(kwargs['port'], None)
            self.assertEqual(kwargs['baudrate'], 2000000)
            self.assertTrue(kwargs['exclusive'])
            port = FakePort(self.available)
            self.ports.append(port)
            return port
        source = Path(__file__).parents[1]/'src/controller_driver.py'
        tree = ast.parse(source.read_text())
        node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'ControllerDriver')
        self.ns = dict(Node=FakeNode, ROOT=self.root, json=json, hashlib=hashlib, math=math,
            Path=Path, secrets=secrets, struct=struct, time=SimpleNamespace(monotonic_ns=lambda: self.now),
            serial=SimpleNamespace(Serial=factory, SerialException=OSError),
            Parser=Parser, Session=Session, ClockMapping=ClockMapping, Kind=Kind, encode=encode, RESULT_NAMES=RESULT_NAMES,
            ArmFeedback=ArmFeedback, ScanAssembler=ScanAssembler, load_calibration=load_calibration,
            select_controller_profile=select_controller_profile,
            arm_commissioning_allowed=arm_commissioning_allowed,
            PendingRequests=PendingRequests, base_payload=base_payload, arm_payload=arm_payload,
            decode_status=lambda payload: self.status, qos_profile_sensor_data=None,
            **{name: SimpleNamespace for name in ('Odometry', 'Imu', 'MagneticField', 'Float32',
                                                 'LaserScan', 'JointState', 'String', 'Time')})
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), self.ns)
        self.driver_class = self.ns['ControllerDriver']

    def ready(self, driver, boot=7):
        driver.identity_boot(boot, self.now)
        driver.session.synchronize(ClockMapping(boot, self.now, 10_000_000, 10_000_000))
        driver.identity = dict(boot=boot)
        self.status = dict(boot=boot, acquired_us=self.now//1000+10_000_000, highest_session=2,
                           mode=0, encoder_measurement_valid=True)
        driver.publish_odom = Mock()
        driver.receive(Kind.STATUS, b'', self.now)

    def request(self, operation, source=None, sequence=1):
        return SimpleNamespace(data=json.dumps(dict(operation=operation, source_id='test-owner',
            source_sequence=sequence, source_monotonic_ns=self.now if source is None else source,
            expires_monotonic_ns=self.now+200_000_000, velocity=[.1, 0, 0])))

    def test_missing_at_startup_retries_without_process_crash_or_motion(self):
        self.available = False
        driver = self.driver_class()
        self.assertIsNone(driver.serial)
        self.assertTrue(self.ports[0].closed)
        self.assertEqual(driver.session.state, 'disconnected')
        driver.poll()
        self.assertEqual(len(self.ports), 1)
        self.now += 500_000_000
        self.available = True
        driver.poll()
        self.assertEqual(len(self.ports), 2)
        self.assertEqual([decode(p[:-1])[0] for p in driver.serial.writes], [Kind.HELLO])
        self.assertEqual(driver.serial.input_resets, 1)
        self.assertEqual(driver.serial.output_resets, 1)
        self.assertFalse(driver.serial.dtr)
        self.assertFalse(driver.serial.rts)
        self.assertIsNone(driver.session.clock)

    def test_unplug_closes_and_invalidates_every_old_observation(self):
        driver = self.driver_class()
        self.ready(driver)
        driver.session.state, driver.session.session = 'active', 3
        driver.sensor_health = {'battery': {'received_ns': self.now}}
        driver.source_sha256 = 'ab'*32
        driver.last_odom_us = 10
        driver.scans[0].started_us = 10
        driver.parser.feed(b'\x03')
        driver.pending_requests.pending[(1, 3)] = {'sent_ns': self.now}
        old = driver.serial
        old.read_error = OSError('USB disconnected')
        driver.poll()
        self.assertTrue(old.closed)
        self.assertIsNone(driver.serial)
        self.assertIsNone(driver.identity)
        self.assertIsNone(driver.last_status_ns)
        self.assertIsNone(driver.session.clock)
        self.assertIsNone(driver.source_sha256)
        self.assertEqual(driver.state, {})
        self.assertEqual(driver.sensor_health, {})
        self.assertEqual(driver.pending_requests.pending, {})
        self.assertFalse(driver.parser.buffer)
        self.assertFalse(driver.challenges)
        self.assertIsNone(driver.scans[0].started_us)
        self.assertFalse(driver.arm.snapshot(self.now)['all_fresh'])

    def test_silent_initial_port_and_silent_active_port_are_reopened(self):
        driver = self.driver_class()
        old = driver.serial
        self.now += 2_000_000_001
        driver.poll()
        self.assertTrue(old.closed)
        self.assertIn('startup', driver.fault)
        self.now = driver.next_connect_ns
        driver.poll()
        self.ready(driver)
        old = driver.serial
        self.now += 250_000_001
        driver.poll()
        self.assertTrue(old.closed)
        self.assertIsNone(driver.last_status_ns)
        self.assertIn('status deadline', driver.fault)

    def test_partial_command_is_never_replayed_after_reconnect(self):
        driver = self.driver_class()
        self.ready(driver)
        driver.session.state, driver.session.session = 'active', 3
        self.now += 1_000_000
        old = driver.serial
        old.partial = True
        driver.command(self.request('BASE'))
        self.assertTrue(old.closed)
        self.assertIsNone(driver.serial)
        self.assertFalse(driver.pending_requests.pending)
        self.assertEqual(json.loads(driver.result_pub.publish.call_args.args[0].data)['accepted'], False)
        self.now = driver.next_connect_ns
        driver.poll()
        self.assertEqual([decode(p[:-1])[0] for p in driver.serial.writes], [Kind.HELLO])
        self.ready(driver)
        self.now += 1_000_000
        driver.command(self.request('BASE', sequence=2))
        self.assertEqual(len(driver.serial.writes), 1)
        self.assertIn('explicit session', json.loads(driver.result_pub.publish.call_args.args[0].data)['reason'])

    def test_command_from_before_new_ready_state_cannot_open_session(self):
        driver = self.driver_class()
        old_source = self.now
        self.now += 50_000_000
        self.ready(driver)
        self.now += 1_000_000
        driver.command(self.request('OPEN', source=old_source))
        self.assertEqual(len(driver.serial.writes), 1)
        self.assertIn('before current', json.loads(driver.result_pub.publish.call_args.args[0].data)['reason'])
        driver.command(self.request('OPEN', sequence=2))
        self.assertEqual(decode(driver.serial.writes[-1][:-1])[0], Kind.OPEN)
        self.assertEqual(driver.session.state, 'opening')

    def test_boot_change_clears_ack_and_partial_sensor_state(self):
        driver = self.driver_class()
        self.ready(driver)
        driver.session.state, driver.session.session = 'active', 3
        driver.pending_requests.pending[(1, 8)] = {'sent_ns': self.now}
        driver.scans[1].started_us = 100
        driver.parser.feed(b'\x04')
        self.now += 1_000_000
        driver.identity_boot(8, self.now)
        self.assertIsNotNone(driver.serial)
        self.assertIsNone(driver.last_status_ns)
        self.assertIsNone(driver.session.clock)
        self.assertFalse(driver.pending_requests.pending)
        self.assertFalse(driver.parser.buffer)
        self.assertIsNone(driver.scans[1].started_us)
        self.assertEqual(driver.command_not_before_ns, self.now)
        driver.session.synchronize(ClockMapping(8, self.now, 10_000_000, 10_000_000))
        self.assertEqual(driver.session.state, 'disarmed')
        self.assertEqual(driver.session.session, 0)

    def test_same_boot_clock_refresh_preserves_live_session(self):
        driver = self.driver_class()
        self.ready(driver)
        driver.session.state, driver.session.session = 'active', 3
        fence = driver.command_not_before_ns
        self.now += 100_000_000
        driver.identity_boot(7, self.now)
        self.assertEqual(driver.session.state, 'active')
        self.assertEqual(driver.command_not_before_ns, fence)

    def test_changed_source_with_reused_boot_discards_session_ack_and_samples(self):
        driver = self.driver_class()
        self.ready(driver)
        driver.source_sha256 = 'ab'*32
        driver.session.state, driver.session.session = 'active', 3
        driver.pending_requests.pending[(1, 8)] = {'sent_ns': self.now}
        driver.arm.samples[0].update(position_valid=True, raw_valid=True,
                                    acquired_monotonic_ns=self.now)
        driver.scans[0].started_us = 100
        self.now += 1_000_000
        driver.identity_boot(7, self.now, 'cd'*32)
        self.assertIsNotNone(driver.serial)
        self.assertIsNone(driver.session.clock)
        self.assertEqual(driver.session.state, 'disconnected')
        self.assertEqual(driver.state, {})
        self.assertIsNone(driver.last_status_ns)
        self.assertFalse(driver.pending_requests.pending)
        self.assertFalse(driver.arm.samples[0]['position_valid'])
        self.assertIsNone(driver.scans[0].started_us)
        self.assertEqual(driver.command_not_before_ns, self.now)
        self.assertEqual(driver.connected_ns, self.now)

    def test_malformed_traffic_does_not_postpone_status_deadline(self):
        driver = self.driver_class()
        self.ready(driver)
        old = driver.serial
        self.now += 250_000_001
        old.input = encode(Kind.STATUS, b'invalid')
        driver.receive = Mock(side_effect=ValueError('invalid status'))
        driver.poll()
        self.assertTrue(old.closed)
        self.assertIsNone(driver.serial)
        self.assertIn('status deadline', driver.fault)

    def test_remainder_of_read_batch_is_discarded_after_disconnect(self):
        driver = self.driver_class()
        driver.serial.input = encode(Kind.STATUS, b'one') + encode(Kind.RESULT, b'two')
        received = []
        def receive(kind, payload, now):
            received.append(kind)
            driver.disconnect('new boot', now)
        driver.receive = receive
        driver.poll()
        self.assertEqual(received, [Kind.STATUS])
        self.assertIsNone(driver.serial)

    def test_reconnect_backoff_is_bounded_and_no_busy_loop(self):
        self.available = False
        driver = self.driver_class()
        delays = []
        for _ in range(6):
            delays.append(driver.next_connect_ns-self.now)
            before = len(self.ports)
            driver.poll()
            self.assertEqual(len(self.ports), before)
            self.now = driver.next_connect_ns
            driver.poll()
        self.assertEqual(delays, [500_000_000, 1_000_000_000, 2_000_000_000,
                                  4_000_000_000, 5_000_000_000, 5_000_000_000])

    def test_repeated_stop_requests_do_not_postpone_reconnection(self):
        self.available = False
        driver = self.driver_class()
        retry_at = driver.next_connect_ns
        for _ in range(4):
            self.now += 100_000_000
            driver.command(self.request('ESTOP'))
            self.assertEqual(driver.next_connect_ns, retry_at)
        self.available = True
        self.now = retry_at
        driver.poll()
        self.assertIsNotNone(driver.serial)
        self.assertEqual([decode(p[:-1])[0] for p in driver.serial.writes], [Kind.HELLO])

    def test_malformed_request_does_not_crash_link(self):
        driver = self.driver_class()
        for value in ('[]', 'null', 'true', '4', '"BASE"'):
            driver.command(SimpleNamespace(data=value))
            self.assertFalse(json.loads(driver.result_pub.publish.call_args.args[0].data)['accepted'])
        self.assertIsNotNone(driver.serial)
        self.assertEqual(len(driver.serial.writes), 1)

    def test_real_identity_branch_resets_boot_and_discards_same_batch_ack(self):
        driver = self.driver_class()
        self.ready(driver)
        driver.session.state, driver.session.session = 'active', 3
        driver.pending_requests.pending[(1, 2)] = {'sent_ns': self.now}
        self.now += 60_000_000_000
        driver.handshake()
        nonce = next(iter(driver.challenges))
        self.now += 1_000_000
        identity = bytearray(88)
        struct.pack_into('<QQQIIH', identity, 0, nonce, self.now//1000+10_000_000,
                         8, 0x450, 0x2003, 1024)
        identity[36:48] = bytes.fromhex('01'*12)
        identity[48] = 1
        identity[49:81] = bytes.fromhex('ab'*32)
        old_result = struct.pack('<QQBBBB', self.now//1000+10_000_000, 1, Kind.OPEN, 0, 1, 0)
        driver.serial.input = encode(Kind.IDENTITY, identity) + encode(Kind.RESULT, old_result)
        driver.poll()
        self.assertEqual(driver.session.clock.boot, 8)
        self.assertIsNotNone(driver.serial)
        self.assertEqual(driver.connected_ns, self.now)
        self.assertEqual(driver.identity['boot'], 8)
        self.assertEqual(driver.session.state, 'disarmed')
        self.assertIsNone(driver.last_status_ns)
        self.assertFalse(driver.pending_requests.pending)
        driver.result_pub.publish.assert_not_called()
