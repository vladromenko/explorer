"""Exercise the actual Core methods without requiring a ROS installation."""
import ast
import json
import math
from pathlib import Path
import secrets
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from commissioning import Pulse
from controller_control import BaseTransport
from motion_session import MotionSession
from safety import SafetyGate


class CoreNativeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.03
        source = Path(__file__).resolve().parents[1]/'src/core.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        names = {'Core', 'publish_stop', 'publish_hold', 'main'}
        selected = ast.Module(body=[item for item in tree.body if getattr(item, 'name', None) in names], type_ignores=[])
        self.clock = SimpleNamespace(monotonic=lambda:self.now,
                                     monotonic_ns=lambda:round(self.now*1e9), sleep=Mock())
        self.namespace = dict(Node=object, time=self.clock, json=json, math=math,
                              secrets=secrets, Pulse=Pulse, String=SimpleNamespace,
                              UInt64=SimpleNamespace, Twist=lambda:SimpleNamespace(
                                  linear=SimpleNamespace(x=0., y=0.), angular=SimpleNamespace(z=0.)))
        exec(compile(selected, str(source), 'exec'), self.namespace)
        self.core = self.namespace['Core'].__new__(self.namespace['Core'])
        c = self.core
        c.config = dict(max_velocity=[.15,.15,.4], max_acceleration=[.3,.3,.7],
                        battery_stop_voltage=10.5, lidar_tf_validated=True,
                        base_commissioned=True, mcu_watchdog_verified=True)
        c.gate = SafetyGate(c.config); c.gate.estop = False
        c.session = MotionSession(); c.probe = None; c.arm_active_until = 0.
        c.power_state = dict(motion_allowed=True, speed_scale=1.)
        c.power_policy = SimpleNamespace(evaluate=Mock(return_value=c.power_state))
        c.seen = dict.fromkeys(('odom','imu','scan0','scan1','battery'), self.now)
        c.source_freshness = SimpleNamespace(diagnostics={})
        c.charging = None; c.battery = 12.; c.last_tick = 100.
        c.scans = {'scan0':{'nearest':1.}, 'scan1':{'nearest':1.}}
        c.odom_velocity = [0.,0.,0.]; c.stationary_since = None
        c.hold_requested = False; c.autonomy_lease = -1e9; c.joy_held = False
        c.pub = Mock(); c.heartbeat = Mock(); c.request_ack = Mock(); c.event = Mock()
        c.probe_id = 'probe'; c.probe_token = 'token'
        self.requests = []
        c.native = BaseTransport(self.requests.append)
        c.native.observe(dict(identity=dict(boot=1), session_state='active',
                              telemetry_fresh=True, telemetry_only=False), round(self.now*1e9))

    def test_probe_uses_lease_origin_instead_of_stale_gate_timestamp(self):
        c = self.core
        c.probe = Pulse([.04,0.,0.], 1., 100.)
        c.probe.last_lease = 100.02
        c.tick()
        request = self.requests[-1]
        self.assertEqual(request['operation'], 'BASE')
        self.assertEqual(request['source_monotonic_ns'], 100_020_000_000)
        self.assertEqual(request['expires_monotonic_ns'], 100_220_000_000)
        self.assertGreater(request['velocity'][0], 0.)
        self.assertEqual(c.gate.command_at, -1e9)

    def test_stale_probe_lease_cannot_be_refreshed_by_a_tick(self):
        c = self.core
        c.probe = Pulse([.04,0.,0.], 1., 100.)
        self.now = 100.18
        c.tick()
        self.assertEqual(self.requests[-1]['operation'], 'HOLD')
        self.assertEqual(c.last_probe_result['reason'], 'PROBE LEASE EXPIRED')
        self.assertIsNone(c.probe)

    def test_arm_cancel_pending_stops_base_until_measured_controller_completion(self):
        c = self.core
        c.gate.submit([.04,0.,0.], 'manual', self.now)
        c.native.state['controller'] = dict(arm_enabled=False, arm_cancel_pending=True)
        c.tick()
        self.assertEqual(self.requests[-1]['operation'], 'HOLD')
        self.assertEqual(c.reason, 'ARM EXECUTION ACTIVE')
        self.assertFalse(c.gate.estop)
        c.native.state['controller']['arm_cancel_pending'] = False
        c.gate.submit([.04,0.,0.], 'manual', self.now)
        self.now += .02
        c.tick()
        self.assertEqual(self.requests[-1]['operation'], 'BASE')

    def test_near_lidar_return_does_not_lock_present_operator(self):
        c = self.core;c.scans['scan1']['nearest']=.15
        c.gate.submit([.04,0.,0.], 'manual', self.now)
        c.tick()
        self.assertEqual(self.requests[-1]['operation'],'BASE')
        self.assertEqual(c.reason,'ACTIVE')

    def test_near_lidar_return_still_stops_autonomy(self):
        c = self.core;c.gate.mode='AUTONOMOUS';c.scans['scan1']['nearest']=.15
        c.autonomy_lease=self.now+.5;c.gate.submit([.04,0.,0.], 'autonomy', self.now)
        c.tick()
        self.assertEqual(self.requests[-1]['operation'],'HOLD')
        self.assertEqual(c.reason,'OBSTACLE')

    def test_probe_and_renewal_keep_request_acquisition_time(self):
        c = self.core
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'data').mkdir()
            self.namespace['ROOT'] = root
            (root/'data/commissioning-permit.json').write_text(json.dumps(dict(expires=101., token='token')))
            c.request(SimpleNamespace(data=json.dumps(dict(op='commission_pulse', at=100.,
                token='token', velocity=[.04,0.,0.], duration=1.))))
            self.assertTrue(c.last_result['ok'])
            self.assertEqual(c.probe.started, 100.)
            self.assertEqual(c.probe.last_lease, 100.)
            c.request(SimpleNamespace(data=json.dumps(dict(op='commission_lease', at=100.01, token='token'))))
            self.assertEqual(c.probe.last_lease, 100.01)

    def test_stop_works_even_with_an_already_latched_stop(self):
        self.core.gate.estop = True
        self.core.request(SimpleNamespace(data=json.dumps(dict(op='stop'))))
        self.assertEqual(self.requests[-1]['operation'], 'ESTOP')
        self.assertTrue(self.core.last_result['ok'])

    def test_normal_hold_preserves_session_and_sends_immediate_native_hold(self):
        self.core.request(SimpleNamespace(data=json.dumps(dict(op='hold_base', at=100.02))))
        self.assertEqual(self.requests[-1]['operation'], 'HOLD')
        self.assertFalse(self.core.gate.estop)

    def test_manual_release_establishes_hold_for_arm_without_latching_stop(self):
        c=self.core;c.gate.submit([.04,0.,0.], 'manual', self.now)
        c.request(SimpleNamespace(data=json.dumps(dict(op='manual_release',at=self.now))))
        self.assertTrue(c.hold_requested)
        self.assertFalse(c.gate.estop)
        self.assertEqual(self.requests[-1]['operation'],'HOLD')

    def test_shutdown_sends_native_stop_before_destroying_node(self):
        c = self.core; c.destroy_node = Mock()
        self.namespace.update(Core=lambda:c, rclpy=SimpleNamespace(init=Mock(),
            spin=Mock(side_effect=KeyboardInterrupt), shutdown=Mock()),
            SignalHandlerOptions=SimpleNamespace(NO=0),
            signal=SimpleNamespace(SIGTERM=15, SIGINT=2, signal=Mock()))
        self.namespace['main']()
        self.assertEqual([r['operation'] for r in self.requests], ['ESTOP']*5)
        c.destroy_node.assert_called_once()
