from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from controller_feedback import vendor_calibration
from native_arm import NativeManualArm, measured_reference


class Message:
    def __init__(self, data):
        self.data = data


class Node:
    def __init__(self):
        self.sent = []

    def create_publisher(self, *_):
        return self

    def create_subscription(self, kind, topic, callback, size):
        self.callback = callback
        return callback

    def publish(self, message):
        self.sent.append(json.loads(message.data))


class Geometry:
    def path(self, start, goal, shape):
        return {'valid': True}


def state(calibration, now, raw=None):
    raw = [2000, 2000, 2000, 2000, 1487, 1633] if raw is None else raw
    samples = [dict(c.observe(value), joint=i+1, error=0, device_error=0,
                    acquired_monotonic_ns=now) for i, (c, value) in enumerate(zip(calibration, raw))]
    return dict(monotonic_ns=now, telemetry_fresh=True, identity={'boot': 123,'source_sha256':'a'*64},
                controller={'session': 11, 'mode': 1}, session_state='active',
                telemetry_only=False, arm={'joints': samples})


class NativeArmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'config').mkdir(); (self.root/'data').mkdir()
        self.calibration = vendor_calibration()
        content = json.dumps(dict(schema=1, joints=[asdict(c) for c in self.calibration])).encode()
        (self.root/'config/controller-calibration.json').write_bytes(content)
        (self.root/'config/controller-profile.json').write_text(json.dumps(dict(
            transport='controller_v1',firmware_source_sha256='a'*64,
            calibration_sha256=hashlib.sha256(content).hexdigest(), telemetry_only=False)))
        self.node = Node()
        self.arm = NativeManualArm(self.root, self.node, lambda: Geometry(), Message)
        self.write_state()

    def write_state(self, raw=None):
        value = state(self.calibration, time.monotonic_ns(), raw)
        (self.root/'data/controller-state.json').write_text(json.dumps(value))
        return value

    def test_measurements_preserve_negative_coordinates_and_reject_bad_identity(self):
        now = time.monotonic_ns()
        value = state(self.calibration, now, [2000, 2000, 3594, 2000, 1487, 1633])
        result = measured_reference(value, self.calibration, now)
        self.assertLess(result['servo_deg'][2], 0)
        self.assertLess(result['position_rad'][2], 0)
        self.assertEqual(result['outside_soft_limits'], [3])
        with self.assertRaises(ValueError):
            measured_reference(value, self.calibration, now, expected_boot=456)
        with self.assertRaises(ValueError):
            measured_reference(value, self.calibration, now, expected_session=10)
        value['arm']['joints'][0]['position_rad'] += .01
        with self.assertRaises(ValueError):
            measured_reference(value, self.calibration, now)

    def test_stale_or_missing_servo_feedback_cannot_be_a_reference(self):
        now = time.monotonic_ns()
        value = state(self.calibration, now)
        value['arm']['joints'][4]['acquired_monotonic_ns'] -= 250_000_001
        with self.assertRaises(ValueError):
            measured_reference(value, self.calibration, now)
        value = state(self.calibration, now)
        value['arm']['joints'][0]['error'] = 14
        with self.assertRaises(ValueError):
            measured_reference(value, self.calibration, now)

    def test_telemetry_only_and_outside_pose_never_emit_motion_or_open_session(self):
        self.arm.ready = True
        current = self.arm.reference()['servo_deg']
        self.arm.profile['telemetry_only'] = True
        (self.root/'config/controller-profile.json').write_text(json.dumps(self.arm.profile))
        with self.assertRaisesRegex(ValueError, 'телеметрию'):
            self.arm.move(current, current)
        self.arm.profile['telemetry_only'] = False
        (self.root/'config/controller-profile.json').write_text(json.dumps(self.arm.profile))
        self.write_state([2000, 2000, 3594, 2000, 1487, 1633])
        with self.assertRaisesRegex(ValueError, 'восстановление'):
            self.arm.move(current, current)
        self.assertEqual(self.node.sent, [])

    def test_new_controller_profile_is_seen_without_restarting_web(self):
        profile=dict(self.arm.profile,firmware_source_sha256='b'*64,telemetry_only=True,
            blocking_reason_ru='Новая прошивка: проверка на стенде')
        (self.root/'config/controller-profile.json').write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError,'Ожидаю'):
            self.arm.reference()
        value=self.write_state();value['identity']['source_sha256']='b'*64
        (self.root/'data/controller-state.json').write_text(json.dumps(value))
        self.assertTrue(self.arm.reference()['measured'])
        self.assertIn('Новая прошивка',self.arm.status()['blocked_by'])
        self.assertEqual(self.node.sent,[])

    def test_calibration_change_requires_reloading_coordinates(self):
        profile=dict(self.arm.profile,calibration_sha256='f'*64)
        (self.root/'config/controller-profile.json').write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError,'Калибровка'):
            self.arm.reference()

    def test_ack_only_matches_own_source_and_rejection_is_not_reached(self):
        sequence = self.arm._send('ARM_ENABLE')
        self.arm._result(Message(json.dumps(dict(source_id='other', source_sequence=sequence, accepted=True))))
        self.assertEqual(self.arm.results, {})
        self.arm._result(Message(json.dumps(dict(source_id=self.arm.source_id, source_sequence=sequence,
                                                accepted=False, error=16))))
        with self.assertRaisesRegex(ValueError, '16'):
            self.arm._check_results()
        result = self.arm.stop()
        self.assertTrue(result['cancel_requested'])
        self.assertFalse(result['physical_stop_latency_verified'])
        self.assertEqual([r['operation'] for r in self.node.sent], ['ARM_ENABLE', 'ARM_CANCEL'])

    def test_source_ttl_is_preserved_and_invalid_deadlines_never_publish(self):
        with patch('native_arm.time.monotonic_ns', return_value=1_000_000_000):
            for source, expires in ((850_000_000,1_000_000_001), (1_000_000_001,1_150_000_000),
                                    (900_000_000,1_000_000_000), (900_000_000,1_050_000_001)):
                with self.subTest(source=source, expires=expires):
                    with self.assertRaisesRegex(ValueError, 'истёк'):
                        self.arm._send('ARM', source_ns=source, expires_ns=expires, position_rad=[0.]*6)
            self.assertEqual(self.node.sent, [])
            self.assertEqual(self.arm.pending, {})
            self.arm._send('ARM', source_ns=900_000_000, expires_ns=1_040_000_000, position_rad=[0.]*6)
        self.assertEqual(self.node.sent[-1]['source_monotonic_ns'], 900_000_000)
        self.assertEqual(self.node.sent[-1]['expires_monotonic_ns'], 1_040_000_000)

    def test_activity_record_keeps_linux_and_controller_boots_separate(self):
        reference = self.arm.reference()
        original_read = Path.read_text
        def read_text(path, *args, **kwargs):
            if str(path) == '/proc/sys/kernel/random/boot_id':
                return 'jetson-linux-boot\n'
            return original_read(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read_text):
            self.arm._persist(reference, 'command_in_progress', ends_monotonic=42.)
        saved = json.loads((self.root/'data/arm-state.json').read_text())
        self.assertEqual(saved['boot_id'], 'jetson-linux-boot')
        self.assertEqual(saved['controller_boot_id'], 123)
        self.assertEqual(reference['boot_id'], 123)
        self.assertEqual(saved['phase'], 'command_in_progress')
        self.assertEqual(saved['raw_ticks'], reference['raw_ticks'])
        self.assertEqual(saved['acquired_monotonic'], reference['acquired_monotonic'])
        self.assertFalse((self.root/'data/arm-state.tmp').exists())

    def test_cancel_ack_waits_for_controller_flags_to_clear(self):
        value=state(self.calibration,1_000_000_000)
        pending=copy.deepcopy(value); pending['controller'].update(arm_enabled=False,arm_cancel_pending=True)
        enabled=copy.deepcopy(value); enabled['controller'].update(arm_enabled=True,arm_cancel_pending=False)
        done=copy.deepcopy(value); done['controller'].update(arm_enabled=False,arm_cancel_pending=False)
        permits=[]
        with patch.object(self.arm,'_state',side_effect=[pending,enabled,done]), \
             patch('native_arm.time.monotonic',side_effect=[1.,1.1,1.2,1.3]), \
             patch('native_arm.time.monotonic_ns',return_value=1_000_000_000), \
             patch('native_arm.time.sleep'):
            self.arm._wait_cancel_complete(123,11,lambda:permits.append(True))
        self.assertEqual(len(permits),3)

    def test_cancel_ack_without_controller_completion_times_out(self):
        value=state(self.calibration,1_000_000_000)
        value['controller'].update(arm_enabled=False,arm_cancel_pending=True)
        with patch.object(self.arm,'_state',return_value=value), \
             patch('native_arm.time.monotonic',side_effect=[1.,1.1,1.2,1.4]), \
             patch('native_arm.time.monotonic_ns',return_value=1_000_000_000), \
             patch('native_arm.time.sleep'):
            with self.assertRaisesRegex(ValueError,'не подтвердила'):
                self.arm._wait_cancel_complete(123,11,lambda:None)

    def test_complete_moveit_path_and_midpoint_velocity_survive_with_appended_gripper(self):
        current = self.arm.reference()
        initial = current['position_rad'][:5]
        middle = initial.copy(); middle[1] += math.radians(5)
        final = initial.copy(); final[1] += math.radians(10)
        goal = self.arm._physical(final+[current['position_rad'][5]])
        goal[5] += 5
        trajectory = dict(names=['arm'+str(i)+'_Joint' for i in range(1, 6)],
                          times=[0., 2., 4.], positions=[initial, middle, final],
                          velocities=[[0.]*5, [0., .02, 0., 0., 0.], [0.]*5],
                          accelerations=[[0.]*5]*3)
        path = self.arm._path(current, goal, trajectory)
        self.assertEqual(path.positions.shape, (4, 6))
        np.testing.assert_allclose(path.positions[1, :5], middle, atol=0, rtol=0)
        self.assertGreater(path.sample(path.times[1])['velocity'][1], 0)
        self.assertAlmostEqual(path.positions[2, 5], current['position_rad'][5])
        self.assertGreater(path.positions[3, 5], path.positions[2, 5])
        np.testing.assert_allclose(path.positions[3, :5], final, atol=0, rtol=0)
        self.assertEqual([r['operation'] for r in self.node.sent], [])


if __name__ == '__main__':
    unittest.main()
