import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
from command_arm_state import describe,execution_state,SOURCE_SHA
from controller_feedback import vendor_calibration
from manual_reference_host import reference_from_state
from delivery_vision import CommandHistory,JointSamplePending
from gamepad_panel import GamepadPanel


def snapshot():
    cal=vendor_calibration();raw=[2000,2000,2000,2000,1487,2850]
    ref=dict(phase='READY',error=0,error_name='EX_OK',boot=12,session=5,generation=1,reference_generation=1,
        received_monotonic_ns=1_000_000_000,reference_valid=True,goal_verified_mask=63,torque_on_mask=63,
        raw_sent=raw,raw_reference=list(raw),slopes=[c.radians_per_tick for c in cal],
        offsets=[c.radians_at_raw_zero for c in cal],lower=[c.lower for c in cal],upper=[c.upper for c in cal])
    return dict(identity=dict(boot=12,source_sha256=SOURCE_SHA),telemetry_fresh=True,session_state='active',
                monotonic_ns=1_000_000_000,manual_reference=ref,controller=dict(session=5,fault=0,mode=1))


class CommandContractTests(unittest.TestCase):
    def setUp(self):self.s=snapshot();self.cal=vendor_calibration()
    def describe(self):return describe(self.s,self.cal,1_050_000_000)
    def test_capture_is_not_measurement_and_all_six_are_required(self):
        r=self.describe();self.assertTrue(r['command_enabled']);self.assertFalse(r['measured'])
        self.assertEqual(r['state_source'],'operator_reference');self.assertEqual(len(r['q_estimated']),6)
        self.s['manual_reference']['torque_on_mask']=31
        self.assertFalse(self.describe()['command_enabled']);self.assertIsNone(self.describe()['q_estimated'])
    def test_sent_sample_not_goal_advances_estimate(self):
        ref=self.s['manual_reference'];ref.update(phase='EXECUTING',generation=2,position_rad=[1.]*6)
        before=self.describe()['q_estimated'];ref['raw_sent'][0]-=10
        r=self.describe();self.assertNotEqual(before[0],r['q_estimated'][0]);self.assertNotEqual(r['q_estimated'],ref['position_rad'])
        self.assertEqual(r['state_source'],'command_estimate');self.assertFalse(r['measured'])
    def test_error_or_stale_or_wrong_identity_drops_pose(self):
        for path,value in [('error',14),('reference_valid',False),('received_monotonic_ns',0)]:
            self.s=snapshot();self.s['manual_reference'][path]=value
            self.assertFalse(self.describe()['command_enabled']);self.assertIsNone(self.describe()['q_estimated'])
        self.s=snapshot();self.s['identity']['source_sha256']='0'*64
        self.assertEqual(self.describe()['phase'],'UNKNOWN')
    def test_negative_ros_angle_is_preserved_and_raw_limits_remain(self):
        self.s['manual_reference']['raw_sent'][0]=1900
        r=reference_from_state(self.s,self.cal,1_050_000_000)
        self.assertLess(r['position_rad'][0],0);self.assertFalse(r['measured'])
        self.s['manual_reference']['raw_sent'][0]=65000
        self.assertFalse(self.describe()['command_enabled'])
    def test_camera_history_uses_last_sent_not_future_sample_and_clears_reference(self):
        h=CommandHistory();self.s['arm']=self.describe();h.add(self.s)
        old=h.at(1.02,12)
        self.s=copy.deepcopy(self.s);self.s['monotonic_ns']=1_040_000_000
        self.s['arm']['servo_deg'][0]+=3;h.add(self.s)
        self.assertEqual(h.at(1.02,12),old)
        self.s['arm']['reference_valid']=False;h.add(self.s)
        with self.assertRaises(JointSamplePending):h.at(1.05,12)
    def test_curve_metadata_never_replaces_sent_estimate(self):
        self.s['manual_reference'].update(phase='EXECUTING',generation=2)
        arm=self.describe();before=list(arm['q_estimated'])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir()
            p=root/'data/arm-state.json'
            p.write_text(json.dumps(dict(phase='command_in_progress',controller_boot_id=12,session=5,
                command_generation=2,q_goal=[.1]*6,trajectory={'duration':1},trajectory_time=.2)))
            result=execution_state(arm,root)
            self.assertEqual(result['q_estimated'],before);self.assertEqual(result['q_goal'],[.1]*6)
            self.assertEqual(result['trajectory_time'],.2)
