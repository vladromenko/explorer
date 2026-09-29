import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from controller_arm_commissioning import arm_commissioning_allowed, require_arm_test_power
from controller_feedback import vendor_calibration


class ArmPermitTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);(self.root/'config').mkdir();(self.root/'data').mkdir()
        (self.root/'config/controller-calibration.json').write_text('{}')
        self.digest=hashlib.sha256(b'{}').hexdigest();self.now=1_000_000_000
        self.profile=dict(firmware_source_sha256='a'*64,calibration_sha256=self.digest,telemetry_only=True)
        (self.root/'data/status.json').write_text(json.dumps(dict(at=time.time(),battery=12.,
            power=dict(motion_allowed=True),sensor_age=dict(battery=.01))))
        self.identity=dict(source_sha256='a'*64,uid='1'*24,boot=7)
        self.state=dict(wheels=[dict(pwm=0,target_rad_s=0,measured_rad_s=0) for _ in range(4)])
        self.cal=vendor_calibration()
        self.samples=[dict(joint=i+1,raw_ticks=2000,position_valid=True,raw_valid=True,
            error=0,device_error=0,acquired_monotonic_ns=self.now) for i in range(6)]
        self.request=dict(source_id='test',operation='ARM_ENABLE')
        self.permit=dict(source_id='test',issued_ns=self.now-1,expires_ns=self.now+1_000_000_000,
            calibration_sha256=self.digest,**self.identity)
    def save(self):
        (self.root/'data/controller-arm-permit.json').write_text(json.dumps(self.permit))
    def allowed(self):
        return arm_commissioning_allowed(self.root,self.profile,self.identity,self.state,
            self.samples,self.cal,self.request,self.now)
    def test_only_exact_owner_and_arm_scope(self):
        self.assertFalse(self.allowed());self.save();self.assertTrue(self.allowed())
        self.request['source_id']='other';self.assertFalse(self.allowed())
        self.request.update(source_id='test',operation='BASE')
        with self.assertRaisesRegex(ValueError,'operation'):self.allowed()
        self.assertTrue(self.profile['telemetry_only'])
        self.assertNotIn('hardware_accepted',self.profile)
    def test_expiry_reboot_calibration_and_motion(self):
        self.save();self.permit['expires_ns']=self.now;self.save()
        with self.assertRaisesRegex(ValueError,'expired'):self.allowed()
        self.permit['expires_ns']=self.now+1;self.permit['boot']=8;self.save()
        with self.assertRaisesRegex(ValueError,'boot'):self.allowed()
        self.permit['boot']=7;self.save();self.state['wheels'][0]['pwm']=.1
        with self.assertRaisesRegex(ValueError,'stationary'):self.allowed()
        self.state['wheels'][0]['pwm']=0;(self.root/'config/controller-calibration.json').write_text('{ }')
        with self.assertRaisesRegex(ValueError,'calibration'):self.allowed()
    def test_stale_joint_and_short_measured_target(self):
        self.save();self.samples[5]['acquired_monotonic_ns']=self.now-250_000_000
        with self.assertRaisesRegex(ValueError,'measurement'):self.allowed()
        self.samples[5]['acquired_monotonic_ns']=self.now
        self.request.update(operation='ARM',position_rad=[c.observe(2000)['position_rad'] for c in self.cal])
        self.assertTrue(self.allowed());self.request['position_rad'][0]+=.04
        with self.assertRaisesRegex(ValueError,'excursion'):self.allowed()
    def test_no_unapproved_recovery_or_navigation_requirement(self):
        self.save();self.samples[2]['raw_ticks']=3576
        self.request.update(operation='ARM_RECOVER',position_rad=[c.observe(s['raw_ticks'])['position_rad'] for c,s in zip(self.cal,self.samples)])
        with self.assertRaisesRegex(ValueError,'corridor'):self.allowed()
        self.request['operation']='ARM_CANCEL';self.samples=[]
        self.assertTrue(self.allowed())

    def test_power_policy_is_not_bypassed_by_arm_permit(self):
        self.save()
        self.request.update(operation='ARM',position_rad=[c.observe(2000)['position_rad'] for c in self.cal])
        (self.root/'data/status.json').write_text(json.dumps(dict(at=time.time(),battery=10.47,
            power=dict(motion_allowed=False),sensor_age=dict(battery=.01))))
        with self.assertRaisesRegex(ValueError,'adequate power'):self.allowed()

    def test_power_preflight_rejects_stale_or_charging_even_if_allowed(self):
        status=dict(at=time.time(),battery=12.,power=dict(motion_allowed=True),
                    sensor_age=dict(battery=.01))
        path=self.root/'data/status.json'
        path.write_text(json.dumps(status));require_arm_test_power(self.root)
        status['power']['charging']=True;path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError,'adequate power'):require_arm_test_power(self.root)
        status['power']['charging']=False;status['at']=time.time()-2
        path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError,'adequate power'):require_arm_test_power(self.root)
        status['at']=time.time();status['sensor_age']['battery']=3
        path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError,'adequate power'):require_arm_test_power(self.root)

if __name__=='__main__':unittest.main()
