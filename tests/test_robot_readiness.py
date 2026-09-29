import json
from pathlib import Path
import tempfile
import unittest
from robot_readiness import status

class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.write('config/controller-profile.json',dict(firmware_source_sha256='new',hardware_accepted=True,telemetry_only=False))
        self.write('config/controller-releases.json',dict(staged_source_sha256='new',entries={'new':dict(version='fixture')}))
        self.write('data/controller-state.json',dict(at=100.,telemetry_fresh=True,identity=dict(source_sha256='new')))
        for c in ('navigation','planning'):self.write('data/'+c+'-health.json',dict(at=100.,stage='active',inputs_ready=True))
    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(value))
    def test_prepared_is_not_physical_acceptance(self):
        self.write('config/controller-profile.json',dict(firmware_source_sha256='new',hardware_accepted=False,telemetry_only=True))
        s=status(self.root,dict(blocked_by=[]),101.)
        self.assertFalse(s['delivery_ready']);self.assertTrue(s['physical_acceptance_required'])
    def test_old_or_stale_controller_cannot_be_ready(self):
        self.write('data/controller-state.json',dict(at=100.,telemetry_fresh=True,identity=dict(source_sha256='old')))
        s=status(self.root,dict(blocked_by=[]),101.);self.assertTrue(s['firmware_installation_pending']);self.assertFalse(s['delivery_ready'])
        self.assertFalse(status(self.root,dict(blocked_by=[]),110.)['controller_connected'])
    def test_navigation_and_delivery_evidence_both_required(self):
        self.assertTrue(status(self.root,dict(blocked_by=[]),101.)['delivery_ready'])
        self.assertFalse(status(self.root,dict(blocked_by=['measured geometry missing']),101.)['delivery_ready'])
        self.write('data/navigation-health.json',dict(at=100.,stage='starting',inputs_ready=True))
        self.assertFalse(status(self.root,dict(blocked_by=[]),101.)['delivery_ready'])
if __name__=='__main__':unittest.main()
