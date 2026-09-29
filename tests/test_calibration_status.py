import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from calibration_status import status


class CalibrationStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'config').mkdir();(self.root/'data').mkdir()

    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value))

    def test_scoped_acceptance_unlocks_manual_work_but_not_delivery(self):
        evidence=self.root/'data/evidence.json';evidence.write_text('{}')
        digest=hashlib.sha256(evidence.read_bytes()).hexdigest()
        records={name:{'state':'accepted','evidence':['data/evidence.json'],
                       'evidence_sha256':{'data/evidence.json':digest}} for name in
                 ('chassis','command_validity_stop','lidar_geometry','arm_motion','hand_eye')}
        records.update(localization={'state':'partial'},arm_feedback={'state':'missing'},
                       gripper={'state':'missing'},pick_and_deliver={'state':'blocked'})
        self.write('config/calibration-acceptance.json',{'updated_at':'now','records':records})
        self.write('data/status.json',{'at':100})
        result=status(self.root,101)
        available={item['id']:item['available'] for item in result['unlocked']}
        self.assertTrue(available['manual_holonomic_drive'])
        self.assertTrue(available['gamepad_base_arm'])
        self.assertTrue(available['stationary_camera_arm_geometry'])
        delivery=next(item for item in result['blocked'] if item['id']=='pick_and_deliver')
        self.assertEqual(delivery['blocked_by'],['localization','arm_feedback','gripper'])
        self.assertTrue(result['robot_live'])

    def test_missing_evidence_does_not_unlock(self):
        self.write('config/calibration-acceptance.json',{'records':{}})
        result=status(self.root,100)
        self.assertFalse(any(item['available'] for item in result['unlocked']))
        self.assertFalse(result['robot_live'])

    def test_changed_evidence_revokes_effective_acceptance(self):
        path=self.root/'data/evidence.json';path.write_text('changed')
        record={'state':'accepted','evidence':['data/evidence.json'],
                'evidence_sha256':{'data/evidence.json':'0'*64}}
        self.write('config/calibration-acceptance.json',{'records':{'chassis':record}})
        result=status(self.root,100)
        self.assertNotIn('chassis',result['accepted'])
        self.assertFalse(result['records']['chassis']['evidence_valid'])


if __name__=='__main__':unittest.main()
