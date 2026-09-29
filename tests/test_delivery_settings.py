import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from delivery_robot import load_settings

class SettingsTests(unittest.TestCase):
    def fixture(self,root):
        (root/'config').mkdir();(root/'data').mkdir()
        config=dict(transport_deg=[90]*6,search_deg=[90]*6,search_places=['sock'],destination='basket',
            floor_plane_base=[0,0,1,0],grasp_quaternion_xyzw=[0,0,0,1],approach_height_m=.04,
            lift_height_m=.08,grasp_tcp_offset_m=0,gripper_linkage_rad=-.5,
            drop_zone=dict(center_xyz=[.3,0,.02],radius_m=.1,support_tolerance_m=.01))
        files={'config/delivery.json':config,
            'config/handeye-accepted.json':dict(execution_authorized=True,measured_joint_positions=True,reference_mount='arm4',
                camera_to_mount_reference=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
            'config/gripper-accepted.json':dict(execution_authorized=True,aperture_mm_calibrated=True,open_deg=30,sock_close_deg=160),
            'config/explorer.urdf':'test geometry', 'config/explorer.srdf':'test groups',
            'data/test.json':dict(hardware_executed=True,simulation=False,kind='controller',outcome='passed',firmware_source_sha256='a'*64,controller_calibration_sha256='b'*64)}
        files['data/arm.json']=dict(files['data/test.json'],kind='arm')
        files['data/geometry.json']=dict(files['data/test.json'],kind='geometry')
        for name,value in files.items():(root/name).write_text(json.dumps(value))
        digest=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        evidence=dict(accepted=True,firmware_source_sha256='a'*64,controller_calibration_sha256='b'*64,
            artifacts={name:digest(name) for name in files if name.startswith('config/')},
            physical_test_records={name:digest(name) for name in files if name.startswith('data/')})
        (root/'config/delivery-acceptance.json').write_text(json.dumps(evidence))
    def test_mismatched_firmware_and_modified_geometry_cannot_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            self.assertEqual(load_settings(root,'a'*64,'b'*64)['destination'],'basket')
            with self.assertRaises(ValueError):load_settings(root,'c'*64,'b'*64)
            (root/'config/explorer.urdf').write_text('changed')
            with self.assertRaises(ValueError):load_settings(root,'a'*64,'b'*64)
    def test_old_reference_calibration_is_not_execution_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            path=root/'config/handeye-accepted.json';data=json.loads(path.read_text());data['execution_authorized']=False
            path.write_text(json.dumps(data));evidence=root/'config/delivery-acceptance.json';e=json.loads(evidence.read_text())
            e['artifacts']['config/handeye-accepted.json']=hashlib.sha256(path.read_bytes()).hexdigest();evidence.write_text(json.dumps(e))
            with self.assertRaises(ValueError):load_settings(root,'a'*64,'b'*64)

    def test_computer_only_result_cannot_admit_physical_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            path=root/'data/arm.json';value=json.loads(path.read_text());value['simulation']=True;path.write_text(json.dumps(value))
            evidence=root/'config/delivery-acceptance.json';data=json.loads(evidence.read_text())
            data['physical_test_records']['data/arm.json']=hashlib.sha256(path.read_bytes()).hexdigest();evidence.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_settings(root,'a'*64,'b'*64)
