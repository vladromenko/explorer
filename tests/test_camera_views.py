import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import numpy as np
from camera_views import CameraViews,VIEWS
from servo_coordinates import to_radians

class CameraViewTests(unittest.TestCase):
    def test_left_right_follow_factory_base_joint_sign(self):
        self.assertGreater(to_radians(VIEWS["left"])[0],0.)
        self.assertLess(to_radians(VIEWS["right"])[0],0.)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/"data").mkdir();(self.root/"config").mkdir()
        self.start=[*VIEWS["forward"],77]
        (self.root/"data/arm-state.json").write_text(json.dumps(dict(boot_id="boot",phase="command_elapsed_observation_required",servo_deg=self.start)))
        self.arm=SimpleNamespace(boot="boot",reference=lambda:{"servo_deg":self.start})
        self.camera=CameraViews(self.root,self.arm,SimpleNamespace(),lambda:None)
    def test_fresh_camera_required_and_not_borrowed_from_external_camera(self):
        path=self.root/"data/perception.json";path.write_text(json.dumps({"image_stamp":time.time()-3}))
        with self.assertRaisesRegex(ValueError,"бортовой камеры"):self.camera.fresh_frame()
        path.write_text(json.dumps({"image_stamp":time.time()}));self.camera.fresh_frame()
    def test_camera_must_face_forward_when_base_moves(self):
        (self.root/"data/perception.json").write_text(json.dumps({"image_stamp":time.time()}))
        arm=self.root/"data/arm-state.json"
        value=dict(boot_id="boot",phase="command_elapsed_observation_required",servo_deg=self.start)
        arm.write_text(json.dumps(value));self.camera.navigation_guard({"velocity":[.05,0.,0.]})
        value["servo_deg"]=[*VIEWS["left"],77];arm.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"вперёд"):self.camera.navigation_guard({"velocity":[.05,0.,0.]})
        self.camera.navigation_guard({"velocity":[0.,0.,0.]})
    def test_existing_gripper_command_is_preserved(self):
        self.camera.geometry=lambda goal:{"measured_joints":False}
        self.camera.fresh_frame=lambda after:time.time()
        result=self.camera.move("forward",lambda:None)
        self.assertEqual(result["servo_deg"][-1],77)
        self.assertFalse(result["measured_joints"])
    def test_fault_and_nonfinite_estimate_stop_navigation(self):
        (self.root/"data/perception.json").write_text(json.dumps({"image_stamp":time.time()}))
        path=self.root/"data/arm-state.json";value=json.loads(path.read_text())
        value["servo_deg"][0]=float("nan");path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"Invalid"):self.camera.navigation_guard({"velocity":[.05,0.,0.]})
        value["servo_deg"][0]=90;path.write_text(json.dumps(value))
        (self.root/"data/arm-telemetry-fault.json").write_text(json.dumps({"at":time.time()}))
        with self.assertRaisesRegex(ValueError,"link fault"):self.camera.navigation_guard({"velocity":[0.,0.,0.]})
