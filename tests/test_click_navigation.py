import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from click_navigation import TargetFrames, approach_goal, safe_approach_step, selected_depth_point


class ClickNavigationTests(unittest.TestCase):
    def test_click_uses_registered_depth_and_bounded_standoff(self):
        depth=np.full((100,100),2.,dtype=np.float32)
        k=np.array([[500.,0.,50.],[0.,500.,50.],[0.,0.,1.]])
        point,distance,spread=selected_depth_point(depth,k,np.zeros(5),50,50)
        self.assertEqual(distance,2.)
        self.assertEqual(spread,0.)
        transform=[[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]
        goal=approach_goal(point,transform,dict(x=1.,y=2.,yaw=0.))
        self.assertAlmostEqual(goal["x"],1.8)
        self.assertAlmostEqual(goal["y"],2.)
        self.assertEqual(goal["stand_off_m"],.55)

    def test_missing_or_mixed_depth_never_starts_a_goal(self):
        k=np.array([[500.,0.,50.],[0.,500.,50.],[0.,0.,1.]])
        with self.assertRaisesRegex(ValueError,"глубины"):
            selected_depth_point(np.zeros((100,100)),k,np.zeros(5),50,50)
        mixed=np.full((100,100),2.,dtype=np.float32);mixed[43:58,43:50]=1.
        with self.assertRaisesRegex(ValueError,"неоднозначна"):
            selected_depth_point(mixed,k,np.zeros(5),50,50)

    def test_blocked_full_path_uses_only_planner_approved_short_step(self):
        goal=dict(x=.8,y=0.,start_x=0.,start_y=0.,advance_m=.8)
        checked=[]
        def preview(x,y):
            checked.append(x)
            if x>.11:raise ValueError("No safe path: 208")
        selected=safe_approach_step(goal,preview)
        self.assertAlmostEqual(selected["advance_m"],.1)
        self.assertTrue(selected["partial_approach"])
        self.assertEqual(len(checked),5)

    def test_frame_binding_rejects_arm_motion_and_map_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/"data").mkdir()
            now=time.time();pose=dict(x=0.,y=0.,yaw=0.)
            (root/"data/status.json").write_text(json.dumps(dict(at=now,stop_latched=True,odom_velocity=[0.,0.,0.])))
            transform=[[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]
            (root/"data/camera-view.json").write_text(json.dumps(dict(phase="ready",view="forward",servo_deg=[90]*6,
                settled_at=now-5,camera_to_base_estimate=transform)))
            arm=dict(servo_deg=[90]*6,updated_at=now-5)
            (root/"data/arm-state.json").write_text(json.dumps(arm))
            np.savez_compressed(root/"data/rgbd-snapshot.npz",rgb=np.zeros((100,100,3),dtype=np.uint8),
                depth=np.full((100,100),2.,dtype=np.float32),
                k=np.array([[500.,0.,50.],[0.,500.,50.],[0.,0.,1.]]),d=np.zeros(5),
                stamp=now,frame="camera_color_optical_frame")
            epoch=["map-a"]
            frames=TargetFrames(root,lambda:pose,lambda:epoch[0])
            identifier,_=frames.capture()
            self.assertAlmostEqual(frames.goal(identifier,50,50)["x"],.8)
            epoch[0]="map-b"
            with self.assertRaisesRegex(ValueError,"карта"):
                frames.goal(identifier,50,50)
            epoch[0]="map-a";arm["servo_deg"][0]=91
            (root/"data/arm-state.json").write_text(json.dumps(arm))
            with self.assertRaisesRegex(ValueError,"камеру руки"):
                frames.goal(identifier,50,50)
