from collections import deque
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import numpy as np
from delivery_vision import MeasuredVision
from factory_arm_observation import settled_reference,CommandPosePending


class FactoryObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);(self.root/"data").mkdir()
        self.state=dict(at=10.,runtime_ms=200,servo_deg=[90]*6,boot_id="actual",
            phase="command_in_progress",source="timed_factory_command_estimate")
        self.save()

    def save(self):
        (self.root/"data/arm-state.json").write_text(json.dumps(self.state))

    def test_pending_pose_and_settled_exposure(self):
        with self.assertRaises(CommandPosePending):settled_reference(self.root,"actual",10.4)
        self.state["phase"]="command_elapsed_observation_required";self.save()
        with self.assertRaises(CommandPosePending):settled_reference(self.root,"actual",10.21)
        result=settled_reference(self.root,"actual",10.4)
        self.assertFalse(result["measured"]);self.assertFalse(result["attained"])

    def test_boot_and_link_fault_are_real_failures(self):
        with self.assertRaisesRegex(ValueError,"boot"):settled_reference(self.root,"different",10.4)
        (self.root/"data/arm-telemetry-fault.json").write_text(json.dumps(dict(at=11.)))
        with self.assertRaisesRegex(ValueError,"link fault"):settled_reference(self.root,"actual",10.4)

    def test_motion_keeps_2d_association_without_publishing_metric_coordinates(self):
        vision=MeasuredVision.__new__(MeasuredVision)
        vision.root=self.root;vision.arm=SimpleNamespace(boot="actual")
        vision.factory_mode=True;vision.command_mode=True
        vision.lock=threading.RLock();vision.generation=0;vision.frames=deque()
        vision.pending=deque();vision.enqueued_stamp=10.;vision.error=None;vision.waiting_for_joints=None
        vision.settings=dict(floor_plane_base=[0,0,1,0],open_deg=90)
        vision.maps=SimpleNamespace(pose=lambda:dict(x=1,y=2,yaw=.1))
        track=SimpleNamespace(stamp=10.,object_id="same-object",ended=False)
        def update(sample):
            track.stamp=sample["stamp"]
            return dict(point_camera=np.array([.2,0,.04]),object_id=track.object_id,
                association_fraction=.9,object_extent_camera_m=np.array([.03,.02,.02]))
        track.update=update;vision.track=track
        vision.geometry=lambda *args:(np.eye(4),np.zeros(3),[90]*6)
        self.assertEqual(vision.process_snapshot(dict(stamp=10.1),now=0.),0)
        self.assertEqual(track.stamp,10.1);self.assertFalse(vision.frames);self.assertFalse(vision.pending)
        self.assertEqual(vision.process_snapshot(dict(stamp=10.3),now=1.),0)
        self.assertIsNone(vision.error)
        self.state["phase"]="command_elapsed_observation_required";self.save()
        self.assertEqual(vision.process_snapshot(dict(stamp=10.5),now=2.),1)
        self.assertEqual(vision.frames[-1]["at"],10.5)
        self.assertFalse(vision.frames[-1]["camera_pose_measured"])
