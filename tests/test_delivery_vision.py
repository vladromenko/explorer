import unittest
import threading
from collections import deque
from types import SimpleNamespace
import cv2
import numpy as np
from delivery_vision import FeatureObject,JointHistory,JointSamplePending,MeasuredVision,MAX_PENDING_FRAMES

class VisionTests(unittest.TestCase):
    def sample(self,t=1.,dx=0):
        rng=np.random.default_rng(33)
        rgb=np.zeros((120,160,3),np.uint8)
        rgb[40:80,60+dx:100+dx]=rng.integers(0,256,(40,40,3),np.uint8)
        depth=np.ones((120,160));depth[40:80,60+dx:100+dx]=.98
        return dict(rgb=rgb,depth=depth,k=np.array([[200.,0,80],[0,200,60],[0,0,1.]]),
                    d=np.zeros(5),stamp=t,frame='camera')
    def test_real_feature_continuity_not_detector_label_id(self):
        track=FeatureObject(self.sample(),[59,39,101,81])
        result=track.update(self.sample(1.1,2))
        self.assertEqual(result['object_id'],track.object_id)
        self.assertGreater(result['association_fraction'],.85)
        self.assertAlmostEqual(result['point_camera'][2],.98)
        with self.assertRaises(ValueError):track.update(self.sample(2,2))
        with self.assertRaises(ValueError):track.update(self.sample(2.1,2))
    def test_occlusion_never_inherits_old_identity(self):
        track=FeatureObject(self.sample(),[59,39,101,81])
        sample=self.sample(1.1);sample['rgb'][:]=0
        with self.assertRaises(ValueError):track.update(sample)
    def test_camera_pose_interpolation_never_extrapolates_or_crosses_boot(self):
        history=JointHistory()
        for index in range(1,7):
            for t,q in [(1000000000,-10.),(1020000000,-9.)]:
                history.add(dict(boot_id=1,joint=index,position_valid=True,error=0,device_error=0,
                                 acquired_monotonic_ns=t,physical_deg=q))
        np.testing.assert_allclose(history.at(1.01,1),[-9.5]*6)
        with self.assertRaises(ValueError):history.at(1.1,1)
        with self.assertRaises(ValueError):history.at(1.01,2)
        history.add(dict(boot_id=1,joint=3,position_valid=False,error=14,device_error=0))
        with self.assertRaises(ValueError):history.at(1.01,1)

    def test_reacquisition_uses_true_timestamp_and_new_identity(self):
        from delivery_vision import reacquire_candidate
        old=self.sample(1);fresh=self.sample(20)
        candidate=reacquire_candidate(old,fresh,[59,39,101,81])
        self.assertEqual(candidate['image_stamp'],20)
        self.assertFalse(candidate['continuous_from_previous_image'])
        self.assertFalse(candidate['semantic_identity_verified'])
        a=FeatureObject(old,candidate['bbox']);b=FeatureObject(fresh,candidate['bbox'])
        self.assertNotEqual(a.object_id,b.object_id)
        changed=self.sample(20,8)
        with self.assertRaises(ValueError):reacquire_candidate(old,changed,candidate['bbox'])

    def test_future_joint_sample_wait_differs_from_real_feedback_or_geometry_error(self):
        history=JointHistory()
        def add(joint,t,q=-10):history.add(dict(boot_id=1,joint=joint,position_valid=True,error=0,device_error=0,
            acquired_monotonic_ns=int(t*1e9),physical_deg=q))
        for joint in range(1,7):add(joint,1.)
        with self.assertRaises(JointSamplePending):history.at(1.01,1)
        for joint in range(1,7):add(joint,1.02,-9)
        np.testing.assert_allclose(history.at(1.01,1),[-9.5]*6)
        history.add(dict(boot_id=1,joint=3,position_valid=False,error=14,device_error=0))
        try:history.at(1.01,1)
        except ValueError as exc:self.assertNotIsInstance(exc,JointSamplePending)
        else:self.fail('Actual feedback error accepted as a pose')
        for joint in range(1,7):add(joint,1.2)
        try:history.at(1.1,1)
        except ValueError as exc:self.assertNotIsInstance(exc,JointSamplePending)
        else:self.fail('Measurement gap accepted as a pose')

    def measured_vision(self):
        vision=MeasuredVision.__new__(MeasuredVision)
        vision.lock=threading.RLock();vision.generation=0;vision.frames=deque(maxlen=400)
        vision.pending=deque();vision.track=None;vision.settings=None;vision.error=None
        vision.last_stamp=0.;vision.enqueued_stamp=0.;vision.waiting_for_joints=None
        vision.maps=SimpleNamespace(pose=lambda:dict(x=0,y=0,yaw=0))
        updated=[]
        def update(sample):
            updated.append(float(sample['stamp']))
            return dict(object_id='original-object',point_camera=np.array([.2,0,.04]),association_fraction=1.,
                        object_extent_camera_m=np.array([.03,.02,.02]),object_position_uncertainty_m=.002)
        tracker=SimpleNamespace(stamp=1.,update=update)
        vision.start_tracker(tracker,dict(floor_plane_base=[0,0,1,0],open_deg=30))
        vision.geometry=lambda *args:(np.eye(4),np.zeros(3),[90,90,90,90,90,30])
        return vision,updated

    def test_pending_frame_retried_with_original_exposure_then_queue_drains_in_order(self):
        vision,updated=self.measured_vision();ready=False;queried=[]
        def geometry(sample,settings):
            queried.append(sample['stamp'])
            if not ready:raise JointSamplePending('right measurement not arrived')
            return np.eye(4),np.zeros(3),[90,90,90,90,90,30]
        vision.geometry=geometry
        self.assertEqual(vision.process_snapshot(dict(stamp=1.02),now=0.),0)
        self.assertIsNone(vision.error);self.assertTrue(vision.waiting_for_joints)
        self.assertEqual(vision.process_snapshot(dict(stamp=1.04),now=.04),0)
        self.assertEqual(len(vision.pending),2);self.assertEqual(updated,[])
        ready=True
        self.assertEqual(vision.process_snapshot(dict(stamp=1.04),now=.05),2)
        self.assertEqual(updated,[1.02,1.04]);self.assertEqual([f['at'] for f in vision.frames],[1.02,1.04])
        self.assertEqual(queried,[1.02,1.02,1.02,1.04])
        self.assertIsNone(vision.waiting_for_joints);self.assertIsNone(vision.error)

    def test_pending_joint_bracket_deadline_is_bounded_and_latches_failure(self):
        vision,updated=self.measured_vision()
        def pending(*args):raise JointSamplePending('wait')
        vision.geometry=pending;vision.process_snapshot(dict(stamp=1.02),now=0.)
        with self.assertRaisesRegex(ValueError,'deadline'):vision.process_snapshot(dict(stamp=1.03),now=.101)
        self.assertTrue(vision.error);self.assertFalse(vision.pending);self.assertEqual(updated,[])
        vision.geometry=lambda *args:(np.eye(4),np.zeros(3),[90]*6)
        self.assertEqual(vision.process_snapshot(dict(stamp=1.04),now=.11),0)
        self.assertTrue(vision.error)

    def test_real_geometry_error_is_not_cleared_by_later_successful_input(self):
        vision,updated=self.measured_vision()
        def invalid(*args):raise ValueError('controller boot changed')
        vision.geometry=invalid
        with self.assertRaisesRegex(ValueError,'boot changed'):vision.process_snapshot(dict(stamp=1.02),now=0.)
        vision.geometry=lambda *args:(np.eye(4),np.zeros(3),[90]*6)
        self.assertEqual(vision.process_snapshot(dict(stamp=1.03),now=.02),0)
        self.assertEqual(updated,[]);self.assertEqual(vision.error,'controller boot changed')

    def test_pending_queue_has_fixed_capacity(self):
        vision,updated=self.measured_vision()
        def pending(*args):raise JointSamplePending('wait')
        vision.geometry=pending
        for index in range(MAX_PENDING_FRAMES):vision.process_snapshot(dict(stamp=1.01+index*.001),now=.001*index)
        with self.assertRaisesRegex(ValueError,'overflow'):
            vision.process_snapshot(dict(stamp=1.03),now=.01)
        self.assertEqual(updated,[]);self.assertFalse(vision.pending)

    def test_cancelled_generation_cannot_publish_late_geometry_result(self):
        vision,updated=self.measured_vision()
        def cancelled(*args):
            vision.stop()
            return np.eye(4),np.zeros(3),[90]*6
        vision.geometry=cancelled
        self.assertEqual(vision.process_snapshot(dict(stamp=1.02),now=0.),0)
        self.assertEqual(updated,[]);self.assertFalse(vision.frames);self.assertFalse(vision.pending)

    def test_command_estimate_is_never_labeled_camera_or_gripper_measurement(self):
        vision,_=self.measured_vision();vision.command_mode=True
        vision.process_snapshot(dict(stamp=1.02),now=0.)
        frame=vision.frames[-1]
        self.assertFalse(frame['camera_pose_measured'])
        self.assertIsNone(frame['gripper_open_measured'])
        self.assertTrue(frame['gripper_open_commanded'])
        self.assertTrue(frame['gripper_open_estimated'])
        self.assertIsNone(frame['gripper_aperture_measured_deg'])
        self.assertEqual(frame['gripper_aperture_command_deg'],30.)
        self.assertEqual(frame['camera_pose_source'],'command_estimate')

    def test_factory_geometry_uses_only_settled_command_estimate(self):
        vision=MeasuredVision.__new__(MeasuredVision);vision.factory_mode=True
        import tempfile,json
        from pathlib import Path
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        vision.root=Path(temporary.name);(vision.root/"data").mkdir()
        (vision.root/"data/arm-state.json").write_text(json.dumps(dict(at=10.,runtime_ms=1000,
            servo_deg=[90]*6,phase="command_elapsed_observation_required",boot_id="current")))
        vision.arm=SimpleNamespace(boot="current")
        model=SimpleNamespace(lock=threading.Lock(),state=SimpleNamespace(get_global_link_transform=lambda name:np.eye(4)),
                              set_state=lambda *args:None)
        vision.model=lambda:model
        settings=dict(gripper_linkage_rad=0.,camera_to_mount=np.eye(4))
        with self.assertRaises(JointSamplePending):vision.geometry(dict(stamp=11.05),settings)
        transform,tcp,angles=vision.geometry(dict(stamp=11.2),settings)
        np.testing.assert_allclose(transform,np.eye(4));self.assertEqual(angles,[90]*6)
