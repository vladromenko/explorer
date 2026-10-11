import unittest
import threading
from collections import deque
from types import SimpleNamespace
import cv2
import numpy as np
from delivery_vision import FeatureObject,JointHistory,JointSamplePending,MeasuredVision,MAX_PENDING_FRAMES

class VisionTests(unittest.TestCase):
    def test_visibility_preflight_distinguishes_reach_from_valid_camera_observation(self):
        model = SimpleNamespace(lock=threading.RLock(), set_state=lambda *args: None,
            state=SimpleNamespace(get_global_link_transform=lambda name: np.eye(4)))
        vision = MeasuredVision.__new__(MeasuredVision)
        vision.model = lambda: model
        vision.snapshot = self.sample
        transform = np.eye(4); transform[:3, 3] = [.1, 0, .02]
        settings = {"gripper_linkage_rad": -.8, "camera_to_mount": transform.tolist()}
        visible = vision.preview_visibility([.1, 0, .52], [90] * 5, settings)
        self.assertTrue(visible["observable"])
        self.assertFalse(visible["measured"])
        close = vision.preview_visibility([.1, 0, .12], [90] * 5, settings)
        self.assertTrue(close["in_image"])
        self.assertFalse(close["within_tracking_depth_policy"])
        outside = vision.preview_visibility([.1, .4, .52], [90] * 5, settings)
        self.assertFalse(outside["in_image"])
        self.assertFalse(outside["observable"])

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
        pixel = np.rint(result["grasp_pixel"]).astype(int)
        self.assertAlmostEqual(self.sample(1.1,2)["depth"][pixel[1],pixel[0]], .98)
        projected = np.array([pixel[0]-80, pixel[1]-60, 200.])*.98/200.
        np.testing.assert_allclose(result["grasp_point_camera"], projected, atol=.005)
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

    def test_grasp_surface_has_its_own_transformed_coordinate_not_the_centroid(self):
        vision, updated = self.measured_vision()
        vision.track.update = lambda sample: dict(object_id="object", point_camera=[.2,0,.04],
            grasp_point_camera=[.23,.02,.045], grasp_pixel=[50,60],
            grasp_selection="observed_associated_foreground_feature", association_fraction=1.)
        transform = np.eye(4)
        transform[:3,3] = [.1,-.2,.3]
        vision.geometry = lambda *args: (transform, np.zeros(3), [90,90,90,90,90,30])
        self.assertEqual(vision.process_snapshot(dict(stamp=1.02), now=0.), 1)
        frame = vision.frames[-1]
        np.testing.assert_allclose(frame["object_xyz"], [.3,-.2,.34])
        np.testing.assert_allclose(frame["grasp_xyz"], [.33,-.18,.345])
        self.assertFalse(frame["grasp_contact_verified"])

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


def cloth_sample(stamp=1., color=25, dx=0):
    rgb = np.full((120,160,3), 140, np.uint8)
    rgb[40:80,60+dx:100+dx] = color
    depth = np.ones((120,160))
    depth[40:80,60+dx:100+dx] = .98
    return {"rgb":rgb,"depth":depth,"k":np.array([[200.,0,80],[0,200,60],[0,0,1.]]),
        "d":np.zeros(5),"stamp":stamp,"frame":"camera"}


def test_stationary_cloth_has_real_depth_without_invented_feature_tracking():
    from delivery_vision import StationaryDepthObject, reacquire_candidate
    context = lambda sample: {"epoch":("boot",.5,"path"),"servo_deg":[90]*6,"odom_xyyaw":[0,0,0]}
    old, fresh = cloth_sample(), cloth_sample(1.1)
    candidate = reacquire_candidate(old, fresh, [59,39,101,81])
    assert candidate["appearance_correlation"] is None
    assert candidate["continuous_from_previous_image"] is False
    tracker = StationaryDepthObject(old,[59,39,101,81],context)
    observation = tracker.update(fresh)
    assert observation["association_kind"] == "stationary_depth_and_appearance"
    assert observation["point_camera"][2] == .98
    assert not observation["contact_verified"] and not observation["held_verified"]
    assert StationaryDepthObject(fresh,[59,39,101,81],context).object_id != tracker.object_id


def test_static_cloth_change_or_motion_loses_identity_permanently():
    import pytest
    from delivery_vision import StationaryDepthObject
    base = {"epoch":("boot",.5,"path"),"servo_deg":[90]*6,"odom_xyyaw":[0,0,0]}
    for field, value in (("epoch",("boot",.6,"newpath")),("servo_deg",[91]*6),
                         ("odom_xyyaw",[.004,0,0]),("odom_xyyaw",[0,0,.01])):
        current = dict(base)
        tracker = StationaryDepthObject(cloth_sample(),[59,39,101,81],lambda sample:current)
        current[field] = value
        with pytest.raises(ValueError):tracker.update(cloth_sample(1.1))
        current[field] = base[field]
        with pytest.raises(ValueError,match="already lost"):tracker.update(cloth_sample(1.2))
    for changed in (cloth_sample(1.1,90),cloth_sample(1.1,dx=8)):
        tracker = StationaryDepthObject(cloth_sample(),[59,39,101,81],lambda sample:base)
        with pytest.raises(ValueError):tracker.update(changed)


def test_measured_vision_cloth_fallback_requires_explicit_stationary_context():
    import pytest
    vision = MeasuredVision.__new__(MeasuredVision)
    vision.factory_mode = False
    with pytest.raises(ValueError,match="texture"):
        vision.make_tracker(cloth_sample(),[59,39,101,81])
    vision.factory_mode = True
    def moving(sample):raise ValueError("Base is moving")
    vision.stationary_context = moving
    with pytest.raises(ValueError,match="moving"):
        vision.make_tracker(cloth_sample(),[59,39,101,81])
    vision.stationary_context = lambda sample: {"epoch":("boot",.5,"path"),
        "servo_deg":[90]*6,"odom_xyyaw":[0,0,0]}
    assert vision.make_tracker(cloth_sample(),[59,39,101,81]).stationary_only


def test_feature_tracking_does_not_reuse_camera_geometry_after_calibration_change():
    import pytest
    first = VisionTests().sample()
    tracker = FeatureObject(first,[59,39,101,81])
    second = VisionTests().sample(1.1)
    second["k"] = second["k"].copy()
    second["k"][0,0] += 1
    with pytest.raises(ValueError,match="calibration"):
        tracker.update(second)
    assert tracker.ended


def test_static_observation_explicitly_blocks_motion_before_caller_can_publish():
    import pytest
    import time
    from delivery_vision import StationaryDepthObject
    vision = MeasuredVision.__new__(MeasuredVision)
    vision.lock = threading.RLock()
    vision.error = None
    vision.waiting_for_joints = None
    context = lambda sample:{"epoch":("boot",1,"path"),"servo_deg":[90]*6,"odom_xyyaw":[0,0,0]}
    vision.track = StationaryDepthObject(cloth_sample(time.time()),[59,39,101,81],context)
    assert vision.tracking_status()["stationary_only"] is True
    assert vision.tracking_status()["motion_tracking_available"] is False
    with pytest.raises(ValueError,match="stationary inspection only"):
        vision.tracking_status(require_motion=True)


def test_delivery_snapshot_checks_depth_exposure_and_registered_geometry(tmp_path):
    import time
    import pytest
    (tmp_path/"data").mkdir()
    vision = MeasuredVision.__new__(MeasuredVision)
    vision.root = tmp_path
    sample = cloth_sample(time.time())
    sample["depth_stamp"] = sample["stamp"]-.1
    path = tmp_path/"data/rgbd-snapshot.npz"
    np.savez(path,**sample)
    with pytest.raises(ValueError,match="synchronized"):
        vision.snapshot()
    sample["depth_stamp"] = sample["stamp"]
    sample["depth"] = sample["depth"][:60]
    np.savez(path,**sample)
    with pytest.raises(ValueError,match="registered"):
        vision.snapshot()
    sample = cloth_sample(time.time())
    sample["depth_stamp"] = sample["stamp"]
    np.savez(path,**sample)
    assert float(vision.snapshot()["depth_stamp"]) == sample["stamp"]
