import unittest
from grasp_verification import verify_lift

def frame(t,obj,tcp,clearance):
    return dict(at=t,object_id='sock-1',confidence=.95,depth_validated=True,identity_association_verified=True,
                frame='base_footprint',camera_pose_measured=True,
                object_xyz=[.3,0,obj],tcp_xyz=[.3,0,tcp],floor_clearance_m=clearance,base_xyyaw=[0,0,0],
                object_extent_xyz_m=[.04,.03,.06],object_position_uncertainty_m=.002)

class OutcomeTests(unittest.TestCase):
    def test_physically_validated_command_pose_is_accepted_without_claiming_measurement(self):
        before=[frame(t,.03,.07,0) for t in (0,.5,1)]
        after=[frame(t,.09,.13,.06) for t in (3,3.5,4)]
        for item in before+after:
            item.update(camera_pose_measured=False,camera_pose_source='command_estimate',
                        camera_pose_validated_for_execution=True,camera_pose_validation_record='physical.json')
        self.assertEqual(verify_lift(before,after,True)['outcome'],'success')

    def test_lift_and_failure_require_measured_object_motion(self):
        before=[frame(t,.03,.07,0) for t in (0,.5,1)]
        held=[frame(t,.09,.13,.06) for t in (3,3.5,4)]
        missed=[frame(t,.03,.13,0) for t in (3,3.5,4)]
        self.assertEqual(verify_lift(before,held,True)['outcome'],'success')
        self.assertEqual(verify_lift(before,missed,True)['outcome'],'failure')
        self.assertEqual(verify_lift(before,held,False)['outcome'],'unknown')
        held[-1]['object_id']='other'
        self.assertEqual(verify_lift(before,held,True)['outcome'],'unknown')

    def test_motion_occlusion_and_short_hold_do_not_create_labels(self):
        before=[frame(t,.03,.07,0) for t in (0,.5,1)]
        after=[frame(t,.09,.13,.06) for t in (3,3.5,4)]
        after[-1]['base_xyyaw']=[.02,0,0]
        self.assertEqual(verify_lift(before,after,True)['outcome'],'unknown')
        self.assertEqual(verify_lift(before,[],True)['outcome'],'unknown')


class PlaceTests(unittest.TestCase):
    def placement(self,command_mode=False):
        from grasp_verification import verify_place
        before=[frame(t,.03,.05,.03) for t in (0,.5,1)]
        after=[frame(t,.03,.15,.03) for t in (3,3.5,4)]
        for item in after:
            item['gripper_open_measured']=True if not command_mode else None
            item['gripper_open_commanded']=True if command_mode else None
        zone=dict(center_xyz=[.3,0,0],radius_m=.1,support_tolerance_m=.01)
        return before,after,zone,verify_place

    def test_stationary_supported_object_and_withdrawn_tool(self):
        before,after,zone,verify_place=self.placement()
        result=verify_place(before,after,zone,True)
        self.assertEqual(result['outcome'],'success')
        self.assertEqual(result['evidence']['gripper_open_provenance'],'measured')
        after[-1]['camera_pose_measured']=False
        self.assertEqual(verify_place(before,after,zone,True)['outcome'],'unknown')

    def test_command_mode_can_succeed_without_fabricated_open_measurement(self):
        before,after,zone,verify_place=self.placement(command_mode=True)
        for item in before+after:
            item.update(camera_pose_measured=False,camera_pose_source='command_estimate',
                        camera_pose_validated_for_execution=True,camera_pose_validation_record='handeye.json')
        result=verify_place(before,after,zone,True)
        self.assertEqual(result['outcome'],'success')
        self.assertEqual(result['evidence']['gripper_open_provenance'],'command_estimate')

    def test_still_held_or_open_command_without_separation_is_not_success(self):
        before,after,zone,verify_place=self.placement(command_mode=True)
        for item in after:
            item['object_xyz']=[.4,0,.13];item['tcp_xyz']=[.4,0,.15]
        self.assertEqual(verify_place(before,after,zone,True)['outcome'],'failure')

    def test_outside_fallen_and_closed_occluded_are_negative_or_unknown(self):
        before,after,zone,verify_place=self.placement()
        outside=[dict(item,object_xyz=[.5,0,.03]) for item in after]
        self.assertEqual(verify_place(before,outside,zone,True)['outcome'],'failure')
        fallen=[dict(item,object_xyz=[.3,0,-.03]) for item in after]
        self.assertEqual(verify_place(before,fallen,zone,True)['outcome'],'failure')
        occluded=[dict(item,identity_association_verified=False,gripper_open_measured=False) for item in after]
        self.assertEqual(verify_place(before,occluded,zone,True)['outcome'],'unknown')

    def test_track_switch_camera_or_unstable_object_never_passes(self):
        before,after,zone,verify_place=self.placement()
        switched=[dict(item) for item in after];switched[-1]['object_id']='other'
        self.assertEqual(verify_place(before,switched,zone,True)['outcome'],'unknown')
        moving=[dict(item) for item in after];moving[-1]['object_xyz']=[.34,0,.03]
        self.assertEqual(verify_place(before,moving,zone,True)['outcome'],'unknown')
        moved_camera=[dict(item) for item in after];moved_camera[-1]['base_xyyaw']=[.02,0,0]
        self.assertEqual(verify_place(before,moved_camera,zone,True)['outcome'],'unknown')
    def test_same_label_or_uncompensated_camera_motion_does_not_prove_lift(self):
        before=[frame(t,.03,.07,0) for t in (0,.5,1)]
        after=[frame(t,.09,.13,.06) for t in (3,3.5,4)]
        after[0]['identity_association_verified']=False
        self.assertEqual(verify_lift(before,after,True)['outcome'],'unknown')
