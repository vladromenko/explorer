import unittest
from grasp_verification import verify_lift

def frame(t,obj,tcp,clearance):
    return dict(at=t,object_id='sock-1',confidence=.95,depth_validated=True,identity_association_verified=True,
                frame='base_footprint',camera_pose_measured=True,
                object_xyz=[.3,0,obj],tcp_xyz=[.3,0,tcp],floor_clearance_m=clearance,base_xyyaw=[0,0,0])

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
    def test_stationary_supported_object_and_withdrawn_tool(self):
        from grasp_verification import verify_place
        before=[frame(t,.09,.11,.06) for t in (0,.5,1)]
        after=[dict(frame(t,.03,.16,0),gripper_open_measured=True) for t in (3,3.5,4)]
        zone=dict(center_xyz=[.3,0,.03],radius_m=.1,support_tolerance_m=.01)
        self.assertEqual(verify_place(before,after,zone,True)['outcome'],'success')
        after[-1]['camera_pose_measured']=False
        self.assertEqual(verify_place(before,after,zone,True)['outcome'],'unknown')
    def test_same_label_or_uncompensated_camera_motion_does_not_prove_lift(self):
        before=[frame(t,.03,.07,0) for t in (0,.5,1)]
        after=[frame(t,.09,.13,.06) for t in (3,3.5,4)]
        after[0]['identity_association_verified']=False
        self.assertEqual(verify_lift(before,after,True)['outcome'],'unknown')
