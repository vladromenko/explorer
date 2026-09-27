import unittest
from grasp_verification import verify_lift

def frame(t,obj,tcp,clearance):
    return dict(at=t,object_id='sock-1',confidence=.95,depth_validated=True,
                object_xyz=[.3,0,obj],tcp_xyz=[.3,0,tcp],floor_clearance_m=clearance,base_xyyaw=[0,0,0])

class OutcomeTests(unittest.TestCase):
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
