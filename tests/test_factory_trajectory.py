import unittest
import math
from types import SimpleNamespace
import numpy as np
from factory_trajectory import compile_path

class FactoryTrajectoryTests(unittest.TestCase):
    def setUp(self):self.model=SimpleNamespace(path=lambda *a:dict(valid=True))
    def test_keeps_intermediate_waypoint_and_axis_sign(self):
        q=[[0.]*5,[math.radians(-5),0.,0.,0.,0.],[0.]*5]
        trajectory=dict(names=['arm'+str(i)+'_Joint' for i in range(1,6)],times=[0.,2.,4.],
                        positions=q,velocities=[[0.]*5]*3,accelerations=[[0.]*5]*3)
        path=compile_path([90]*6,[90]*6,trajectory,self.model)
        self.assertTrue(path['full_moveit_path_retained'])
        self.assertEqual(path['commands'][-1]['pose'],[90]*6)
        self.assertGreaterEqual(max(c['pose'][0] for c in path['commands']),94)
        self.assertTrue(all(c['runtime_ms']<=200 for c in path['commands']))
        self.assertEqual(path['profile'],'coordinated_quintic_lookahead')
    def test_no_final_pose_shortcut_and_collision_before_execution(self):
        self.model.path=lambda *a:dict(valid=False)
        with self.assertRaises(ValueError):compile_path([90]*6,[100]*6,None,self.model)
    def test_wrong_first_point_rejected(self):
        trajectory=dict(names=['arm'+str(i)+'_Joint' for i in range(1,6)],times=[0.,1.],
                        positions=[[.1]*5]*2,velocities=[[0.]*5]*2,accelerations=[[0.]*5]*2)
        with self.assertRaises(ValueError):compile_path([90]*6,[90]*6,trajectory,self.model)
    def test_duration_limits_and_no_measured_claim(self):
        path=compile_path([90]*6,[100,90,90,90,90,90],None,self.model)
        self.assertLess(path['duration'],1.)
        self.assertEqual(path['commands'][-1]['pose'],[100,90,90,90,90,90])
        self.assertTrue(all(c['end']>c['at'] for c in path['commands']))
        self.assertTrue(any(b['at']<a['end'] for a,b in zip(path['commands'],path['commands'][1:])))
        self.assertGreater(len(path['commands']),2)

    def test_totg_boundary_rest_blend_preserves_goal(self):
        trajectory=dict(names=['arm'+str(i)+'_Joint' for i in range(1,6)],times=[0.,2.],
                        positions=[[0.]*5,[0.,.1,0.,0.,0.]],
                        velocities=[[0.,.00005,0.,0.,0.],[0.]*5],
                        accelerations=[[0.,.05,0.,0.,0.],[0.,-.05,0.,0.,0.]])
        goal=[90,90+math.degrees(.1),90,90,90,90]
        path=compile_path([90]*6,goal,trajectory,self.model)
        self.assertEqual(path['commands'][-1]['pose'][1],96)
        trajectory['velocities'][0][1]=.1
        with self.assertRaises(ValueError):compile_path([90]*6,goal,trajectory,self.model)
