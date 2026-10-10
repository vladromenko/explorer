import unittest
import math
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from factory_trajectory import compile_path,_straight_retime
from timed_trajectory import Limits,TimedPath

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
        self.assertEqual(path["source_waypoint_count"],3)
        self.assertEqual(path["source_duration_s"],4.0)
        self.assertAlmostEqual(path["duration"],4.0*path["time_scale"])
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

    def test_jazzy_first_integration_step_artifact(self):
        trajectory=dict(names=["arm"+str(i)+"_Joint" for i in range(1,6)],times=[0.,2.],
            positions=[[0.]*5,[0.,0.,-.1,-.25,0.]],
            velocities=[[0.,0.,-.000977384381,-.002443460953,0.],[0.]*5],
            accelerations=[[0.,0.,-.977384381,-2.443460953,0.],[0.]*5])
        goal=[90,90,90-math.degrees(.1),90-math.degrees(.25),90,90]
        path=compile_path([90]*6,goal,trajectory,self.model)
        self.assertEqual(path["commands"][-1]["pose"],np.rint(goal).astype(int).tolist())
        trajectory["velocities"][0][3]=-.0035
        with self.assertRaisesRegex(ValueError,"boundary velocity"):
            compile_path([90]*6,goal,trajectory,self.model)

    def test_real_totg_last_sample_keeps_line_and_every_waypoint(self):
        record=json.loads((Path(__file__).parent/"fixtures/totg-forward-20261010.json").read_text())
        source=record["joint_trajectory"]
        path=compile_path(record["start"],record["goal"],source,self.model)
        self.assertEqual(path["retiming_algorithm"],"collinear_synchronized_jerk_limited")
        self.assertEqual(path["source_waypoint_count"],26)
        self.assertEqual(len(path["retained_source_waypoint_times_s"]),26)
        self.assertLess(path["duration"],2.6)
        self.assertGreater(path["duration"],76/35)
        self.assertEqual(path["commands"][-1]["pose"],record["goal"])
        self.assertEqual(len(path["original_moveit_sha256"]),64)
        delta=np.asarray(record["goal"])-record["start"]
        for item in path["commands"]:
            progress=(item["pose"][3]-record["start"][3])/delta[3]
            ideal=np.asarray(record["start"])+progress*delta
            self.assertLessEqual(np.max(abs(np.asarray(item["pose"])-ideal)),1.0)

    def test_straight_retiming_preserves_crossed_waypoints_and_dynamics(self):
        limits=Limits(np.array([-2.0]*2),np.array([2.0]*2),np.array([0.5,0.8]),
            np.array([1.0,1.4]),np.array([10.0,14.0]))
        q=np.array([[0.0,0.0],[0.02,0.04],[0.2,0.4],[0.3,0.6]])
        v=np.array([[0.0,0.0],[0.1,0.2],[0.1,0.2],[0.0,0.0]])
        retimed=_straight_retime([0.0,0.4,2.2,3.0],q,v,np.zeros_like(q),limits)
        self.assertIsNotNone(retimed)
        curve=TimedPath(["j1","j2"],retimed["times"],retimed["positions"],
            retimed["velocities"],retimed["accelerations"],limits,lambda *args:True)
        for original,at in zip(q,retimed["source_waypoint_times"]):
            self.assertTrue(np.allclose(curve.sample(at*curve.scale)["position"],original,atol=1e-9))
        for at in np.linspace(0.0,curve.duration,501):
            sample=curve.sample(at)
            for key,maximum in (("velocity",limits.velocity),("acceleration",limits.acceleration),("jerk",limits.jerk)):
                self.assertTrue(np.all(np.abs(sample[key])<=maximum+1e-6))
            self.assertAlmostEqual(sample["position"][1],sample["position"][0]*2,places=8)
        for at in curve.times[1:-1]:
            left,right=curve.sample(at-1e-7),curve.sample(at+1e-7)
            self.assertTrue(np.allclose(left["velocity"],right["velocity"],atol=1e-5))
            self.assertTrue(np.allclose(left["acceleration"],right["acceleration"],atol=1e-5))

    def test_curved_and_reversing_paths_are_not_scalar_shortcut(self):
        limits=Limits(np.array([-2.0]*2),np.array([2.0]*2),np.ones(2),np.ones(2),np.ones(2)*10)
        for q in (np.array([[0.,0.],[.1,.2],[.3,.3]]),np.array([[0.,0.],[.4,.4],[.3,.3]])):
            self.assertIsNone(_straight_retime([0.,1.,2.],q,np.zeros_like(q),np.zeros_like(q),limits))
