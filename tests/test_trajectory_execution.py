import tempfile,threading,time,unittest,json
from pathlib import Path
from unittest.mock import Mock,patch
from trajectory_execution import command_steps,gripper_steps,TrajectoryExecution,motion_budget

HOME=[90,125,3,0,90,30]
class TrajectoryTests(unittest.TestCase):
    def test_full_gripper_closure_is_finite_not_cut_off_at_30_steps(self):
        steps=gripper_steps(HOME,160)
        self.assertEqual(len(steps),65);self.assertEqual(steps[-1],HOME[:5]+[160])
        self.assertTrue(all(p[:5]==HOME[:5] for p in steps))
        with self.assertRaises(ValueError):gripper_steps(HOME,171)
    def test_local_execution_uses_mission_permission_not_browser_heartbeat(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.runner(root);p.execution_mode='local_mission';p.deadline=time.monotonic()+1
            p.revision=0;p.lease=-1;p.local_permit=Mock();p.permit();p.local_permit.assert_called_once()
            p.local_permit.side_effect=ValueError('mission cancelled')
            with self.assertRaises(ValueError):p.permit()
    def test_finite_operator_execution_still_checks_deadline_and_stop(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.runner(root);p.execution_mode='operator_finite';p.deadline=time.monotonic()+1;p.revision=0
            p.lease=-1;p.permit();p.manual.stop_revision=1
            with self.assertRaises(ValueError):p.permit()
            p.manual.stop_revision=0;p.deadline=time.monotonic()-1
            with self.assertRaises(ValueError):p.permit()
    def test_quantization_keeps_edges_short_and_gripper_unchanged(self):
        points=[HOME[:5],[91.4,124.3,3,1.5,90],[100,115,4,10,90]]
        steps=command_steps(points,HOME);previous=HOME
        for goal in steps:
            self.assertLessEqual(max(abs(a-b) for a,b in zip(goal,previous)),2)
            self.assertEqual(goal[5],30);previous=goal
        self.assertEqual(steps[-1],[100,115,4,10,90,30])
    def test_invalid_path_never_becomes_commands(self):
        for points in ([[0]*5,[1]*5],[[90,125,3,0,90],[90,125,3,-1,90]],[[90,125,3,0,90],[90,125,3,float('nan'),90]]):
            with self.assertRaises(ValueError):command_steps(points,HOME)
    def runner(self,root):
        teaching=Mock();teaching.lock=threading.Lock();teaching.active=None;teaching.observation.return_value=(HOME,None)
        manual=Mock();manual.stop_revision=0;manual.model.return_value.path.return_value={'valid':True}
        planner=Mock();planner.return_value.plan.return_value={'planned':True,'servo_waypoints':[HOME[:5],[92,125,3,0,90]]}
        return TrajectoryExecution(root,teaching,manual,planner)
    def test_stale_or_changed_start_rejected_before_movement(self):
        with tempfile.TemporaryDirectory() as root,patch('trajectory_execution.motion_budget'):
            p=self.runner(root);r=p.plan([92,125,3,0,90]);p.pending['at']-=61
            with self.assertRaises(ValueError):p.start(r['plan_id'],True)
            p.manual.move.assert_not_called();self.assertFalse(p.lock.locked());self.assertFalse(p.teaching.lock.locked())
    def test_rounded_collision_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.runner(root);p.manual.model.return_value.path.return_value={'valid':False}
            with self.assertRaises(ValueError):p.plan([92,125,3,0,90])
            self.assertIsNone(p.pending);p.manual.move.assert_not_called()
    def test_lost_lease_never_publishes_next_step(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'data').mkdir();p=self.runner(root);p.lock.acquire();p.teaching.lock.acquire()
            p.session='a'*32;p.revision=0;p.lease=time.monotonic()-1
            p.run({'start':HOME,'steps':[[92,125,3,0,90,30]]})
            p.manual.move.assert_not_called();self.assertEqual(p.state['phase'],'stopped')
            self.assertFalse(p.lock.locked());self.assertFalse(p.teaching.lock.locked())

    def test_motion_policy_is_separate_from_gpu_training_and_rejects_faults(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'data/status.json';path.parent.mkdir()
            status=dict(at=time.time(),stop_latched=True,velocity=[0,0,0],battery=11.45,
                        power={'state':'NORMAL'},sensor_age={'odom':.02,'battery':.1},odom_velocity=[0,0,0])
            path.write_text(json.dumps(status));motion_budget(root)
            for change in (dict(power={'state':'LOW_POWER'}),dict(power={'state':'CRITICAL'}),
                           dict(power={'state':'CHARGING'}),dict(stop_latched=False),
                           dict(battery=10.9),dict(sensor_age={'odom':1,'battery':.1}),
                           dict(at=time.time()-2)):
                path.write_text(json.dumps(dict(status,**change)))
                with self.assertRaises(ValueError):motion_budget(root)
