import fcntl
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
from manual_arm import ManualArm

class ManualArmTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        (self.root/'data').mkdir();self.node=Mock();self.pub=self.node.create_publisher.return_value
        self.pub.get_subscription_count.return_value=1;self.pub.wait_for_all_acked.return_value=True
        self.model=Mock();self.model.path.return_value={'valid':True}
        self.arm=ManualArm(self.root,self.node,lambda:self.model);self.arm.ready=True
        self.start=[90,125,3,0,90,30];self.goal=[92,125,3,0,90,30]
        self.state=dict(at=time.time(),servo_deg=self.start,boot_id=self.arm.boot,phase='command_elapsed_observation_required')
        self.save()
        (self.root/'data/status.json').write_text('{}')
        self.patch=patch('manual_arm.stationary_status');self.stationary=self.patch.start()

    def tearDown(self):self.patch.stop();self.directory.cleanup()
    def save(self):(self.root/'data/arm-state.json').write_text(json.dumps(self.state))

    def test_rejects_large_step_before_publication(self):
        with self.assertRaises(ValueError):self.arm.move(self.start,[110,125,3,0,90,30])
        self.pub.publish.assert_not_called()

    def test_expired_gamepad_decision_never_moves(self):
        self.arm.gamepad_permit=lambda:True
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal,time.monotonic()-1)
        self.pub.publish.assert_not_called()

    def test_release_during_collision_check_never_moves(self):
        self.arm.gamepad_permit=lambda:False
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal,time.monotonic()+1)
        self.pub.publish.assert_not_called()

    def test_stop_during_collision_check_never_moves(self):
        def cancel(*args):self.arm.stop();return {'valid':True}
        self.model.path.side_effect=cancel
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal)
        self.pub.publish.assert_not_called()

    def test_stop_during_external_planning_never_moves(self):
        revision=self.arm.stop_revision
        self.arm.stop()
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal,expected_stop_revision=revision)
        self.pub.publish.assert_not_called()

    def test_policy_hold_expiring_during_geometry_check_never_moves(self):
        def expired():raise ValueError('hold expired')
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal,execution_permit=expired,source='supervised_policy')
        self.pub.publish.assert_not_called()

    def test_prior_boot_never_moves(self):
        self.state['boot_id']='old';self.save()
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal)
        self.pub.publish.assert_not_called()

    def test_telemetry_fault_invalidates_reference(self):
        (self.root/'data/arm-telemetry-fault.json').write_text(json.dumps({'at':time.time()+1}))
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal)
        self.pub.publish.assert_not_called()

    def test_delivery_failure_prevents_next_step(self):
        self.pub.wait_for_all_acked.return_value=False
        with self.assertRaises(ValueError):self.arm.move(self.start,self.goal)
        first_attempt=self.pub.publish.call_count
        self.assertGreaterEqual(first_attempt,2)
        state=json.loads((self.root/'data/arm-state.json').read_text())
        self.assertEqual(state['phase'],'monitor_failed_state_unknown')
        with self.assertRaises(ValueError):self.arm.move(self.goal,self.start)
        self.assertEqual(self.pub.publish.call_count,first_attempt)

    def test_smooth_finite_position_stream_and_honest_state(self):
        result=self.arm.move(self.start,self.goal)
        self.assertGreaterEqual(self.pub.publish.call_count,2)
        self.assertLessEqual(max(c.args[0].time for c in self.pub.publish.call_args_list),200)
        self.assertEqual(self.pub.publish.call_args.args[0].joint1,self.goal[0])
        self.assertEqual(result['motion_profile'],'coordinated_quintic_lookahead')
        self.assertFalse(result['measured']);self.assertFalse(result['attained'])

    def test_fast_manual_profile_has_fourfold_dynamics(self):
        path=dict(commands=[dict(at=0.,pose=self.goal,runtime_ms=20)],duration=.02,
                  source_sha256='test',profile='coordinated_quintic_lookahead')
        with patch('factory_trajectory.compile_path',return_value=path) as compile_path:
            self.arm.move(self.start,self.goal,speed='fast')
        motion=compile_path.call_args.args[4]
        self.assertEqual(motion['velocity_deg_s'][0],448.)
        self.assertEqual(motion['acceleration_deg_s2'][0],6400.)
        self.assertEqual(motion['jerk_deg_s3'][0],256000.)
        self.assertEqual(motion['min_duration_s'],.0875)
        self.assertEqual(motion['publish_period_s'],.04)
        self.assertAlmostEqual(motion['lookahead_s'],.08)

    def test_observed_reference_never_publishes_or_claims_measurement(self):
        result=self.arm.accept_reference([90]*6,True)
        self.pub.publish.assert_not_called()
        self.assertFalse(result['motion_sent']);self.assertFalse(result['measured'])
        self.assertEqual(self.arm.reference()['servo_deg'],[90]*6)

    def test_factory_home_commands_known_pose_without_claiming_measurement(self):
        result=self.arm.home_reference(True)
        message=self.pub.publish.call_args.args[0]
        self.assertEqual([getattr(message,'joint'+str(i)) for i in range(1,7)],[90]*6)
        self.assertEqual(message.time,4000)
        self.assertTrue(result['motion_sent'])
        self.assertFalse(result['attainment_measured'])
        self.assertEqual(self.arm.reference()['servo_deg'],[90]*6)

    def test_factory_home_requires_observer_when_manual(self):
        with self.assertRaises(ValueError):self.arm.home_reference(False)
        self.pub.publish.assert_not_called()

    def test_startup_home_waits_for_transient_power_readiness(self):
        self.arm.startup_config=dict(delay_after_web_start_s=1,startup_window_s=180,
                                     requires_latched_stop=True)
        (self.root/'data/status.json').write_text(json.dumps({'stop_latched':True}))
        self.stationary.side_effect=[ValueError('Power policy UNKNOWN'),None]
        self.arm._uptime=Mock(side_effect=[10.,11.])
        with patch('manual_arm.time.sleep'),patch.object(self.arm,'home_reference') as home:
            self.arm._automatic_startup_home()
        home.assert_called_once_with(False,'automatic_factory_startup')
        self.assertIn('ожидает готовности',self.arm.error)

    def test_timed_cancel_sends_no_following_target(self):
        commands=[dict(at=0.,end=.05,pose=self.goal,runtime_ms=50),
                  dict(at=.05,end=.10,pose=self.start,runtime_ms=50)]
        path=dict(commands=commands,duration=.10,source_sha256='test',full_moveit_path_retained=True)
        def permit():
            if self.pub.publish.call_count:raise ValueError('STOP')
        with patch('factory_trajectory.compile_path',return_value=path):
            with self.assertRaisesRegex(ValueError,'STOP'):
                self.arm.execute_path(self.start,self.start,{},permit,self.arm.stop_revision)
        self.pub.publish.assert_called_once()
        state=json.loads((self.root/'data/arm-state.json').read_text())
        self.assertFalse(state['attained']);self.assertFalse(state['command_completed'])
        self.assertTrue(state['cancelled'])

    def test_timed_path_rejects_other_process_owner(self):
        with (self.root/'data/arm-commissioning.lock').open('a') as owner:
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.arm.execute_path(self.start,self.goal,None,lambda:None,self.arm.stop_revision)
        self.pub.publish.assert_not_called()
        self.assertFalse(self.arm.lock.locked())

if __name__=='__main__':unittest.main()
