import unittest,copy,threading,math
import numpy as np
from types import SimpleNamespace
from missions import Missions,readiness,navigation_attained,navigation_position_attained,position_only_arrival,FLAGS
class MissionReadinessTest(unittest.TestCase):
    def backup_fixture(self, status=4, error=0, after=None):
        from unittest.mock import Mock
        mission = Missions.__new__(Missions)
        mission.lock = threading.RLock()
        mission.active = {"id": "m"}
        mission.survey_permit = Mock()
        mission.emit = Mock()
        mission.hold_base = Mock()
        mission.maps = SimpleNamespace(pose=Mock(side_effect=[{"x": 0., "y": 0., "yaw": 0.},
            after or {"x": -.06, "y": 0., "yaw": 0.}]), preview=Mock())
        result = SimpleNamespace(status=status, result=SimpleNamespace(error_code=error))
        result_future = SimpleNamespace(done=lambda: True, result=lambda: result)
        handle = SimpleNamespace(accepted=True, cancel_goal_async=Mock(), get_result_async=lambda: result_future)
        future = SimpleNamespace(result=lambda: handle, add_done_callback=lambda callback: callback(None))
        mission.backup_client = SimpleNamespace(wait_for_server=lambda **kwargs: True,
            send_goal_async=Mock(return_value=future))
        return mission, handle

    def test_inspection_retreat_is_real_backup_not_a_nearby_nav_success(self):
        mission, handle = self.backup_fixture()
        result = mission.back_off("m", .06)
        goal = mission.backup_client.send_goal_async.call_args.args[0]
        self.assertAlmostEqual(goal.target.x, -.06)
        self.assertAlmostEqual(goal.speed, .04)
        mission.hold_base.assert_called_once_with("m")
        self.assertFalse(result["physical_accuracy_measured"])

    def test_backup_obstacle_and_false_arrival_never_pass(self):
        for options in ({"error": 3}, {"status": 6}, {"after": {"x": 0., "y": 0., "yaw": 0.}}):
            mission, handle = self.backup_fixture(**options)
            with self.assertRaises(ValueError):
                mission.back_off("m", .06)
            mission.hold_base.assert_called_once_with("m")

    def test_cancelled_owner_never_resumes_backup(self):
        mission, handle = self.backup_fixture()
        mission.active = None
        with self.assertRaisesRegex(ValueError, "cancelled"):
            mission.back_off("m", .06)
        handle.cancel_goal_async.assert_called_once()
        mission.emit.assert_not_called()

    def ready(self):return dict(at=100,stop_latched=False,mode='AUTONOMOUS',commissioning=dict.fromkeys(FLAGS,True),sensor_age=dict.fromkeys(['imu','odom','scan0','scan1','battery'],0),battery=12,reason='COMMAND EXPIRED')
    def test_each_missing_prerequisite_blocks(self):
        s=self.ready();self.assertEqual(readiness(s,100),[])
        for f in FLAGS:
            q=copy.deepcopy(s);q['commissioning'][f]=False;self.assertIn(f,readiness(q,100))
    def test_manual_stop_stale_and_low_battery(self):
        for key,value in [('stop_latched',True),('mode','MANUAL'),('at',98),('battery',10.5),('battery',float('nan'))]:
            s=self.ready();s[key]=value;self.assertTrue(readiness(s,100))
    def test_stale_lidar_blocks(self):
        s=self.ready();s['sensor_age']['scan1']=.61;self.assertIn('scan1_stale',readiness(s,100))

    def test_invalid_sensor_age_blocks(self):
        for age in (-1,float('nan'),float('inf'),None):
            s=self.ready();s['sensor_age']['scan0']=age;self.assertIn('scan0_stale',readiness(s,100))

    def test_base_stop_attempted_even_when_nav2_cancel_throws(self):
        mission=Missions.__new__(Missions);mission.lock=threading.RLock();calls=[]
        def failed_cancel():raise ValueError('cancel unavailable')
        mission.active=dict(id='m',handle=SimpleNamespace(cancel_goal_async=failed_cancel))
        mission.emit=lambda op,**kw:calls.append(op)
        mission.record=lambda m,state,details:calls.append(('record',details))
        with self.assertRaisesRegex(ValueError,'Nav2 cancel'):mission.finish('m','failed',dict(reason='test'))
        self.assertEqual(calls[0],'cancel_mission')
        self.assertTrue(calls[1][1]['stop_errors'])
        self.assertIsNone(mission.active)

    def test_navigation_result_requires_position_and_wrapped_heading(self):
        self.assertTrue(navigation_attained(dict(x=0,y=0,yaw=-math.pi),0,0,math.pi))
        self.assertFalse(navigation_attained(dict(x=0,y=0,yaw=.16),0,0,0))
        self.assertFalse(navigation_attained(dict(x=.16,y=0,yaw=0),0,0,0))
        self.assertFalse(navigation_attained(dict(x=math.nan,y=0,yaw=0),0,0,0))

    def test_explicit_holonomic_waypoint_checks_position_without_claiming_yaw(self):
        pose=dict(x=.95,y=.47,yaw=.70)
        self.assertTrue(navigation_position_attained(pose,.95,.39))
        self.assertFalse(navigation_attained(pose,.95,.39,.49))
        self.assertFalse(navigation_position_attained(pose,1.2,.39))
        self.assertTrue(navigation_position_attained(pose,np.float64(.95),np.float64(.39)))
        self.assertFalse(navigation_position_attained(pose,True,.39))

    def test_frontier_accepts_measured_position_but_rotation_and_return_need_heading(self):
        self.assertTrue(position_only_arrival("explore",False,False))
        self.assertFalse(position_only_arrival("explore",False,True))
        self.assertFalse(position_only_arrival("navigate",False,False))
        self.assertTrue(position_only_arrival("navigate",True,False))
        self.assertFalse(position_only_arrival("navigate",True,True))
