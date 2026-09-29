import math
from types import SimpleNamespace
import unittest
import numpy as np

from delivery_robot import DeliveryRobot, placement_zone, validated_places


class DeliveryRobotTests(unittest.TestCase):
    def adapter(self, phase='reached', measured=None):
        initial=[90.]*6;goal=[91.]*6
        references=iter([dict(servo_deg=initial),dict(servo_deg=goal if measured is None else measured)])
        arm=SimpleNamespace(status=lambda:dict(busy=False),reference=lambda:next(references))
        trajectory=SimpleNamespace(status=lambda:dict(busy=False,phase=phase,reached=True,session='path'),
            plan=lambda goal:dict(plan_id='plan'),start_local=lambda *args:dict(session='path'))
        robot=DeliveryRobot('.',None,arm,trajectory,None,None,None)
        robot.mid='delivery';robot.permit=lambda mid:None;robot.hold=lambda:None
        return robot,goal

    def test_stale_reached_flag_from_previous_path_is_not_attainment(self):
        robot,goal=self.adapter(phase='stopped')
        with self.assertRaisesRegex(ValueError,'Поза не достигнута'):robot._move(goal)

    def test_path_result_requires_real_current_position(self):
        robot,goal=self.adapter(measured=[90.]*6)
        with self.assertRaisesRegex(ValueError,'Поза не достигнута'):robot._move(goal)

    def test_matching_measured_path_can_finish(self):
        robot,goal=self.adapter()
        self.assertTrue(robot._move(goal)['attained'])

    def test_cleanup_attempts_all_stops_when_arm_or_camera_stop_fails(self):
        calls=[]
        def fail(name):
            calls.append(name)
            raise ValueError(name+' failed')
        missions=SimpleNamespace(finish=lambda *args:calls.append('base'))
        finder=SimpleNamespace(cancel=lambda sid:calls.append('finder:'+sid))
        robot=DeliveryRobot('.',missions,None,SimpleNamespace(stop=lambda:fail('arm')),finder,
                            SimpleNamespace(stop=lambda:fail('vision')),None)
        robot.mid='delivery';robot.search_id='search';robot.tracking=True
        with self.assertRaisesRegex(ValueError,'Ошибки завершения'):robot.stop('delivery')
        self.assertEqual(calls,['arm','base','vision','finder:search'])
        self.assertIsNone(robot.mid);self.assertFalse(robot.tracking)
        robot.stop('delivery')
        self.assertEqual(len(calls),4)

    def test_placement_point_remains_fixed_in_map_despite_nav_stop_offset(self):
        zone=dict(center_xyz=[.3,.1,.04],radius_m=.1,support_tolerance_m=.01)
        saved=dict(x=2.,y=3.,yaw=math.pi/2)
        actual=dict(x=1.95,y=3.04,yaw=math.pi/2+.12)
        transformed=placement_zone(zone,saved,actual)
        def world(pose,point):
            x,y,z=point;c,s=math.cos(pose['yaw']),math.sin(pose['yaw'])
            return [pose['x']+c*x-s*y,pose['y']+s*x+c*y,z]
        np.testing.assert_allclose(world(saved,zone['center_xyz']),world(actual,transformed['center_xyz']),atol=1e-12)
        self.assertEqual(transformed['radius_m'],zone['radius_m'])
        self.assertEqual(zone['center_xyz'],[.3,.1,.04])

    def test_changed_place_or_map_does_not_reuse_accepted_geometry(self):
        settings=dict(map_epoch='m',place_poses={'basket':dict(x=1,y=2,yaw=math.pi)})
        place=dict(name='basket',compatible_map=True,x=1,y=2,yaw=-math.pi)
        self.assertIn('basket',validated_places(settings,[place],'m'))
        with self.assertRaisesRegex(ValueError,'Карта отличается'):validated_places(settings,[place],'different')
        with self.assertRaisesRegex(ValueError,'Место|место'):
            validated_places(settings,[dict(place,x=1.01)],'m')


if __name__=='__main__':unittest.main()
