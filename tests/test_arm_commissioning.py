import copy
import unittest
from arm_commissioning import HOME, validate_pose, validate_incremental, stationary_status, coordinated_status


class ArmCommissioningTests(unittest.TestCase):
    def test_incremental_bounds_and_no_accidental_group_motion(self):
        validate_incremental(HOME,[90,115,3,0,90,30],4000)
        with self.assertRaises(ValueError):validate_incremental(HOME,[90,100,3,0,90,30],4000)
        with self.assertRaises(ValueError):validate_incremental(HOME,[90,115,13,0,90,30],4000)
        validate_incremental(HOME,[90,115,13,0,90,30],4000,coordinated=True)
        with self.assertRaises(ValueError):validate_incremental(HOME,[90,115,13,0,90,30],500,coordinated=True)
    def test_rejects_unsafe_gripper_and_unobserved_region(self):
        for pose in ([90,125,3,0,90,0], [150,125,3,0,90,30], [90]*5,
                     [90,125,3,0,float('nan'),30]):
            with self.assertRaises(ValueError):
                validate_pose(pose, 1500)
        validate_pose(HOME, 4000)


    def test_coordinated_status_allows_bounded_manual_motion(self):
        s=dict(at=100,mode='MANUAL',stop_latched=False,mission=None,
               velocity=[1.2,-1.08,2.5],odom_velocity=[1.3,-1.15,2.7],battery=12.0,
               power={'state':'NORMAL'},sensor_age={'odom':.1,'battery':.2})
        coordinated_status(s,100.1)
        for key,value in [('mode','AUTONOMOUS'),('stop_latched',True),('mission',{'id':'x'}),
                          ('velocity',[1.22,0,0]),('odom_velocity',[1.36,0,0]),('battery',10.9)]:
            changed=copy.deepcopy(s);changed[key]=value
            with self.assertRaises(ValueError):coordinated_status(changed,100.1)

    def test_stale_nonfinite_moving_and_unlatched_states_fail_closed(self):
        s = dict(at=100, stop_latched=True, velocity=[0,0,0], battery=11.2,
                 sensor_age=dict(odom=.1, battery=.2),odom_velocity=[0,0,0])
        stationary_status(s,100.1)
        for key, value in [('at',98),('at',float('nan')),('stop_latched',False),
                           ('velocity',[.02,0,0]),('velocity',[float('nan'),0,0]),
                           ('battery',10.8),('battery',None),('odom_velocity',None),('odom_velocity',[.02,0,0]),
                           ('sensor_age',dict(odom=.1,battery=3))]:
            changed = copy.deepcopy(s)
            changed[key] = value
            with self.assertRaises(ValueError):
                stationary_status(changed,100.1)
        s.update(stop_latched=False,base_hold_confirmed=True)
        stationary_status(s,100.1)
