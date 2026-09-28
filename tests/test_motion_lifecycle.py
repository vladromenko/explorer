import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from std_msgs.msg import String
from core import Core
from missions import Missions
from motion_session import MotionSession
from safety import SafetyGate

class Harness:
    request=Core.request
    snapshot=Core.snapshot
    nav_cb=Core.nav_cb
    emergency=Core.emergency
    def __init__(self):
        self.config=dict.fromkeys(('base_commissioned','mcu_watchdog_verified','lidar_tf_validated','localization_verified'),True)
        self.config.update(max_velocity=[.15,.15,.4],max_acceleration=[.3,.3,.7])
        self.gate=SafetyGate(self.config);self.gate.estop=False;self.gate.mode='AUTONOMOUS'
        self.session=MotionSession();self.probe=None;self.joy_held=False
        self.nav_not_before=0;self.nav_last_stamp=0
        self.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(time.time()*1e9)))
        self.hold_requested=False;self.stationary_since=None;self.autonomy_lease=-1e9
        self.pub=Mock();self.request_ack=Mock();self.event=Mock();self.power_state={'motion_allowed':True}
    def send(self,op,**kw):
        self.request(String(data=json.dumps(dict(op=op,at=time.monotonic(),**kw))))
        return self.last_result['ok']

class MotionLifecycleTests(unittest.TestCase):
    def setUp(self):self.c=Harness()
    def test_arrival_hold_preserves_context_but_discards_nav_commands(self):
        from geometry_msgs.msg import TwistStamped
        c=self.c;self.assertTrue(c.send('begin_mission',mission='one'))
        c.send('resume_base',mission='one');msg=TwistStamped();msg.twist.linear.x=.1
        def stamp():
            now=time.time_ns();msg.header.stamp.sec=now//1_000_000_000;msg.header.stamp.nanosec=now%1_000_000_000
        stamp();c.nav_cb(msg)
        self.assertEqual(c.gate.command[0],.1)
        c.send('hold_base',mission='one');c.nav_cb(msg)
        self.assertEqual(c.gate.command,[0,0,0]);self.assertEqual(c.gate.mode,'AUTONOMOUS')
        self.assertFalse(c.gate.estop);self.assertEqual(c.session.mission,'one')
        c.send('resume_base',mission='one');c.nav_cb(msg);self.assertEqual(c.gate.command[0],0)
        stamp();c.nav_cb(msg);self.assertEqual(c.gate.command[0],.1)
    def test_finish_cancel_and_estop_have_different_semantics(self):
        for op in ('finish_mission','cancel_mission'):
            c=Harness();c.send('begin_mission',mission='one');c.send(op,mission='one')
            self.assertFalse(c.gate.estop);self.assertEqual(c.gate.mode,'AUTONOMOUS')
            self.assertFalse(c.send('autonomy_lease',mission='one'))
            self.assertFalse(c.send('begin_mission',mission='one'))
            self.assertTrue(c.send('begin_mission',mission='two'))
            self.assertFalse(c.send('cancel_mission',mission='one'))
            self.assertEqual(c.session.mission,'two')
        c=self.c;c.send('begin_mission',mission='one');c.send('estop')
        self.assertTrue(c.gate.estop);self.assertEqual(c.gate.mode,'MANUAL');self.assertIsNone(c.session.mission)
    def test_manual_disconnect_does_not_cancel_autonomy(self):
        c=self.c;c.send('begin_mission',mission='one');c.send('resume_base',mission='one')
        c.gate.submit([.1,0,0],'autonomy',time.monotonic());c.send('manual_release')
        self.assertEqual(c.session.mission,'one');self.assertEqual(c.gate.command[0],.1)
    def test_repeated_hold_does_not_restart_stationary_settling(self):
        c=self.c;c.send('begin_mission',mission='one');c.send('hold_base',mission='one')
        c.stationary_since=10
        c.send('hold_base',mission='one');self.assertEqual(c.stationary_since,10)
    def test_expired_lease_and_previous_session_cannot_resume(self):
        s=MotionSession();s.begin('one',1);s.resume('one',1.1)
        self.assertFalse(s.permits(1.36))
        with self.assertRaises(ValueError):s.renew('one',1.36)
        with self.assertRaises(ValueError):s.resume('one',1.37)
        s.begin('two',2)
        with self.assertRaises(ValueError):s.hold('one')
    def test_mission_finish_uses_normal_end_and_cancels_child(self):
        for state,op in [('succeeded','finish_mission'),('failed','cancel_mission'),('cancelled','cancel_mission')]:
            child=Mock();m=SimpleNamespace(lock=threading.RLock(),active=dict(id='one',handle=child),emit=Mock(),record=Mock())
            Missions.finish(m,'one',state,{})
            m.emit.assert_called_once_with(op,mission='one');child.cancel_goal_async.assert_called_once()
            self.assertIsNone(m.active)
            Missions.finish(m,'one',state,{})
            self.assertEqual(m.emit.call_count,1)
