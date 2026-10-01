import unittest
from navigation_health import LaunchHealth, fresh


class NavigationHealthTests(unittest.TestCase):
    def test_clock_and_receipt_must_both_be_current(self):
        self.assertTrue(fresh(.1,.1))
        for a,b in [(1.,.1),(.1,1.),(-.2,.1),(.1,-.1),(float('nan'),.1),(.1,float('inf'))]:
            self.assertFalse(fresh(a,b))

    def test_late_controller_waits_without_starting_or_timing_out(self):
        h=LaunchHealth(['planner_server'])
        self.assertEqual(h.evaluate(9000,['odom_stale'],{}),('wait',['odom_stale']))
        self.assertEqual(h.evaluate(9001,[],{}),('start',[]))

    def test_live_process_with_inactive_lifecycle_must_restart(self):
        h=LaunchHealth(['planner_server']); h.launched(0)
        self.assertEqual(h.evaluate(1,[],{'planner_server':(2,1)})[0],'starting')
        phase,why=h.evaluate(91,[],{'planner_server':(2,91)})
        self.assertEqual(phase,'restart'); self.assertIn('lifecycle_start_timeout:planner_server',why)

    def test_all_servers_need_fresh_active_state(self):
        h=LaunchHealth(['controller','behavior','bt']); h.launched(0)
        self.assertEqual(h.evaluate(1,[],{'controller':(3,1),'behavior':(3,1)})[0],'starting')
        states={k:(3,2) for k in h.nodes}
        self.assertEqual(h.evaluate(2,[],states)[0],'active')
        self.assertEqual(h.evaluate(6,[],states)[0],'starting')
        self.assertEqual(h.evaluate(15,[],states)[0],'restart')

    def test_controller_loss_ends_launch_and_does_not_resume_goal(self):
        h=LaunchHealth(['controller']); h.launched(0)
        self.assertEqual(h.evaluate(1,[],{'controller':(3,1)})[0],'active')
        self.assertEqual(h.evaluate(2,['scan1_stale'],{'controller':(3,2)})[0],'starting')
        self.assertEqual(h.evaluate(4,['scan1_stale'],{'controller':(3,4)}),('restart',['scan1_stale']))
        # A new supervisor waits for inputs and starts new servers; the policy
        # contains neither a stored goal nor an action resend operation.
        replacement=LaunchHealth(['controller'])
        self.assertEqual(replacement.evaluate(5,['scan1_stale'],{})[0],'wait')
        self.assertEqual(replacement.evaluate(6,[],{})[0],'start')

    def test_transient_gap_recovers_before_restart(self):
        h=LaunchHealth(['controller']); h.launched(0)
        h.evaluate(1,[],{'controller':(3,1)})
        h.evaluate(2,['odom_stale'],{'controller':(3,2)})
        self.assertEqual(h.evaluate(3,[],{'controller':(3,3)})[0],'active')
        self.assertIsNone(h.data_bad_since)


class OwnedProcessTests(unittest.TestCase):
    def test_stop_targets_only_its_own_launch_group(self):
        import subprocess
        import sys
        from navigation_service import stop_child
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],
                               start_new_session=True,stderr=subprocess.DEVNULL)
        other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],
                               start_new_session=True,stderr=subprocess.DEVNULL)
        try:
            stop_child(child)
            self.assertIsNotNone(child.poll())
            self.assertIsNone(other.poll())
        finally:
            if child.poll() is None: child.kill()
            if other.poll() is None: other.terminate()
            child.wait(timeout=3)
            other.wait(timeout=3)


class NavigationPowerTests(unittest.TestCase):
    def records(self, state='NORMAL', voltage=12., stamp=100.):
        status=dict(at=stamp,battery=voltage,sensor_age=dict(battery=.1),
                    power=dict(state=state,motion_allowed=state not in ('CRITICAL','UNKNOWN','CHARGING')))
        report=dict(at=stamp,state=state,battery_voltage_v=voltage)
        return status, report

    def errors(self, status=None, report=None, now=100.2):
        from navigation_health import power_errors
        return power_errors(status,report,now,10.8,3.)

    def test_either_recent_source_works(self):
        status, report=self.records()
        self.assertEqual(self.errors(status,{}),[])
        self.assertEqual(self.errors({},report),[])
        self.assertEqual(self.errors(status,report),[])

    def test_critical_and_unknown_block_even_if_sensors_are_fresh(self):
        for state in ('CRITICAL','UNKNOWN','CHARGING'):
            self.assertIn('power_'+state.lower(),self.errors(*self.records(state)))

    def test_actual_below_stop_blocks_before_policy_delay_finishes(self):
        self.assertIn('battery_below_stop',self.errors(*self.records(voltage=10.64)))
        self.assertIn('battery_below_stop',self.errors(*self.records(voltage=10.8)))

    def test_voltage_rebound_does_not_clear_existing_critical_latch(self):
        self.assertIn('power_critical',self.errors(*self.records('CRITICAL',12.4)))
        status,report=self.records('NORMAL',12.4)
        report['state']='CRITICAL'
        self.assertIn('power_critical',self.errors(status,report))
        self.assertEqual(self.errors(*self.records('LOW_POWER',11.25)),[])

    def test_missing_stale_future_nan_and_old_measurement_block(self):
        self.assertEqual(self.errors({},{}),['power_stale'])
        for stamp in (97.,101.,float('nan')):
            self.assertEqual(self.errors(*self.records(stamp=stamp)),['power_stale'])
        status,_=self.records(); status['sensor_age']['battery']=2.9
        self.assertIn('battery_stale',self.errors(status,{}))
        self.assertIn('battery_unknown',self.errors(*self.records(voltage=float('nan'))))

    def test_healthy_stale_report_does_not_override_fresh_critical_status(self):
        status,_=self.records('CRITICAL',10.64)
        _,report=self.records('NORMAL',12.4,97.)
        self.assertIn('power_critical',self.errors(status,report))

    def test_active_launch_stops_after_sustained_power_loss(self):
        h=LaunchHealth(['planner_server']); h.launched(0)
        self.assertEqual(h.evaluate(1,[],{'planner_server':(3,1)})[0],'active')
        blockers=self.errors(*self.records('CRITICAL',10.64))
        self.assertEqual(h.evaluate(2,blockers,{'planner_server':(3,2)})[0],'starting')
        self.assertEqual(h.evaluate(4,blockers,{'planner_server':(3,4)})[0],'restart')
        replacement=LaunchHealth(['planner_server'])
        self.assertEqual(replacement.evaluate(5,blockers,{})[0],'wait')
        self.assertEqual(replacement.evaluate(6,self.errors(*self.records('NORMAL',12.4)),{})[0],'start')
