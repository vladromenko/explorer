import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from delivery_task import DeliveryTask

RealThread = threading.Thread


class Robot:
    def __init__(self):
        self.calls=[]; self.owner=None; self.find_hook=None; self.stop_hook=None
        self.hold_outcome='success'; self.place_outcome='success'
    def blockers(self):return []
    def begin(self, mid, check):
        check(); self.owner=mid; self.calls.append(('begin',mid))
    def permit(self, mid):
        if self.owner!=mid:raise ValueError('owner changed')
    def stop(self, mid):
        self.calls.append(('stop',mid))
        if self.stop_hook:self.stop_hook()
        if self.owner==mid:self.owner=None
    def finish(self, mid, success):
        self.calls.append(('finish',mid,success))
        if self.owner==mid:self.owner=None
    def find(self):
        self.calls.append(('find',))
        return self.find_hook() if self.find_hook else {'target':'measured object'}
    def verify_hold(self, before):
        self.calls.append(('verify_hold',))
        return {'outcome':self.hold_outcome}
    def verify_place(self, before):
        self.calls.append(('verify_place',))
        return {'outcome':self.place_outcome}
    def __getattr__(self, name):
        if name not in ('approach','hold','reobserve','pregrasp','observe','grasp',
                        'lift','transport','carry','support','release','withdraw'):
            raise AttributeError(name)
        def operation(*args):
            self.calls.append((name,))
            return {'measured':True}
        return operation


class DeliveryTaskTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name); self.robot=Robot(); self.task=DeliveryTask(self.root,self.robot)

    def queued(self):
        with patch('delivery_task.threading.Thread') as thread:
            result=self.task.start()
            args=thread.call_args.kwargs
        return result,lambda:args['target'](*args['args'])

    def test_unknown_grasp_prevents_transport_and_carry(self):
        self.robot.hold_outcome='unknown'
        result,run=self.queued(); run()
        names=[call[0] for call in self.robot.calls]
        self.assertIn('verify_hold',names)
        self.assertNotIn('transport',names); self.assertNotIn('carry',names)
        self.assertEqual(self.task.last['state'],'failed')
        self.assertFalse(self.task.last['delivered'])
        self.assertIn('unknown',self.task.last['reason'])
        self.assertFalse(self.task.status()['busy'])

    def test_cancel_invalidates_late_result_and_reserves_worker_until_cleanup(self):
        def delayed_result():
            self.task.cancel()
            self.assertTrue(self.task.status()['cancelling'])
            with self.assertRaises(ValueError):self.task.start()
            return {'late_target':'must not be used'}
        self.robot.find_hook=delayed_result
        result,run=self.queued(); run()
        self.assertNotIn('approach',[call[0] for call in self.robot.calls])
        self.assertEqual(self.task.last['state'],'cancelled')
        self.assertEqual(self.task.last['events'],[])
        self.assertFalse(self.task.status()['busy'])
        self.robot.find_hook=None
        newer,new_run=self.queued()
        self.assertNotEqual(newer['id'],result['id'])
        new_run()
        self.assertTrue(self.task.last['delivered'])
        self.assertEqual(self.task.last['id'],newer['id'])

    def test_cancel_before_worker_begin_never_opens_robot_session(self):
        _,run=self.queued(); self.task.cancel(); run()
        self.assertNotIn('begin',[call[0] for call in self.robot.calls])
        self.assertEqual(self.task.last['state'],'cancelled')
        self.assertFalse(self.task.status()['busy'])

    def test_cancel_stop_reservation_outlives_worker_cleanup(self):
        entered=threading.Event(); release=threading.Event()
        self.robot.stop_hook=lambda:(entered.set(),release.wait(2.))
        _,run=self.queued()
        cancellation=RealThread(target=self.task.cancel)
        cancellation.start()
        self.assertTrue(entered.wait(1.))
        try:
            run()  # Cancelled before begin: worker exits while stop is still in flight.
            self.assertTrue(self.task.status()['busy'])
            with self.assertRaises(ValueError):self.task.start()
        finally:
            release.set(); cancellation.join(2.)
        self.assertFalse(cancellation.is_alive())
        self.assertFalse(self.task.status()['busy'])

    def test_process_restart_marks_running_record_interrupted_without_resuming(self):
        record=dict(id='prior',state='running',started=time.time(),phase='carry',events=[],delivered=False)
        path=self.root/'data/delivery-runs/prior.json'; path.write_text(json.dumps(record))
        robot=Robot(); restarted=DeliveryTask(self.root,robot)
        saved=json.loads(path.read_text())
        self.assertEqual(saved['state'],'interrupted')
        self.assertFalse(saved['delivered'])
        self.assertIsNone(restarted.active)
        self.assertFalse(restarted.status()['busy'])
        self.assertEqual(robot.calls,[])

    def test_failed_thread_creation_releases_reservation_without_robot_actions(self):
        with patch('delivery_task.threading.Thread') as thread:
            thread.return_value.start.side_effect=RuntimeError('no thread')
            with self.assertRaises(RuntimeError):self.task.start()
        self.assertEqual(self.task.last['state'],'failed')
        self.assertFalse(self.task.status()['busy'])
        self.assertEqual(self.robot.calls,[])

    def test_adapter_readiness_runs_outside_task_lock(self):
        def blockers():
            observed=[]
            def probe_lock():
                acquired=self.task.lock.acquire(blocking=False)
                observed.append(acquired)
                if acquired:self.task.lock.release()
            probe=RealThread(target=probe_lock); probe.start(); probe.join(1.)
            self.assertEqual(observed,[True])
            return []
        self.robot.blockers=blockers
        _,run=self.queued()
        self.task.status()
        run()

    def test_unknown_placement_never_reports_delivery(self):
        self.robot.place_outcome='unknown'
        _,run=self.queued(); run()
        self.assertIn('carry',[call[0] for call in self.robot.calls])
        self.assertEqual(self.task.last['state'],'failed')
        self.assertFalse(self.task.last['delivered'])


if __name__=='__main__':unittest.main()
