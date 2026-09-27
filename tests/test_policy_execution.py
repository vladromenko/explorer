import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import threading
import numpy as np
from policy_execution import PolicyExecution,bounded_goal

class PolicyExecutionTests(unittest.TestCase):
    def test_model_cannot_expand_envelope_by_rounding_or_nan(self):
        s=[90,125,3,0,90,30]
        for a in ([92.1,125,3,0,90,30],[90,125,3,-.1,90,30],[90,125,3,0,90,float('nan')],[1,2]):
            with self.assertRaises(ValueError):bounded_goal(s,a)
        self.assertEqual(bounded_goal(s,[91.8,123.2,3,0,90,30]),[92,123,3,0,90,30])

    def make_runner(self,root):
        jobs=Mock();jobs.status.return_value={'jobs':[]}
        teaching=Mock();teaching.lock=threading.Lock();teaching.active=None
        preview=Mock();preview.lock=threading.Lock()
        arm=Mock();arm.stop_revision=0
        return PolicyExecution(root,jobs,teaching,arm,preview)

    def test_no_trained_model_never_starts_or_reserves_arm(self):
        with tempfile.TemporaryDirectory() as root,patch('policy_execution.training_budget'):
            p=self.make_runner(root)
            with self.assertRaises(ValueError):p.start('sock',True)
            self.assertFalse(p.lock.locked());self.assertFalse(p.teaching.lock.locked());p.manual.move.assert_not_called()

    def test_loss_of_hold_and_stop_are_independent_guards(self):
        p=self.make_runner('/tmp');p.revision=0;p.lease=time.monotonic()+1
        p.permit();p.lease=time.monotonic()-1
        with self.assertRaises(ValueError):p.permit()
        p.lease=time.monotonic()+1;p.manual.stop_revision=1
        with self.assertRaises(ValueError):p.permit()

    def test_invalid_session_cannot_extend_lease(self):
        p=self.make_runner('/tmp');p.lock.acquire();p.session='valid';p.lease=1
        with self.assertRaises(ValueError):p.heartbeat('other',True)
        self.assertEqual(p.lease,1);p.lock.release()

    def test_supervisor_rejects_stale_or_oversized_prediction_before_actuation(self):
        for action,stamp in [([94,125,3,0,90,30],10.),([92,125,3,0,90,30],9.)]:
            with self.subTest(action=action,stamp=stamp),tempfile.TemporaryDirectory() as root:
                p=self.make_runner(root);folder=Path(root);p.revision=0;p.unit='fake';p.lease=time.monotonic()+10
                p.lock.acquire();p.teaching.lock.acquire()
                pose=[90,125,3,0,90,30];p.observation=Mock(return_value=(pose,np.zeros((10,10,3),dtype=np.uint8),10.))
                p.wait_file=Mock(side_effect=[{},dict(observed_at=stamp,start_deg=pose,proposed_deg=action)])
                with patch('policy_execution.subprocess.Popen') as process,patch('policy_execution.subprocess.run'),patch('policy_execution.training_budget'),patch('policy_execution.time.time',return_value=10.5):
                    p.run(folder)
                p.manual.move.assert_not_called();self.assertEqual(p.state['phase'],'stopped')
                self.assertFalse(p.lock.locked());self.assertFalse(p.teaching.lock.locked())

    def test_valid_step_passes_through_guard_and_next_failure_stops_sequence(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.make_runner(root);folder=Path(root);p.revision=0;p.unit='fake';p.lease=time.monotonic()+10
            p.lock.acquire();p.teaching.lock.acquire();pose=[90,125,3,0,90,30]
            p.observation=Mock(side_effect=[(pose,np.zeros((10,10,3),dtype=np.uint8),10.),ValueError('new frame unavailable')])
            p.wait_file=Mock(side_effect=[{},dict(observed_at=10.,start_deg=pose,proposed_deg=[92,125,3,0,90,30])])
            p.manual.move.return_value={'commanded_only':True}
            with patch('policy_execution.subprocess.Popen'),patch('policy_execution.subprocess.run'),patch('policy_execution.training_budget'),patch('policy_execution.time.time',return_value=10.5):
                p.run(folder)
            p.manual.move.assert_called_once();self.assertEqual(p.manual.move.call_args.kwargs['source'],'supervised_policy')
            self.assertEqual(p.state['steps'],1);self.assertEqual(p.state['phase'],'stopped')
            self.assertFalse(p.state['success_verified'])
