import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from arm_preparation import stop_base_before_prepare

class ArmPreparationTests(unittest.TestCase):
    def state(self):
        return dict(at=time.time(),stop_latched=True,velocity=[0,0,0],battery=12,
                    sensor_age={'odom':.02,'battery':.1},last_request={'id':'current','ok':True},
                    odom_velocity=[0,0,0],odom_received_monotonic=time.monotonic())

    def run_state(self,state,timeout=.05):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir()
            (root/'data/status.json').write_text(json.dumps(state))
            return stop_base_before_prepare(root,lambda:{'id':'current'},timeout=timeout)

    def test_stop_must_be_acknowledged_by_this_request(self):
        for ack in ({'id':'older','ok':True},{'id':'current','ok':False}):
            state=self.state();state['last_request']=ack
            with self.assertRaises(ValueError):self.run_state(state)

    def test_zero_command_does_not_mean_base_has_stopped(self):
        for velocity in ([.03,0,0],[0,-.02,0],[0,0,.05],[float('nan'),0,0],None):
            state=self.state();state['odom_velocity']=velocity
            with self.assertRaises(ValueError):self.run_state(state)

    def test_stale_status_or_frozen_odometry_is_rejected(self):
        for field,value in (('at',0),('odom_received_monotonic',0),('odom_received_monotonic',None)):
            state=self.state();state[field]=value
            with self.assertRaises(ValueError):self.run_state(state)
        with self.assertRaises(ValueError):self.run_state(self.state(),timeout=.35)

    def test_ack_with_nonzero_command_is_rejected(self):
        state=self.state();state['velocity']=[.1,0,0]
        with self.assertRaises(ValueError):self.run_state(state)

    def test_fresh_stationary_samples_allow_preparation(self):
        def read(*args,**kwargs):return json.dumps(self.state())
        with patch.object(Path,'read_text',read):
            result=stop_base_before_prepare('/unused',lambda:{'id':'current'},timeout=1)
        self.assertEqual(result['odom_velocity'],[0,0,0])
