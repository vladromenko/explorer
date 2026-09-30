import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from teaching import Demonstrations
from gamepad_panel import GamepadPanel
from lerobot_bridge import training_budget

ROOT=Path(__file__).resolve().parents[1]

class OperatorLearningTests(unittest.TestCase):
    def test_only_completed_successful_operator_demos_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Demonstrations(directory)
            for i,(state,outcome,source) in enumerate([
                    ('complete','success','operator_demonstration'),
                    ('recording','success','operator_demonstration'),
                    ('complete','unknown','operator_demonstration'),
                    ('complete','success','observed_calibration')]):
                folder=Path(directory)/str(i);folder.mkdir()
                (folder/'episode.json').write_text(json.dumps(dict(id=str(i),name='sock',
                    state=state,outcome=outcome,source=source,label_source='operator',steps=[{}]*3)))
            self.assertEqual([e['id'] for e in store.eligible()],['0'])

    def panel(self):
        with patch.object(threading.Thread,'start'):
            return GamepadPanel(ROOT,None,lambda:None)

    def test_dpad_requires_r1_new_press_and_live_lease(self):
        panel=self.panel();panel.mode='ARM';panel.lease=11
        self.assertIsNone(panel.decide(17,-1,3,10))
        panel.decide(17,0,3,10);panel.decide(311,1,1,10)
        self.assertEqual(panel.decide(17,-1,3,10),(1,5))
        self.assertIsNone(panel.decide(17,-1,3,10))
        panel.decide(17,0,3,12)
        self.assertIsNone(panel.decide(17,-1,3,12))

    def test_gamepad_proposals_have_no_actuator_path(self):
        panel=self.panel();panel.perform(2,-2)
        self.assertFalse(panel.proposal['executed'])
        self.assertTrue(panel.status()['continuous_motion_enabled'])
        self.assertTrue(panel.status()['operator_chassis_accepted'])
        self.assertFalse(panel.status()['radio_link_verified'])

    def test_low_or_stale_power_rejects_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir()
            for voltage,at in [(11.1,100),(12.3,90)]:
                (root/'data/power.json').write_text(json.dumps(dict(at=at,state='IDLE',battery_voltage_v=voltage)))
                with patch('lerobot_bridge.time.time',return_value=100):
                    with self.assertRaises(ValueError):training_budget(root)

if __name__=='__main__':unittest.main()
