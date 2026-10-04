import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
from teaching import Demonstrations,TeachingController
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

    def test_r1_dpad_directly_controls_shoulder_and_base_joint(self):
        panel=self.panel();panel.axes={str(a['code']):a['center'] for a in panel.config['axes'].values()}
        panel.decide(panel.config['buttons']['r1'],1,1,10)
        panel.decide(17,-1,3,10);self.assertEqual(panel._inputs()['joint2_increase'],1)
        panel.decide(16,1,3,10);self.assertEqual(panel._inputs()['joint1_increase'],1)

    def test_select_and_stick_click_do_not_create_motion(self):
        panel=self.panel();panel.axes={str(a['code']):a['center'] for a in panel.config['axes'].values()}
        before=panel._inputs();b=panel.config['buttons']
        for name in ('select','left_stick','right_stick'):panel.decide(b[name],1,1,10)
        self.assertEqual(before,panel._inputs())

    def test_device_poll_keeps_held_axes_live_and_clears_stale_buttons(self):
        panel=self.panel();b=panel.config['buttons'];panel.keys={b['l1']}
        device=Mock();device.active_keys.return_value=[b['r1']]
        device.absinfo.side_effect=lambda code:Mock(value=0 if code==1 else (-1 if code==17 else 128))
        panel.poll_device(device,10.)
        self.assertEqual(panel.keys,{b['r1']});self.assertEqual(panel.last_event,10.)
        self.assertGreater(panel._inputs()['forward'],.9)
        self.assertEqual(panel._inputs()['joint2_increase'],1.)

    def test_normal_manual_arm_control_uses_direct_eight_degree_segment_and_fast_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=TeachingController(directory);controller.pose=Mock(return_value=[90]*6)
            controller.move=Mock(return_value={'command_completed':True})
            controller.teleop(Mock(),[0,0,0],[0,1,0,0,0,0],True)
            args=controller.move.call_args
            self.assertEqual(args.args[1][1],98)
            self.assertEqual(args.kwargs['speed'],'fast')

    def test_low_or_stale_power_rejects_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir()
            for voltage,at in [(11.1,100),(12.3,90)]:
                (root/'data/power.json').write_text(json.dumps(dict(at=at,state='IDLE',battery_voltage_v=voltage)))
                with patch('lerobot_bridge.time.time',return_value=100):
                    with self.assertRaises(ValueError):training_budget(root)

if __name__=='__main__':unittest.main()
