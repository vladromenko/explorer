"""Input/ownership regression tests; never opens input devices or ROS."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
import yaml
from gamepad_panel import GamepadPanel


class GamepadArmBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);(root/'config').mkdir();(root/'data').mkdir()
        config=Path(__file__).parents[1]/'config/gamepad.yaml'
        (root/'config/gamepad.yaml').write_text(config.read_text())
        (root/'data/status.json').write_text(json.dumps(dict(at=time.time(),stop_latched=True,commissioning={})))
        self.teaching=Mock();self.stop=Mock();self.release=Mock()
        with patch('gamepad_panel.threading.Thread'):
            self.panel=GamepadPanel(root,self.teaching,self.stop,Mock(),self.release)
        # Factory executor has no profile and no native attribute.
        self.arm=type('FactoryArm',(),{'stop':Mock()})()
        self.panel.bind_arm(self.arm,Mock())
        self.now=time.monotonic();self.panel.heartbeat(True)
        self.buttons=self.panel.config['buttons']

    def press(self,name,value=1):
        self.panel.decide(self.buttons[name],value,1,self.now)

    def arm_mode(self):
        self.press('a');self.press('l1')

    def test_factory_cancellation_on_release_and_hidden_panel(self):
        self.arm_mode();self.assertTrue(self.arm.gamepad_permit())
        self.press('l1',0);self.arm.stop.assert_called_once()
        self.assertFalse(self.arm.gamepad_permit())
        self.panel.heartbeat(False);self.assertEqual(self.arm.stop.call_count,2)
        self.assertEqual(self.panel.mode,'DISARMED')
        self.stop.assert_not_called()

    def test_one_step_per_neutral_and_valid_live_permission(self):
        self.arm_mode();axis=self.panel.config['axes']['right_y']
        self.panel.decide(axis['code'],axis['center'],3,self.now)
        step=self.panel.decide(axis['code'],axis['minimum'],3,self.now)
        self.assertEqual(step,(1,5))
        self.assertIsNone(self.panel.decide(axis['code'],axis['minimum'],3,self.now))
        self.panel.perform(*step,self.now+.5)
        self.teaching.jog.assert_called_once_with(1,5,True,self.now+.5)
        self.assertTrue(self.panel.proposal['executed'])
        self.panel.lease=0
        self.panel.perform(*step,self.now+.5)
        self.assertEqual(self.teaching.jog.call_count,1)
        self.assertFalse(self.panel.proposal['executed'])

    def test_changing_mode_cancels_factory_arm_and_does_not_send_motion(self):
        self.arm_mode();self.press('l1',0);self.arm.stop.reset_mock()
        self.press('x');self.arm.stop.assert_called_once()
        self.assertEqual(self.panel.mode,'DRIVE')
        self.panel.drive.assert_not_called()

    def test_idle_status_reports_blockers_before_trying_to_drive(self):
        status=self.panel.status()
        self.assertTrue(status['drive_blocked_by'])
        self.assertFalse(status['radio_link_verified'])
        self.panel.config['radio_loss_verified']=True
        self.assertTrue(self.panel.status()['radio_link_verified'])
        (self.panel.root/'data/status.json').write_text('null')
        self.assertEqual(self.panel.status()['drive_blocked_by'],['Нет достоверного состояния робота'])

    def test_direction_uses_calibration_instead_of_fixed_polarity(self):
        self.arm_mode();axis=self.panel.config['axes']['right_y']
        axis.update(minimum=0,center=100,maximum=200,inverted=False)
        self.panel.decide(axis['code'],100,3,self.now)
        self.assertEqual(self.panel.decide(axis['code'],200,3,self.now),(1,5))

    def test_triggers_control_gripper_in_arm_mode(self):
        self.arm_mode()
        self.assertEqual(self.panel.decide(self.buttons['l2'],1,1,self.now),(6,-5))
        self.assertEqual(self.panel.decide(self.buttons['r2'],1,1,self.now),(6,5))


if __name__=='__main__':unittest.main()
