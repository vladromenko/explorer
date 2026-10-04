import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gamepad_panel import GamepadPanel


class GamepadBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);(root/'config').mkdir();(root/'data').mkdir()
        (root/'config/gamepad.yaml').write_text((Path(__file__).parents[1]/'config/gamepad.yaml').read_text())
        (root/'data/status.json').write_text(json.dumps({'at':time.time(),'stop_latched':False,'commissioning':{}}))
        with patch('gamepad_panel.threading.Thread'),patch('manual_teleop.threading.Thread'):
            self.panel=GamepadPanel(root,Mock(),Mock(),Mock(),Mock())
        self.p=self.panel
        self.panel.connected=True;self.panel.heartbeat(True)
        self.panel.axes={str(item['code']):item['center'] for item in self.panel.config['axes'].values()}
        self.buttons=self.panel.config['buttons'];self.now=time.monotonic()

    def press(self,name,value=1):self.panel.decide(self.buttons[name],value,1,self.now)

    def test_cartesian_ls_rs_and_gripper_are_simultaneous(self):
        self.panel.select('TELEOP');self.panel.axes.update({'1':0,'0':0,'2':0,'5':0});self.press('r2')
        values=self.panel._inputs()
        for action in ('forward','left','arm_x_forward','arm_y_left','grip_close'):
            self.assertGreater(values[action],0)
        self.assertEqual(values.get('turn_left',0),0)

    def test_bumpers_always_turn_base(self):
        self.panel.select('TELEOP');self.press('l1');self.press('r1')
        values=self.panel._inputs();self.assertEqual(values['turn_left'],1);self.assertEqual(values['turn_right'],1)
        self.assertFalse(any(name.startswith('joint') for name in values))

    def test_select_toggles_joint_layout(self):
        self.panel.select('TELEOP');self.press('select');self.press('select',0)
        self.assertEqual(self.panel.arm_mode,'joint')
        self.panel.axes.update({'2':0,'5':0,'16':-1,'17':-1});self.press('y');self.press('r2')
        values=self.panel._inputs()
        self.assertEqual(values['joint1_decrease'],1);self.assertEqual(values['joint2_increase'],1)
        self.assertEqual(values['joint3_increase'],1);self.assertEqual(values['pitch_up'],1)
        self.assertEqual(values['wrist_left'],1);self.assertEqual(values['grip_close'],1)

    def test_mode_change_with_deflected_stick_waits_for_neutral(self):
        self.panel.select('TELEOP');self.panel.axes['2']=0;self.press('select')
        self.assertTrue(self.panel.arm_neutral_required)
        self.assertNotIn('joint1_decrease',self.panel._inputs())
        self.panel.axes['2']=128;self.press('select',0);self.panel.feed(self.now)
        self.assertFalse(self.panel.arm_neutral_required)

    def test_x_stop_and_start_require_neutral(self):
        self.panel.select('TELEOP');self.press('x');self.assertTrue(self.panel.precision)
        self.press('b');self.assertTrue(self.panel.teleop.stop_latched);self.press('b',0)
        self.panel.axes['1']=0
        with self.assertRaisesRegex(ValueError,'отпустите'):self.press('start')
        self.panel.axes['1']=128;self.panel.keys.clear();self.panel.teleop.update('gamepad',{},True)
        self.press('start');self.assertFalse(self.panel.teleop.stop_latched)

    def test_disconnect_releases_only_selected_source(self):
        self.panel.select('TELEOP');self.panel.teleop.drive_active=True;self.panel.heartbeat(False)
        self.assertEqual(self.panel.mode,'DISARMED');self.panel.teleop.stop_all.assert_not_called()


if __name__=='__main__':unittest.main()
