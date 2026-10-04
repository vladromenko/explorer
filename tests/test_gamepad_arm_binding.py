import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from gamepad_panel import GamepadPanel

class GamepadBindingTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name);(root/'config').mkdir();(root/'data').mkdir()
  (root/'config/gamepad.yaml').write_text((Path(__file__).parents[1]/'config/gamepad.yaml').read_text());(root/'data/status.json').write_text(json.dumps({'at':time.time(),'stop_latched':False,'commissioning':{}}))
  with patch('gamepad_panel.threading.Thread'),patch('manual_teleop.threading.Thread'):
   self.p=GamepadPanel(root,Mock(),Mock(),Mock(),Mock())
  self.p.connected=True;self.p.heartbeat(True);self.p.axes={str(a['code']):a['center'] for a in self.p.config['axes'].values()};self.b=self.p.config['buttons'];self.now=time.monotonic()
 def press(self,name,value=1):self.p.decide(self.b[name],value,1,self.now)
 def test_full_mapping_and_simultaneous_inputs(self):
  self.p.select('TELEOP');self.p.teleop.stop_latched=False
  self.p.axes.update({'1':0,'0':0,'2':0,'5':0,'16':-1,'17':-1});self.press('r1');self.press('y');self.press('r2')
  v=self.p._inputs();self.assertGreater(v['forward'],0);self.assertGreater(v['left'],0);self.assertEqual(v['turn_left'],0);self.assertEqual(v['joint1_decrease'],1);self.assertEqual(v['joint2_increase'],1);self.assertGreater(v['joint3_increase'],0);self.assertEqual(v['pitch_down'],1);self.assertEqual(v['wrist_left'],1);self.assertEqual(v['grip_close'],1)
 def test_right_stick_turn_is_proportional_without_arm_modifier(self):
  self.p.axes['2']=64;value=self.p._inputs()['turn_left'];self.assertGreater(value,.4);self.assertLess(value,.7)
 def test_x_toggles_precision_b_stops_and_start_requires_neutral(self):
  self.p.select('TELEOP');self.press('x');self.assertTrue(self.p.precision);self.press('b');self.assertTrue(self.p.teleop.stop_latched)
  self.p.axes['1']=0
  with self.assertRaisesRegex(ValueError,'отпустите'):self.press('start')
  self.p.axes['1']=128;self.p.axes['16']=0;self.p.axes['17']=0;self.p.keys.clear();self.p.teleop.update('gamepad',{},True);self.press('start');self.assertFalse(self.p.teleop.stop_latched)
 def test_select_and_stick_click_have_no_motion_binding(self):
  before=self.p._inputs();self.press('select');self.press('left_stick');self.press('right_stick');self.assertEqual(before,self.p._inputs())
 def test_disconnect_releases_scoped_owner(self):
  self.p.select('TELEOP');self.p.teleop.drive_active=True;self.p.heartbeat(False);self.assertEqual(self.p.mode,'DISARMED');self.p.teleop.stop_all.assert_not_called()
 def test_stale_input_stream_sends_neutral(self):
  self.p.select('TELEOP');self.p.teleop.stop_latched=False;self.p.axes['1']=0;self.p.last_event=self.now-1
  self.p.feed(self.now);self.assertEqual(self.p.teleop.inputs,{})

if __name__=='__main__':unittest.main()
