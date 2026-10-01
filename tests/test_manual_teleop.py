import threading,time,unittest
from unittest.mock import Mock
from manual_teleop import ManualTeleop,keyboard_inputs

class ManualTeleopTests(unittest.TestCase):
 def panel(self):
  t=ManualTeleop.__new__(ManualTeleop);t.drive=Mock();t.release=Mock();t.stop_all=Mock();t.teaching=Mock();t.lock=threading.RLock();t.owner=None;t.lease=0;t.inputs={};t.precision=False;t.stop_latched=True;t.neutral_seen=False;t.drive_active=False;t.arm_busy=False;t.error=None;t.arm=None;t.model=None;t.last_arm=0;t.closed=True;t.resume_callback=Mock();t.takeover_callback=Mock();t.takeover_active=False;return t
 def test_keyboard_multikey_and_keyup_keeps_remaining_action(self):
  t=self.panel();t.update('keyboard',{},True);t.resume('keyboard',True);v=keyboard_inputs(['w','a','e','arrowup','b']);t.update('keyboard',v,True);t.tick()
  t.drive.assert_called_with([.1/2**.5,.09/2**.5,-.25])
  v=keyboard_inputs(['w','e','arrowup','b']);t.update('keyboard',v,True);t.tick();t.drive.assert_called_with([.1,0,-.25])
 def test_precision_scales_chassis_and_arm(self):
  t=self.panel();t.update('keyboard',{},True);t.resume('keyboard',True);v=keyboard_inputs(['w','arrowup','b']);t.update('keyboard',v,True,True)
  self.assertAlmostEqual(t._drive_vector(v,True)[0],.04);xyz,j=t._arm_intent(v,True);self.assertEqual(xyz,[0,0,.45]);self.assertEqual(j,[0,0,.45])
 def test_stop_held_key_and_explicit_neutral_resume(self):
  t=self.panel();t.update('keyboard',keyboard_inputs(['w']),True);t.stop();t.tick();t.drive.assert_not_called()
  with self.assertRaisesRegex(ValueError,'отпустите'):t.resume('keyboard',True)
  t.update('keyboard',{},True);t.resume('keyboard',True);self.assertFalse(t.stop_latched)
 def test_first_motion_announces_manual_takeover_once(self):
  t=self.panel();t.update('keyboard',keyboard_inputs(['w']),True);t.update('keyboard',keyboard_inputs(['w']),True)
  t.takeover_callback.assert_called_once();t.update('keyboard',{},True);t.update('keyboard',keyboard_inputs(['a']),True)
  self.assertEqual(t.takeover_callback.call_count,2)
 def test_focus_loss_releases_without_global_stop(self):
  t=self.panel();t.owner='keyboard';t.drive_active=True;t.disconnect('keyboard');t.release.assert_called_once();t.stop_all.assert_not_called()
 def test_owner_transfer_releases_old_stream(self):
  t=self.panel();t.owner='keyboard';t.drive_active=True;t.update('gamepad',{},True);self.assertEqual(t.owner,'gamepad');t.release.assert_called_once()

if __name__=='__main__':unittest.main()
