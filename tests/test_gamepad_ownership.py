import unittest
from tests.test_gamepad_arm_binding import GamepadBindingTests
class GamepadOwnershipTests(GamepadBindingTests):
 def test_panel_timeout_disconnects_shared_owner(self):
  self.p.select('TELEOP');self.p.teleop.stop_latched=False;self.p.lease=0;self.p.feed(self.now);self.assertIsNone(self.p.teleop.owner);self.assertEqual(self.p.mode,'DISARMED')
