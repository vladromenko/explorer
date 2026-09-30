import threading
import unittest
from gamepad_panel import GamepadPanel
from unittest.mock import Mock

class GamepadOwnershipTests(unittest.TestCase):
    def panel(self):
        p=GamepadPanel.__new__(GamepadPanel)
        p.lock=threading.Lock();p.stop=Mock();p.release=Mock();p.status=Mock(return_value={})
        p.drive_active=False;p.mode='DISARMED';p.neutral=False;p.dpad_y=0
        p.config={'buttons':dict(l1=1,a=2,x=3,b=4,r1=5),'arm_modifier_button':'r1','axes':{'right_y':{'code':6}}}
        p.keys=set();p.lease=0
        return p
    def test_inactive_browser_disconnect_never_stops_autonomy(self):
        p=self.panel();p.heartbeat(False);p.stop.assert_not_called();p.release.assert_not_called()
    def test_release_is_scoped_and_emergency_button_remains_global(self):
        p=self.panel();p.drive_active=True;p.heartbeat(False)
        p.release.assert_called_once();p.stop.assert_not_called()
        p.decide(4,1,1,1);p.stop.assert_called_once()
    def test_idle_l1_release_never_stops_autonomy(self):
        p=self.panel();p.decide(1,0,1,1);p.stop.assert_not_called();p.release.assert_not_called()
