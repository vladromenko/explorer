import math,unittest,yaml
from pathlib import Path
from holonomic_drive import velocity,blocked_by

class HolonomicTests(unittest.TestCase):
    def setUp(self):self.config=yaml.safe_load((Path(__file__).parents[1]/'config/gamepad.yaml').read_text())
    def test_directions_diagonal_and_turn(self):
        c=self.config
        for axes,expected in [({'1':0},[.04,0,0]),({'1':255},[-.04,0,0]),({'0':0},[0,.04,0]),({'0':255},[0,-.04,0]),({'2':0},[0,0,.1]),({'2':255},[0,0,-.1])]:
            with self.subTest(axes=axes):
                for a,b in zip(velocity(axes,c),expected):self.assertAlmostEqual(a,b)
        a=velocity({'0':0,'1':0,'2':255},c)
        self.assertAlmostEqual(math.hypot(*a[:2]),.04);self.assertEqual(a[2],-.1)
    def test_neutral_and_invalid_input(self):
        self.assertEqual(velocity({},self.config),[0,0,0])
        self.assertEqual(velocity({'0':130,'1':124,'2':129},self.config),[0,0,0])
        for v in (float('nan'),256,-1):
            with self.assertRaises(ValueError):velocity({'1':v},self.config)
    def test_receiver_is_not_radio_evidence(self):
        state={'at':10,'stop_latched':False,'commissioning':dict(base_commissioned=True,mcu_watchdog_verified=True,lidar_tf_validated=True)}
        self.assertTrue(blocked_by(state,self.config,10.1))
        self.config.update(radio_loss_verified=True,continuous_motion_enabled=True)
        self.assertEqual(blocked_by(state,self.config,10.1),[])
        self.assertTrue(blocked_by(state,self.config,12))

class GamepadIntegrationTests(unittest.TestCase):
    def test_evdev_drive_is_gated_and_panel_loss_stops(self):
        import tempfile,json,threading,time
        from unittest.mock import Mock,patch
        from gamepad_panel import GamepadPanel
        with tempfile.TemporaryDirectory() as directory,patch('gamepad_panel.threading.Thread'):
            root=Path(directory);(root/'config').mkdir();(root/'data').mkdir()
            config=yaml.safe_load((Path(__file__).parents[1]/'config/gamepad.yaml').read_text())
            (root/'config/gamepad.yaml').write_text(yaml.safe_dump(config))
            state={'at':time.time(),'stop_latched':False,'commissioning':dict(base_commissioned=True,mcu_watchdog_verified=True,lidar_tf_validated=True)}
            (root/'data/status.json').write_text(json.dumps(state))
            stop=Mock();drive=Mock();g=GamepadPanel(root,Mock(),stop,drive);now=time.monotonic();g.heartbeat(True)
            g.decide(config['buttons']['x'],1,1,now);g.decide(config['buttons']['l1'],1,1,now)
            g.decide(config['axes']['left_x']['code'],0,3,now);g.drive_tick(now)
            drive.assert_not_called()
            g.config.update(radio_loss_verified=True,continuous_motion_enabled=True);g.drive_tick(now)
            drive.assert_called_once_with([0,.04,0])
            g.heartbeat(False);stop.assert_called();self.assertFalse(g.drive_active)
