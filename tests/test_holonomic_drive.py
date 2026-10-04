import math,unittest,yaml
from pathlib import Path
from holonomic_drive import velocity,blocked_by

class HolonomicTests(unittest.TestCase):
    def setUp(self):self.config=yaml.safe_load((Path(__file__).parents[1]/'config/gamepad.yaml').read_text())
    def test_directions_diagonal_and_turn(self):
        c=self.config
        for axes,expected in [({'1':0},[.12,0,0]),({'1':255},[-.12,0,0]),({'0':0},[0,.12,0]),({'0':255},[0,-.12,0]),({'2':0},[0,0,.25]),({'2':255},[0,0,-.25])]:
            with self.subTest(axes=axes):
                for a,b in zip(velocity(axes,c),expected):self.assertAlmostEqual(a,b)
        a=velocity({'0':0,'1':0,'2':255},c)
        self.assertAlmostEqual(math.hypot(*a[:2]),.12);self.assertEqual(a[2],-.25)
    def test_neutral_and_invalid_input(self):
        self.assertEqual(velocity({},self.config),[0,0,0])
        self.assertEqual(velocity({'0':130,'1':124,'2':129},self.config),[0,0,0])
        for v in (float('nan'),256,-1):
            with self.assertRaises(ValueError):velocity({'1':v},self.config)
    def test_receiver_is_not_radio_evidence(self):
        state={'at':10,'stop_latched':False,'commissioning':dict(base_commissioned=True,mcu_watchdog_verified=True,lidar_tf_validated=True)}
        self.config['operator_chassis_accepted']=False
        self.assertTrue(blocked_by(state,self.config,10.1))
        self.config.update(radio_loss_verified=True,continuous_motion_enabled=True)
        self.assertEqual(blocked_by(state,self.config,10.1),[])
        self.assertTrue(blocked_by(state,self.config,12))
        self.config.update(operator_chassis_accepted=True,radio_loss_verified=False)
        self.assertEqual(blocked_by(state,self.config,10.1),[])
