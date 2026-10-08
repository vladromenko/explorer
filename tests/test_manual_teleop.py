import threading
import time
import unittest
from unittest.mock import Mock, patch

from manual_controls import keyboard_inputs
from manual_teleop import ManualTeleop


class ManualTeleopTests(unittest.TestCase):
    def panel(self):
        with patch('manual_teleop.threading.Thread'):
            return ManualTeleop(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

    def select(self, panel, source='keyboard'):
        panel.select_source(source, True, True)
        panel.update(source, {}, True)
        panel.resume(source, True)

    def test_physical_keyboard_codes_and_both_arm_modes(self):
        cart=keyboard_inputs(['KeyW','KeyI','KeyJ','KeyM'],'cartesian')
        self.assertEqual(set(cart),{'forward','arm_x_forward','arm_y_left','grip_close'})
        joint=keyboard_inputs(['KeyW','KeyI','KeyJ','KeyU','KeyY','Comma','KeyM'],'joint')
        self.assertEqual(set(joint),{'forward','joint2_increase','joint1_decrease','joint3_increase','pitch_up','wrist_left','grip_close'})
        with self.assertRaises(ValueError):keyboard_inputs(['ц'],'cartesian')

    def test_multiaxis_drive_is_normalized_before_precision(self):
        panel=self.panel()
        values=keyboard_inputs(['KeyW','KeyA','KeyE'],'cartesian')
        normal=panel._drive_vector(values,False);precision=panel._drive_vector(values,True)
        self.assertLessEqual(max(abs(normal[0]/.8-normal[1]/.72-normal[2]/1.67),
                                 abs(normal[0]/.8+normal[1]/.72+normal[2]/1.67)),1.000001)
        self.assertEqual([round(value*.1,8) for value in normal],[round(value,8) for value in precision])

    def test_packets_cannot_steal_owner(self):
        panel=self.panel();panel.select_source('keyboard',True,True)
        with self.assertRaisesRegex(ValueError,'не выбран'):panel.update('gamepad',{},True)
        self.assertEqual(panel.owner,'keyboard')

    def test_stop_requires_observed_neutral(self):
        panel=self.panel();panel.select_source('keyboard',True,True)
        panel.update('keyboard',keyboard_inputs(['KeyW']),True);panel.stop()
        with self.assertRaisesRegex(ValueError,'отпустите'):panel.resume('keyboard',True)
        panel.update('keyboard',{},True);panel.resume('keyboard',True)
        self.assertFalse(panel.stop_latched)

    def test_mode_change_invalidates_arm_but_preserves_base_and_gripper(self):
        panel=self.panel();self.select(panel)
        panel.update('keyboard',keyboard_inputs(['KeyW','KeyI','KeyM'],'cartesian'),True)
        generation=panel.generation;panel.set_arm_mode('keyboard','joint')
        self.assertGreater(panel.generation,generation)
        self.assertEqual(set(panel.inputs),{'forward','grip_close'})

    def test_fractional_joint_motion_is_proportional_without_eight_degree_jump(self):
        panel=self.panel();self.select(panel);panel.set_arm_mode('keyboard','joint')
        panel.last_integrator=10.0
        small={'joint2_increase':.2};medium={'joint2_increase':.6};large={'joint2_increase':1.0}
        steps=[]
        for values in (small,medium,large):
            panel.joint_residual=[0.0]*6;panel.last_integrator=10.0
            segment=panel._prepare_arm_segment_locked(values,10.1,False)
            steps.append(0 if segment is None else segment[1][1])
        self.assertEqual(steps,[0,1,2])
        self.assertNotIn(8,steps)

    def test_release_clears_fractional_remainder(self):
        panel=self.panel();self.select(panel);panel.set_arm_mode('keyboard','joint')
        panel.last_integrator=10.0
        self.assertIsNone(panel._prepare_arm_segment_locked({'joint2_increase':.2},10.1,False))
        panel._prepare_arm_segment_locked({},10.2,False)
        self.assertEqual(panel.joint_residual,[0.0]*6)

    def test_fast_arm_speed_is_independent_of_base_and_short_y_a_is_not_lost(self):
        panel=self.panel();self.select(panel);panel.set_arm_mode('keyboard','joint')
        normal_base=panel._drive_vector({'forward':1.},False)
        panel.set_arm_speed('keyboard','fast')
        self.assertEqual(panel.status()['arm_speed'],'fast')
        self.assertEqual(panel._drive_vector({'forward':1.},False),normal_base)
        panel.last_integrator=10.;panel.arm_busy=True
        panel.update('keyboard',{'joint3_increase':1.},True)
        self.assertIsNone(panel._prepare_arm_segment_locked({'joint3_increase':1.},10.05,False))
        panel.update('keyboard',{},True)
        panel.arm_busy=False
        segment=panel._prepare_arm_segment_locked({},10.10,False)
        self.assertIsNotNone(segment)
        self.assertGreaterEqual(segment[1][2],1)
        self.assertLessEqual(segment[1][2],6)


if __name__=='__main__':unittest.main()
