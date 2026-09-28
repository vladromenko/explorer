import unittest
from arm_feedback import describe

class ArmFeedbackTests(unittest.TestCase):
    def test_absent_transport_is_not_hardware_impossibility(self):
        r=describe({'arm_feedback_transport':{'publishers':[]}},now=100)
        self.assertFalse(r['measured']['available']);self.assertFalse(r['measured']['hardware_readback_impossible'])
        self.assertEqual(r['measured']['reason'],'NO_PUBLISHER_IN_CURRENT_ROS_GRAPH')
        self.assertTrue(all(j['estimated_deg'] is None for j in r['joints']))
    def test_topic_presence_never_promotes_echo_to_measurement(self):
        r=describe(dict(arm_feedback=[90]*6,sensor_age={'arm':.01},
            arm_feedback_transport={'publishers':[{'node':'/unknown'}]}),now=100)
        self.assertIsNone(r['measured']['available']);self.assertIsNone(r['measured']['values'])
    def test_commanded_estimate_retains_unknown_error(self):
        r=describe(dict(arm_command_state=dict(at=99,servo_deg=[90]*6,phase='command_elapsed_observation_required')),now=100)
        self.assertEqual(r['joints'][0]['estimated_deg'],90)
        self.assertIsNone(r['joints'][0]['measured_deg']);self.assertIsNone(r['joints'][0]['error_bound_deg'])
