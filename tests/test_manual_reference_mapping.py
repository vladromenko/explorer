"""Prevent relabeling a firmware target as a different physical reference."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from controller_feedback import vendor_calibration
from manual_reference_host import effective_calibration
from manual_reference_client import ManualClient


class ReferenceMappingTests(unittest.TestCase):
    def test_probe_requires_response_and_retries_no_motion(self):
        client=ManualClient.__new__(ManualClient)
        client.close=Mock()
        client._exchange=Mock(side_effect=[None,dict(accepted=False,error=16)])
        client._probe_transport()
        self.assertEqual([c.args for c in client._exchange.call_args_list],
                         [('MR_DIAGNOSTIC',),('MR_DIAGNOSTIC',)])
        client.close.assert_not_called()
        client._exchange=Mock(return_value=None)
        with self.assertRaisesRegex(ValueError,'команды приводам не отправлены'):
            client._probe_transport()
        client.close.assert_called_once()

    def test_request_discovery_does_not_skip_result_discovery(self):
        client=ManualClient.__new__(ManualClient)
        matched=False
        def discover(*args,**kwargs):
            nonlocal matched
            matched=True
        client.node=object()
        client.pub=SimpleNamespace(get_subscription_count=lambda:1)
        client.sub=SimpleNamespace(get_publisher_count=lambda:int(matched))
        client.rclpy=SimpleNamespace(spin_once=Mock(side_effect=discover))
        client._wait_transport()
        client.rclpy.spin_once.assert_called_once()

    def test_gripper_reference_keeps_vendor_mapping(self):
        nominal=vendor_calibration()
        ref=dict(reference_valid=True,raw_reference=[2000,2000,2000,2000,1487,2850],
                 slopes=[c.radians_per_tick for c in nominal],
                 offsets=[c.radians_at_raw_zero for c in nominal],
                 lower=[c.lower for c in nominal],upper=[c.upper for c in nominal])
        calibrated=effective_calibration(nominal,ref)
        for before,after in zip(nominal,calibrated):
            self.assertEqual(before.physical_degrees_at_raw_zero,after.physical_degrees_at_raw_zero)
            self.assertEqual(before.physical_degrees_per_tick,after.physical_degrees_per_tick)
        gripper=calibrated[5]
        self.assertAlmostEqual(gripper.physical_degrees_at_raw_zero+2850*gripper.physical_degrees_per_tick,
                               159.545454545)
        self.assertAlmostEqual((180-gripper.physical_degrees_at_raw_zero)/gripper.physical_degrees_per_tick,3100)


if __name__=='__main__':unittest.main()
