import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import json
from std_msgs.msg import String
from core import Core
class RequestValidationTest(unittest.TestCase):
    def test_non_object_json_is_rejected_without_killing_controller(self):
        for payload in ('null','[]','42','"stop"','{broken'):
            fake=SimpleNamespace(last_result=None,request_ack=Mock())
            Core.request(fake,String(data=payload))
            self.assertFalse(fake.last_result['ok'])

    def test_stop_publishes_zero_without_waiting_for_tick_and_ack_is_not_mcu_feedback(self):
        fake=SimpleNamespace(last_result=None,request_ack=Mock(),pub=Mock(),emergency=Mock(),snapshot=Mock(return_value={}))
        Core.request(fake,String(data='{"op":"stop","id":"stop-check"}'))
        fake.emergency.assert_called_once();fake.pub.publish.assert_called_once()
        msg=fake.pub.publish.call_args.args[0]
        self.assertEqual([msg.linear.x,msg.linear.y,msg.angular.z],[0,0,0])
        ack=json.loads(fake.request_ack.publish.call_args.args[0].data)
        self.assertFalse(ack['mcu_acknowledged']);self.assertTrue(ack['ok'])
