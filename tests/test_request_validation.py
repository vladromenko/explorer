import unittest
from types import SimpleNamespace
from std_msgs.msg import String
from core import Core
class RequestValidationTest(unittest.TestCase):
    def test_non_object_json_is_rejected_without_killing_controller(self):
        for payload in ('null','[]','42','"stop"','{broken'):
            fake=SimpleNamespace(last_result=None)
            Core.request(fake,String(data=payload))
            self.assertFalse(fake.last_result['ok'])
