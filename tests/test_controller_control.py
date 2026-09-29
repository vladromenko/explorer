import unittest
from controller_control import BaseTransport

class BaseTransportTests(unittest.TestCase):
    def setUp(self):
        self.requests=[]
        self.control=BaseTransport(self.requests.append)
        self.control.observe(dict(identity=dict(boot=1),session_state='active',
            telemetry_fresh=True,telemetry_only=False),1_000_000_000)

    def test_source_expiry_is_preserved_and_late_command_holds(self):
        self.control.velocity([.1,.1,.2],1_000_000_000,1_100_000_000)
        self.assertEqual(self.requests[-1]['expires_monotonic_ns'],1_200_000_000)
        self.control.velocity([.1,.1,.2],1_000_000_000,1_250_000_000)
        self.assertEqual(self.requests[-1]['operation'],'HOLD')

    def test_recovery_is_explicit_and_reboot_does_not_resume(self):
        self.control.state['session_state']='fault'
        self.control.start(1_000_000_000)
        self.assertEqual(self.requests[-1]['operation'],'CLEAR')
        self.control.observe(dict(identity=dict(boot=2),session_state='disarmed'),1_050_000_000)
        self.assertEqual(len(self.requests),1)
        self.control.velocity([.1,0,0],1_050_000_000,1_050_000_000)
        self.assertEqual(len(self.requests),1)

    def test_readonly_acceptance_cannot_be_bypassed(self):
        self.control.state['telemetry_only']=True
        with self.assertRaises(ValueError):self.control.start(1_000_000_000)
        self.control.stop(1_000_000_000)
        self.assertEqual(self.requests[-1]['operation'],'ESTOP')
