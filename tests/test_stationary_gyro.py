import unittest
from stationary_gyro import StationaryGyro

class StationaryUpdates(unittest.TestCase):
    def ready(self):
        f=StationaryGyro(0.)
        for t in [1.,1.5,2.1]:
            f.wheel([0.,0.,0.],t);f.command([0.,0.,0.],t)
            result=f.correct([0.,0.,.004],[0.,0.,9.81],t)
        self.assertEqual(result,(0.,True));return f

    def test_stale_evidence_never_zeroes_rate(self):
        f=self.ready();self.assertFalse(f.correct([0.,0.,.004],[0.,0.,9.81],2.5)[1])

    def test_command_revokes_before_encoders_respond(self):
        f=self.ready();f.command([.04,0.,0.],2.11)
        self.assertFalse(f.correct([0.,0.,.004],[0.,0.,9.81],2.12)[1])

    def test_encoder_motion_and_gyro_rotation_revoke(self):
        f=self.ready();f.wheel([0.,0.,.005],2.11)
        self.assertFalse(f.correct([0.,0.,.004],[0.,0.,9.81],2.12)[1])
        f=self.ready()
        self.assertFalse(f.correct([0.,0.,.10],[0.,0.,9.81],2.12)[1])

    def test_invalid_imu_not_published(self):
        f=self.ready();self.assertEqual(f.correct([0.,0.,float('nan')],[0.,0.,9.81],2.12),(None,False))
