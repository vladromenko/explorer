import unittest,copy
from missions import readiness,FLAGS
class MissionReadinessTest(unittest.TestCase):
    def ready(self):return dict(at=100,stop_latched=False,mode='AUTONOMOUS',commissioning=dict.fromkeys(FLAGS,True),sensor_age=dict.fromkeys(['imu','odom','scan0','scan1','battery'],0),battery=12,reason='COMMAND EXPIRED')
    def test_each_missing_prerequisite_blocks(self):
        s=self.ready();self.assertEqual(readiness(s,100),[])
        for f in FLAGS:
            q=copy.deepcopy(s);q['commissioning'][f]=False;self.assertIn(f,readiness(q,100))
    def test_manual_stop_stale_and_low_battery(self):
        for key,value in [('stop_latched',True),('mode','MANUAL'),('at',98),('battery',10.5),('battery',float('nan'))]:
            s=self.ready();s[key]=value;self.assertTrue(readiness(s,100))
    def test_stale_lidar_blocks(self):
        s=self.ready();s['sensor_age']['scan1']=.61;self.assertIn('scan1_stale',readiness(s,100))

    def test_invalid_sensor_age_blocks(self):
        for age in (-1,float('nan'),float('inf'),None):
            s=self.ready();s['sensor_age']['scan0']=age;self.assertIn('scan0_stale',readiness(s,100))
