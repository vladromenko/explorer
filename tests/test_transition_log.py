import queue
import unittest
from transition_log import TransitionLog

class TransitionLogTests(unittest.TestCase):
    def test_event_preserves_original_sensor_cause(self):
        log=TransitionLog.__new__(TransitionLog);log.queue=queue.Queue(maxsize=1);log.last=None;log.dropped=0
        scans={'scan0':{'nearest':.07}}
        log.emit('lidar','OBSTACLE',{}, {},obstacle=scans)
        scans['scan0']={'nearest':.7}
        self.assertEqual(log.last['obstacle']['scan0']['nearest'],.07)
        log.emit('lidar','ACTIVE',{}, {},obstacle=scans)
        self.assertEqual(log.dropped,1)
        self.assertEqual(log.queue.get_nowait()['obstacle']['scan0']['nearest'],.07)
