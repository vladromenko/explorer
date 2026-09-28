import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from observed_base import ObservedBase,arm_command_pending

class ObservedBaseTests(unittest.TestCase):
    def test_arm_completion_uses_newer_same_boot_executor_record(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data').mkdir();path=root/'data/arm-state.json'
            s=dict(boot_id='boot',arm_command_state=dict(phase='command_in_progress',at=10))
            self.assertTrue(arm_command_pending(root,s))
            current=dict(boot_id='boot',phase='command_elapsed_observation_required',at=10,ends_monotonic=time.monotonic()-.3)
            path.write_text(json.dumps(current));self.assertFalse(arm_command_pending(root,s))
            for changes in (dict(boot_id='old'),dict(at=9),dict(phase='command_in_progress'),dict(ends_monotonic=time.monotonic()+1)):
                path.write_text(json.dumps(dict(current,**changes)))
                self.assertTrue(arm_command_pending(root,s))

    def test_stopped_obstacle_is_interrupted_not_completed(self):
        b=ObservedBase('/tmp/unused')
        for completed,phase in [(False,'interrupted'),(True,'completed')]:
            b.lock.acquire()
            report=dict(final_velocity=dict(vx=0,vy=0,wz=0),execution_outcome=dict(
                completed=completed,reason='PROBE COMPLETE' if completed else 'OBSTACLE'))
            with patch('observed_base.subprocess.run',return_value=SimpleNamespace(
                    returncode=0,stdout=json.dumps(report),stderr='')):
                b.run('right',2)
            self.assertEqual(b.status()['phase'],phase)
            self.assertFalse(b.status()['busy'])

    def test_never_clears_stop_and_rejects_autonomy_or_unobserved_motion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data').mkdir();p=root/'data/status.json';b=ObservedBase(root)
            base=dict(at=time.time(),stop_latched=False,mode='MANUAL',mission=None)
            with patch('observed_base.threading.Thread') as thread:
                for change,observer in [({'stop_latched':True},True),({'mode':'AUTONOMOUS'},True),({'mission':'other'},True),({},False)]:
                    p.write_text(json.dumps(dict(base,**change)))
                    with self.assertRaises(ValueError):b.start('forward',1,observer,True)
                thread.assert_not_called()
                p.write_text(json.dumps(base));r=b.start('left',1,True,True)
                self.assertTrue(r['busy']);thread.return_value.start.assert_called_once()
                with self.assertRaises(ValueError):b.start('right',1,True,True)
