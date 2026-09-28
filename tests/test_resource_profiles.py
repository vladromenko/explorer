import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from resource_profiles import ResourceProfiles

class ProfileTests(unittest.TestCase):
    def test_busy_jobs_prevent_mode_switch_and_conflicting_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            profiles=ResourceProfiles(folder,lambda:['train'])
            with self.assertRaises(ValueError):profiles.select('experiment')
            with self.assertRaises(ValueError):profiles.admit('llm',lambda:None)

    def test_training_profile_requires_stop_and_preserves_control_services(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data';path.mkdir()
            state=dict(at=time.time(),stop_latched=False,velocity=[0,0,0])
            (path/'status.json').write_text(json.dumps(state));profiles=ResourceProfiles(folder,lambda:[])
            with self.assertRaises(ValueError):profiles.select('training')
            state['stop_latched']=True;(path/'status.json').write_text(json.dumps(state))
            with patch('resource_profiles.subprocess.run') as run:
                self.assertEqual(profiles.select('training')['mode'],'training')
                args=run.call_args.args[0]
                self.assertEqual(args[-2:],['explorer-llm.service','explorer-speech.service'])
            with self.assertRaises(ValueError):profiles.admit('grounding',lambda:None)
