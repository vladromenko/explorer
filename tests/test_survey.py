import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock
from survey import SurveyStore,summarize

class SurveyTests(unittest.TestCase):
    def test_observations_keep_robot_and_camera_locations_distinct(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'data').mkdir();now=time.time()
            perception=dict(image_stamp=now,objects=[dict(label='bottle',confidence=.9,position={'x':1,'y':2,'z':3,'frame':'camera'})])
            (p/'data/perception.json').write_text(json.dumps(perception))
            (p/'data/status.json').write_text(json.dumps(dict(at=now,lidar={'scan0':{'nearest':1}},battery=12)))
            store=SurveyStore(p);r=store.capture('m','desk',{'x':4,'y':5},'map1',now-.1,Mock())
            self.assertFalse(r['object_map_positions_verified'])
            self.assertEqual(r['perception']['objects'][0]['position']['frame'],'camera')
            self.assertEqual(r['robot_pose'],{'x':4,'y':5});self.assertEqual(len(store.recent()),1)
    def test_cancel_before_capture_saves_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'data').mkdir();store=SurveyStore(p)
            with self.assertRaises(ValueError):store.capture('m','desk',{},'map1',0,Mock(side_effect=ValueError('cancel')))
            self.assertEqual(store.recent(),[])
    def test_low_confidence_not_spoken_as_identity(self):
        s=summarize({'objects':[{'label':'elephant','confidence':.4}]},'door')
        self.assertNotIn('elephant',s);self.assertIn('предполагает',s)
