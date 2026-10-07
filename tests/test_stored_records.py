import json
from pathlib import Path
import tempfile
import unittest
from mobile_demonstrations import MobileDemonstrations
from teaching import Demonstrations
from stored_records import records

class DamagedRecordTests(unittest.TestCase):
    def test_invalid_field_types_are_isolated_before_sorting(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"bad.json").write_text(json.dumps(dict(id="bad",state="running",started="yesterday")))
            (root/"good.json").write_text(json.dumps(dict(id="good",state="running",started=1)))
            rows,errors=records(root.glob("*.json"),("id","state","started"))
            self.assertEqual([row["id"] for _,row in rows],["good"])
            self.assertEqual(len(errors),1)
    def test_one_damaged_episode_does_not_disable_manual_ui(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"data/mobile-demonstrations/bad").mkdir(parents=True)
            path=root/"data/mobile-demonstrations/bad/episode.json";path.write_text("{")
            mobile=MobileDemonstrations(root)
            self.assertEqual(mobile.status()["successful"],0)
            self.assertEqual(len(mobile.status()["record_errors"]),1)
            self.assertTrue(path.exists())
    def test_arm_catalog_reports_bad_record_without_discarding_other_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"bad").mkdir();(root/"bad/episode.json").write_text("[]")
            demos=Demonstrations(root)
            self.assertEqual(demos.episodes(),[]);self.assertEqual(len(demos.record_errors),1)
