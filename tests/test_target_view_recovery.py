import unittest
from target_view_recovery import FLOOR_VIEWS, recover_view, visible_target


class RecoveryTests(unittest.TestCase):
    def test_observed_nested_toe_does_not_force_another_retreat(self):
        rows = [dict(label="black sock", bbox=[25.,183.75,488.4375,303.28125], confidence=.4728, position={"z":.32}),
                dict(label="black sock", bbox=[24.84375,183.515625,220.46875,302.578125], confidence=.4281, position={"z":.324})]
        result = visible_target(rows, 640, 480)
        self.assertTrue(result["found"])
        self.assertIs(result["target"], rows[0])
        self.assertFalse(result["physical_object_count_verified"])
        self.assertFalse(result["identity_verified"])
        self.assertEqual(len(rows), 2)

    def test_separate_or_clipped_or_unmeasured_targets_require_more_inspection(self):
        main = dict(label="sock", bbox=[30,100,400,300], confidence=.8, position={"z":.3})
        separate = dict(label="sock", bbox=[410,100,600,300], confidence=.9, position={"z":.3})
        nested = dict(main, bbox=[40,100,200,300])
        for rows in ([main,separate], [main,dict(nested,label="shoe")],
                     [dict(main,bbox=[0,100,400,300]),nested], [dict(main,position=None),nested]):
            self.assertFalse(visible_target(rows,640,480)["found"])

    def run_policy(self, observations, **kwargs):
        calls = []
        source = iter(observations)
        result = recover_view(lambda: next(source), lambda view: calls.append(("view", view)),
            lambda distance: calls.append(("retreat", distance)) or {"stopped": True},
            lambda: calls.append(("permit",)), lambda phase, record: calls.append(("record", phase)), **kwargs)
        return result, calls

    def test_visible_target_needs_no_retreat_or_other_views(self):
        result, calls = self.run_policy([{"found": True, "target": "fresh"}])
        self.assertEqual(result["target"], "fresh")
        self.assertFalse(result["contact_authorized"])
        self.assertEqual([call for call in calls if call[0] == "view"], [("view", "down")])
        self.assertFalse(any(call[0] == "retreat" for call in calls))

    def test_lost_target_is_refound_only_after_fresh_retreat_observation(self):
        observations = [{"found": False}] * len(FLOOR_VIEWS) + [{"found": True, "search_id": "new"}]
        result, calls = self.run_policy(observations)
        self.assertEqual(result["search_id"], "new")
        self.assertEqual(result["retreat_count"], 1)
        self.assertEqual([call for call in calls if call[0] == "retreat"], [("retreat", .06)])
        self.assertFalse(result["identity_verified"])

    def test_missing_target_has_finite_budget(self):
        result, calls = self.run_policy([{"found": False}] * (3 * len(FLOOR_VIEWS)))
        self.assertFalse(result["found"])
        self.assertEqual(sum(call[0] == "retreat" for call in calls), 2)

    def test_camera_sensor_and_cancellation_errors_never_trigger_blind_retreat(self):
        for failing in ("observe", "view", "permit"):
            retreats = []
            def fail(*args):
                raise ValueError("interrupted")
            with self.assertRaisesRegex(ValueError, "interrupted"):
                recover_view(fail if failing == "observe" else lambda: {"found": False},
                    fail if failing == "view" else lambda view: None,
                    lambda distance: retreats.append(distance),
                    fail if failing == "permit" else lambda: None, lambda *args: None)
            self.assertEqual(retreats, [])
