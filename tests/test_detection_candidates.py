import unittest
from detection_candidates import unique_candidates


class CandidateTests(unittest.TestCase):
    def test_nested_duplicate_from_observed_sock_frame_is_not_two_targets(self):
        rows = [dict(label="sock", confidence=.3844, bbox=[253.8,142.9,440,373.7]),
            dict(label="sock", confidence=.3819, bbox=[328.9,143,439.7,359.5])]
        self.assertEqual(unique_candidates(rows), rows[:1])
        self.assertEqual(len(rows), 2)

    def test_distinct_targets_and_different_labels_are_not_silently_selected(self):
        a = dict(label="sock", confidence=.9, bbox=[0,0,20,20])
        b = dict(label="sock", confidence=.8, bbox=[25,0,45,20])
        c = dict(label="shoe", confidence=.7, bbox=[0,0,20,20])
        self.assertEqual(unique_candidates([a,b,c]), [a,b,c])

    def test_invalid_box_does_not_become_a_grasp_target(self):
        for box in ([0,0,0,10], [0,0,float("nan"),10], [10,0,0,10]):
            with self.assertRaises(ValueError):
                unique_candidates([dict(label="sock", confidence=.8, bbox=box)])


if __name__ == "__main__":
    unittest.main()
