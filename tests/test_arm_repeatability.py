import unittest
from arm_repeatability import plan
from arm_commissioning import HOME,HARD_LIMITS


class RepeatabilityTests(unittest.TestCase):
    def test_returns_home_and_compares_identical_commands(self):
        steps=plan(HOME);previous=list(HOME);groups={}
        for step in steps:
            pose=step['pose']
            self.assertTrue(all(lo<=v<=hi for v,(lo,hi) in zip(pose,HARD_LIMITS)))
            if 'delta' in step:
                self.assertEqual(sum(a!=b for a,b in zip(previous,pose)),1)
                self.assertEqual(abs(step['delta']),2)
            if 'capture' in step:groups.setdefault(step['capture'].split('_')[0],[]).append(pose)
            previous=pose
        self.assertEqual(previous,HOME)
        self.assertEqual(len(groups),4)
        for poses in groups.values():self.assertTrue(all(p==poses[0] for p in poses))

    def test_unobservable_or_ambiguous_joints_are_rejected(self):
        for joints in ([],[1,1],[5],[6]):
            with self.assertRaises(ValueError):plan(HOME,joints)

    def test_both_joint_limits_have_valid_return_paths(self):
        for value in (0,180):
            start=[value]*4+[90,30]
            self.assertEqual(plan(start)[-1]['pose'],start)

    def test_consistent_approach_preserves_limits_and_two_degree_steps(self):
        for distance in (2,4):
            self.check_approach(distance)

    def check_approach(self,distance):
        previous=list(HOME)
        for step in plan(HOME,approach_from_above=True,approach_degrees=distance):
            if 'capture' in step:self.assertEqual(step['pose'],previous)
            else:
                self.assertEqual(sum(a!=b for a,b in zip(previous,step['pose'])),1)
                self.assertEqual(max(abs(a-b) for a,b in zip(previous,step['pose'])),2)
            self.assertTrue(all(lo<=v<=hi for v,(lo,hi) in zip(step['pose'],HARD_LIMITS)))
            previous=step['pose']
        self.assertEqual(previous,HOME)

    def test_approach_distance_cannot_exceed_joint_limit(self):
        with self.assertRaises(ValueError):plan([178,125,3,0,90,30],[1],True,4)
        with self.assertRaises(ValueError):plan(HOME,[1],True,6)
