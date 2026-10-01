import time
import unittest
from autonomous_day import AutonomousDayPlanner


class DayTests(unittest.TestCase):
    def test_emergency_stop_and_missing_pose_take_priority_over_exploration(self):
        planner=AutonomousDayPlanner();permission={'expires_at':time.time()+100,'base_motion':True,
            'allowed_capabilities':['EXPLORE_LOCAL']}
        self.assertEqual(planner.choose({'emergency_stop':True},permission,{}, {'disk_free_gb':10})['activity'],'pause')
        self.assertEqual(planner.choose({'localization_valid':False},permission,{}, {'disk_free_gb':10})['activity'],'relocalize')
        self.assertEqual(planner.choose({'localization_valid':True,'unknown_object_query':'sock'},permission,{}, {'disk_free_gb':10})['activity'],'bounded_exploration')


if __name__=='__main__':unittest.main()
