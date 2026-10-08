import math
import unittest
from pathlib import Path
import yaml
from navigation_scope import scope_blockers,bound_mapping_velocity,BASE_FLAGS

class NavigationScopeTests(unittest.TestCase):
    def test_live_mapping_does_not_claim_relocalization(self):
        flags=dict.fromkeys(BASE_FLAGS,True)
        health={name:dict(at=100,stage="active",inputs_ready=True) for name in ("navigation","planning")}
        self.assertEqual(scope_blockers(flags,"mapping",health,101),[])
        self.assertEqual(scope_blockers(flags,"localized",health,101),["localization_verified"])
        for component in health:
            health[component]["inputs_ready"]=False
            self.assertIn(component+"_runtime_unavailable",scope_blockers(flags,"mapping",health,101))
            health[component]["inputs_ready"]=True
        self.assertTrue(scope_blockers(flags,"mapping",health,104))
        self.assertTrue(scope_blockers(flags,"mapping",{"planning":[]},101))
    def test_norm_not_per_axis_limit(self):
        result=bound_mapping_velocity([1.,1.,2.])
        self.assertAlmostEqual(math.hypot(*result[:2]),.10)
        self.assertEqual(result[2],.25)
        self.assertEqual(bound_mapping_velocity([0.,0.,-.05]),[0.,0.,-.05])
        with self.assertRaises(ValueError):bound_mapping_velocity([math.nan,0.,0.])

    def test_nav2_velocity_model_matches_controller_mapping_limit(self):
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/"config/navigation.yaml").read_text())
        dwb=config["controller_server"]["ros__parameters"]["FollowPath"]
        self.assertLessEqual(dwb["max_speed_xy"],.10)
        self.assertLessEqual(dwb["max_vel_x"],.10)
        self.assertLessEqual(dwb["max_vel_y"],.10)
        self.assertGreaterEqual(dwb["min_vel_x"],-.10)
        self.assertGreaterEqual(dwb["min_vel_y"],-.10)
        self.assertLessEqual(dwb["max_vel_theta"],.25)
