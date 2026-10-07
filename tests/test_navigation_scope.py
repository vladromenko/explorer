import math
import unittest
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
