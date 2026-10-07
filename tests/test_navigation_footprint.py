import math
from types import SimpleNamespace
import unittest
import time
from navigation_footprint import matches,NavigationFootprint

class FootprintTests(unittest.TestCase):
    def test_costmap_restart_invalidates_old_footprint_echo(self):
        footprint=NavigationFootprint.__new__(NavigationFootprint)
        footprint.active=True;footprint.publishers={"local_costmap":None,"global_costmap":None}
        footprint.node=SimpleNamespace(count_publishers=lambda topic:1)
        footprint.observed={(name,b"single_publisher_only"):time.monotonic() for name in footprint.publishers}
        footprint.guard()
        footprint.observed[("local_costmap",b"single_publisher_only")]=time.monotonic()-3
        with self.assertRaisesRegex(ValueError,"expired"):footprint.guard()
    def test_rotation_and_translation_do_not_break_confirmation(self):
        polygon=[[-.18,-.16],[.29,-.16],[.29,.16],[-.18,.16]]
        padded=[[-.20,-.18],[.31,-.18],[.31,.18],[-.20,.18]]
        yaw=.9
        points=[SimpleNamespace(x=2+x*math.cos(yaw)-y*math.sin(yaw),y=-3+x*math.sin(yaw)+y*math.cos(yaw)) for x,y in padded]
        message=SimpleNamespace(polygon=SimpleNamespace(points=points))
        self.assertTrue(matches(message,polygon))
        footprint=NavigationFootprint.__new__(NavigationFootprint)
        footprint.active=True;footprint.profile={"polygon":polygon};footprint.observed={}
        footprint.receive("local_costmap",message,{"publisher_gid":b"publisher1"})
        self.assertEqual(len(footprint.observed),1)
        footprint.receive("local_costmap",message,{})
        self.assertIn(("local_costmap",b"single_publisher_only"),footprint.observed)
