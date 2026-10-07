import unittest
import numpy as np
from navigation_obstacles import swept_obstacle
POLYGON=[[-.18,-.16],[.29,-.16],[.29,.16],[-.18,.16]]

class SweptObstacleTests(unittest.TestCase):
    def points(self,point):return np.tile(point,(12,1))
    def test_side_returns_do_not_block_driving_away(self):
        self.assertFalse(swept_obstacle(self.points([0.,.27]),[.10,0.,0.],POLYGON))
        self.assertTrue(swept_obstacle(self.points([0.,.27]),[0.,.10,0.],POLYGON))
    def test_front_obstacle_and_rotation_envelope_are_checked(self):
        self.assertTrue(swept_obstacle(self.points([.37,0.]),[.1,0.,0.],POLYGON))
        self.assertFalse(swept_obstacle(self.points([.7,0.]),[.1,0.,0.],POLYGON))
        self.assertTrue(swept_obstacle(self.points([.27,.22]),[0.,0.,.25],POLYGON))
    def test_rigid_self_returns_and_missing_evidence(self):
        self.assertFalse(swept_obstacle(self.points([.1,.1]),[.1,0.,0.],POLYGON))
        self.assertTrue(swept_obstacle([], [.1,0.,0.],POLYGON))
