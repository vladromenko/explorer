import unittest
import numpy as np
from frontiers import candidates

class FrontiersTest(unittest.TestCase):
    def test_unknown_not_traversed(self):
        g=np.full((50,70),-1);g[5:45,5:45]=0
        p=candidates(g,.05,[0,0],[1,1])
        self.assertTrue(p)
        for t in p:
            self.assertEqual(g[int(t['y']/.05),int(t['x']/.05)],0)
            self.assertLess(t['x'],2.25-.35)
    def test_disconnected_space_not_selected(self):
        g=np.full((70,100),-1);g[5:65,5:40]=0;g[5:65,55:95]=0
        p=candidates(g,.05,[0,0],[1,1])
        self.assertTrue(p)
        self.assertTrue(all(t['x']<2 for t in p))
    def test_obstacle_at_robot_blocks(self):
        g=np.full((50,50),-1);g[5:45,5:45]=0;g[20,20]=100
        self.assertEqual(candidates(g,.05,[0,0],[1.025,1.025]),[])
    def test_no_unknown_not_room_completion(self):
        self.assertEqual(candidates(np.zeros((50,50)),.05,[0,0],[1,1]),[])

    def test_rotated_grid_origin_keeps_world_coordinates(self):
        import math
        g=np.full((50,70),-1);g[5:45,5:45]=0
        plain=candidates(g,.05,[0,0],[1.025,1.025])
        rotated=candidates(g,.05,[10,20],[8.975,21.025],origin_yaw=math.pi/2)
        self.assertEqual(len(plain),len(rotated))
        for a,b in zip(plain,rotated):
            self.assertAlmostEqual(b["x"],10-a["y"])
            self.assertAlmostEqual(b["y"],20+a["x"])
            self.assertAlmostEqual(b["yaw"],a["yaw"]+math.pi/2)

    def test_verified_camera_pose_envelope_can_leave_a_narrow_free_start(self):
        g=np.full((60,100),-1);g[10:50,10:90]=0;g[10:50,9]=100
        pose=[.725,1.525]
        self.assertEqual(candidates(g,.05,[0,0],pose),[])
        points=candidates(g,.05,[0,0],pose,footprint=[[-.18,-.16],[.29,-.16],[.29,.16],[-.18,.16]])
        self.assertTrue(points)
        self.assertTrue(all(g[int(p["y"]/.05),int(p["x"]/.05)]==0 for p in points))
