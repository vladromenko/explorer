import unittest
import numpy as np
from click_navigation import floor_goal


class FloorNavigationTests(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[150.,0.,50.],[0.,150.,50.],[0.,0.,1.]])
        c=2.**-.5
        self.transform=np.array([[0.,-c,c,0.],[-1.,0.,0.,0.],[0.,-c,-c,.45],[0.,0.,0.,1.]])
        v,u=np.mgrid[:100,:100]
        rays=np.stack(((u-50)/150.,(v-50)/150.,np.ones_like(u)),axis=-1)
        self.floor=-.45/(rays@self.transform[2,:3])
        self.pose=dict(x=1.,y=2.,yaw=0.)

    def test_floor_point_has_measured_horizontal_patch(self):
        result=floor_goal(self.floor,self.k,np.zeros(5),50,50,self.transform,self.pose)
        self.assertTrue(result["floor_verified"])
        self.assertAlmostEqual(result["x"],1.45)
        self.assertAlmostEqual(result["floor_height_m"],0.)
        self.assertLess(result["floor_fit_p90_m"],1e-6)

    def test_table_and_wall_are_not_floor_goals(self):
        with self.assertRaisesRegex(ValueError,"выше пола"):
            floor_goal(self.floor*.5,self.k,np.zeros(5),50,50,self.transform,self.pose)
        with self.assertRaises(ValueError):
            floor_goal(np.full((100,100),.6364),self.k,np.zeros(5),50,50,self.transform,self.pose)

    def test_hole_and_nonrigid_mount_rejected(self):
        with self.assertRaisesRegex(ValueError,"глубины"):
            floor_goal(np.zeros((100,100)),self.k,np.zeros(5),50,50,self.transform,self.pose)
        wrong=self.transform.copy();wrong[0,0]=2.
        with self.assertRaisesRegex(ValueError,"привязка"):
            floor_goal(self.floor,self.k,np.zeros(5),50,50,wrong,self.pose)
