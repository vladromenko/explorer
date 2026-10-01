import unittest
from scene_preparation import propose,verify


def row(at,point,identity=True,collateral=0):
    return dict(at=at,object_id='sock',object_xyz=point,identity_association_verified=identity,
                depth_validated=True,collateral_displacement_m=collateral)


class PreparationTests(unittest.TestCase):
    def test_proposal_requires_scoped_contact_and_rejects_heavy_object(self):
        item=dict(id='sock',kind='soft_cloth',mass_g=30,center_xyz=[.3,0,.02],extent_xyz=[.08,.05,.02])
        with self.assertRaises(ValueError):propose(item,{},[],{})
        self.assertTrue(propose(item,{},[],{'target_contact':True}))
        item['mass_g']=500
        with self.assertRaises(ValueError):propose(item,{},[],{'target_contact':True})

    def test_verifier_rewards_direction_not_contact_itself(self):
        proposal=dict(start_xyz=[.3,0,.02],goal_xyz=[.35,0,.02],maximum_travel_m=.05,permitted_contact_object='sock')
        before=[row(t,[.3,0,.02]) for t in (0,.5,1)]
        after=[row(t,[.34,0,.02]) for t in (2,2.5,3)]
        self.assertEqual(verify(before,after,proposal)['state'],'success')
        still=[row(t,[.3,0,.02]) for t in (2,2.5,3)]
        self.assertEqual(verify(before,still,proposal)['state'],'failure')
        collateral=[row(t,[.34,0,.02],collateral=.05) for t in (2,2.5,3)]
        self.assertEqual(verify(before,collateral,proposal)['state'],'failure')


if __name__=='__main__':unittest.main()
