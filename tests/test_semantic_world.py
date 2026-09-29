import tempfile
from pathlib import Path
import unittest
from semantic_world import SemanticWorld,active_view,guarded_closure


class SemanticWorldTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.world=SemanticWorld(Path(self.temp.name));self.pose=dict(x=1.,y=2.,yaw=.1)

    def test_unvalidated_camera_point_is_episode_not_fake_map_entity(self):
        p=dict(image_stamp=10.,objects=[dict(label='sock',confidence=.9,position={'x':0,'y':0,'z':.4})])
        result=self.world.ingest(p,self.pose,'map-a')
        self.assertEqual(result['fused_entities'],0)
        q=self.world.query('sock');self.assertEqual(q['entities'],[])
        self.assertEqual(q['last_seen']['observer_pose'],self.pose)

    def test_unchanged_unvalidated_view_is_not_written_each_frame(self):
        first=dict(image_stamp=10.,objects=[dict(label='chair',confidence=.9)])
        second=dict(image_stamp=11.,objects=[dict(label='chair',confidence=.91)])
        self.assertTrue(self.world.ingest(first,self.pose,'map-a')['added'])
        self.assertFalse(self.world.ingest(second,self.pose,'map-a')['added'])
        self.assertEqual(self.world.status()['episodes'],1)

    def test_validated_map_points_update_entity_and_record_movement(self):
        def frame(stamp,x):return dict(image_stamp=stamp,objects=[dict(label='sock',confidence=.9,
            map_position={'x':x,'y':2.,'z':.1},map_position_validated=True)])
        self.world.ingest(frame(10.,1.),self.pose,'map-a')
        self.world.ingest(frame(11.,1.12),self.pose,'map-a')
        q=self.world.query('sock');self.assertEqual(len(q['entities']),1)
        self.assertAlmostEqual(q['entities'][0]['position']['x'],1.12)
        with self.world.db() as db:
            operations=[r[0] for r in db.execute('SELECT operation FROM entity_events ORDER BY id')]
        self.assertEqual(operations,['appeared','moved'])

    def test_place_recognition_reuses_nearby_semantic_signature(self):
        p=dict(image_stamp=10.,objects=[dict(label='chair',confidence=.9)])
        first=self.world.ingest(p,self.pose,'map-a')['place']
        self.world.last_frame=None;p['image_stamp']=50.
        second=self.world.ingest(p,dict(x=1.1,y=2.1,yaw=.12),'map-a')['place']
        self.assertEqual(first,second);self.assertEqual(self.world.status()['places'],1)

    def test_provisional_pose_is_saved_as_viewpoint_but_not_a_place(self):
        p=dict(image_stamp=10.,objects=[dict(label='chair',confidence=.9)])
        result=self.world.ingest(p,dict(self.pose,provisional=True),'map-a')
        self.assertIsNone(result['place']);self.assertEqual(self.world.status()['places'],0)
        self.assertTrue(self.world.query('chair')['last_seen']['observer_pose']['provisional'])

    def test_missing_requires_three_validated_visibility_checks(self):
        p=dict(image_stamp=10.,objects=[dict(label='sock',confidence=.9,map_position={'x':1.,'y':2.,'z':.1},map_position_validated=True)])
        self.world.ingest(p,self.pose,'map-a');ident=self.world.query('sock')['entities'][0]['id']
        for stamp in (11.,12.,13.):
            self.world.ingest(dict(image_stamp=stamp,objects=[],visibility_volume_validated=True,
                                   observable_entity_ids=[ident]),self.pose,'map-a')
        self.assertEqual(self.world.query('sock')['entities'][0]['state'],'missing')

    def test_action_outcomes_are_continual_memory(self):
        self.world.record_action('deliver sock','grasp','failed',{'reason':'slip'})
        self.world.record_action('deliver sock','regrasp','succeeded',{'verified':True})
        self.assertEqual(self.world.status()['actions'],2)

    def test_active_view_keeps_execution_gated(self):
        p=dict(objects=[dict(label='sock',confidence=.4,position=None)])
        r=active_view(p,{},{});self.assertEqual(r['kind'],'inspect_target');self.assertFalse(r['physical_allowed'])

    def test_guarded_closure_uses_contact_deformation_and_slip(self):
        base=dict(at=1.,association_fraction=.9,object_scale=1.,object_tcp_distance_m=.04)
        contact=dict(at=1.1,association_fraction=.9,object_scale=1.01,object_tcp_distance_m=.02)
        self.assertEqual(guarded_closure([base,contact])['action'],'hold')
        slipped=dict(contact,at=1.2,object_tcp_distance_m=.04)
        self.assertEqual(guarded_closure([contact,slipped])['action'],'tighten')
        deformed=dict(contact,at=1.2,object_scale=1.12,object_tcp_distance_m=.02)
        self.assertEqual(guarded_closure([contact,deformed])['reason'],'soft_object_deformation_limit')


if __name__=='__main__':unittest.main()
