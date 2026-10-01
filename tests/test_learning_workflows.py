import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from episode_store import EpisodeStore
from learning_workflows import LearningWorkflows


class TypedEpisodeTests(unittest.TestCase):
    def test_all_sources_share_one_typed_format(self):
        with tempfile.TemporaryDirectory() as folder:
            store=EpisodeStore(folder)
            for source in ('HUMAN_DEMONSTRATION','HUMAN_INTERVENTION','AUTONOMOUS','EVALUATION'):
                item=store.begin('job','grasp sock','scene','policy','reward',source=source,
                                 task='grasp sock',skill='grasp',target='sock')
                store.append(item['id'],'observation',{'scene_version':'scene'})
                store.append(item['id'],'arm_command',{'servo_deg':[90]*6})
                done=store.finish(item['id'],'unknown',{'verifier':'visual'},reward=None)
                self.assertEqual(done['format'],'explorer_typed_episode_v1')
                self.assertEqual(done['source'],source)
                self.assertEqual(done['arm_commands'][0]['servo_deg'],[90]*6)

    def test_intervention_invalidates_action_and_requires_fresh_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            store=EpisodeStore(folder);item=store.begin('job','grasp','scene','p','r',source='HUMAN_INTERVENTION')
            row=store.intervention(item['id'],10,11,{'x':1},{'x':0})
            self.assertTrue(row['invalidated_autonomous_action']);self.assertTrue(row['fresh_observation_required'])


class WorkflowTests(unittest.TestCase):
    def legacy(self,root,outcome='success'):
        path=Path(root)/'data/demonstrations/demo/episode.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'id':'demo','name':'sock','created':time.time(),'ended':time.time(),
            'state':'complete','outcome':outcome,'source':'operator_demonstration','label_source':'operator',
            'joint_state_source':'command_estimate','steps':[{'action':{'arm_command':[90]*6,'gripper_command':90}}]}))
        return path

    def test_human_then_autonomous_has_no_fixed_demo_minimum(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=LearningWorkflows(folder,train_submit=lambda workflow:{'queued':True,'job':'train'})
            job=manager.start('BOOTSTRAP_THEN_AUTONOMOUS','grasp','sock',1,2)
            self.assertEqual(job['state'],'waiting_demo')
            updated=manager.attach_demonstration(job['id'],self.legacy(folder))
            self.assertEqual(updated['state'],'queued_autonomous')
            episode=manager.begin_autonomous_episode(job['id'],'scene')
            self.assertEqual(episode['source'],'AUTONOMOUS')

    def test_intervention_forces_reobserve_before_continuation(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=LearningWorkflows(folder);job=manager.start('HUMAN_INTERVENTION','grasp','sock',0,2)
            manager.begin_autonomous_episode(job['id'],'scene')
            state=manager.intervention(job['id'],{'dx':.1},{'dx':0},1,2)['workflow']
            self.assertEqual(state['state'],'reobserve_and_replan')
            with self.assertRaises(ValueError):manager.finish_autonomous_episode(job['id'],'failure',{})
            state=manager.mark_replanned(job['id'],{'scene_version':'scene-2'})
            self.assertEqual(state['state'],'autonomous')

    def test_needs_reset_resumes_same_series_after_visual_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=LearningWorkflows(folder);job=manager.start('AUTONOMOUS','grasp','sock',0,2)
            manager.needs_reset(job['id']);state=manager.reset_observed(job['id'],{'target_visible':True,'fresh':True})
            self.assertEqual(state['state'],'queued_autonomous')

    def test_human_only_waits_for_real_training_completion(self):
        with tempfile.TemporaryDirectory() as folder,patch('learning_workflows.time.sleep',return_value=None):
            states=iter(({'state':'training'},{'state':'validated_offline','validation':{'mae':1.}}))
            manager=LearningWorkflows(folder,train_submit=lambda workflow:{'queued':True,'job':'train'},
                                      train_status=lambda job:next(states))
            job=manager.start('HUMAN_DEMONSTRATION','grasp','sock',1,0)
            manager.attach_demonstration(job['id'],self.legacy(folder))
            deadline=time.time()+1
            while manager.get(job['id'])['state']=='training' and time.time()<deadline:time.sleep(.01)
            done=manager.get(job['id'])
            self.assertEqual(done['state'],'complete');self.assertEqual(done['training_state'],'validated_offline')

    def test_bootstrap_waits_then_queues_autonomy(self):
        with tempfile.TemporaryDirectory() as folder,patch('learning_workflows.time.sleep',return_value=None):
            manager=LearningWorkflows(folder,train_submit=lambda workflow:{'queued':True,'job':'train'},
                                      train_status=lambda job:{'state':'trained_unvalidated'})
            job=manager.start('BOOTSTRAP_THEN_AUTONOMOUS','grasp','sock',1,2)
            manager.attach_demonstration(job['id'],self.legacy(folder))
            deadline=time.time()+1
            while manager.get(job['id'])['state']=='training' and time.time()<deadline:time.sleep(.01)
            self.assertEqual(manager.get(job['id'])['state'],'queued_autonomous')


if __name__=='__main__':unittest.main()
