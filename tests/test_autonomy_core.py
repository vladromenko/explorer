import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from autonomy_contracts import GoalSpec, Outcome, PolicyVersion, Truth
from episode_store import EpisodeStore
from learning_stack import CandidateScorer, OutcomeModel, InterventionActorLearner, PolicyRegistry
from skill_discovery import segment, replay_anchors
from skill_planner import SkillRegistry, TaskPlanner, recovery_plan
from autonomy_runtime import AutonomySupervisor


class ContractTests(unittest.TestCase):
    def test_goal_outcome_and_policy_reject_ambiguous_contracts(self):
        GoalSpec('g','place sock',{'placed':True}).validate()
        Outcome('unknown','occluded','rgbd','2',()).validate()
        with self.assertRaises(ValueError):GoalSpec('g','bad',{'placed':False}).validate()
        with self.assertRaises(ValueError):PolicyVersion('p','x','v','accepted','c',('same',),('same',),{}).validate()


class PlannerTests(unittest.TestCase):
    def test_plan_changes_with_state_and_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            planner=TaskPlanner(SkillRegistry(folder))
            state={'scene_observed':'true','object_localized':'true','localization_valid':'true',
                   'destination_localized':'true','base_aligned':'true','grasp_reachable':'true',
                   'gripper_aligned':'true','grasp_attempted':'true','object_held':'true',
                   'at_destination':'true','object_supported':'true','release_attempted':'true','placed':'unknown'}
            plan=planner.plan(state,{'placed':True},'scene-a',permissions=('base_motion','arm_motion'))
            self.assertEqual([row['skill'] for row in plan['steps']],['verify_place'])
            state['object_held']='false';state['at_destination']='false';state['object_supported']='false';state['release_attempted']='false'
            changed=planner.plan(state,{'placed':True},'scene-b',permissions=('base_motion','arm_motion'))
            self.assertNotEqual(plan['steps'],changed['steps'])
            self.assertEqual(recovery_plan('track_lost',planner.registry)[0]['id'],'change_view')

    def test_verifier_cannot_probe_its_own_unknown_precondition(self):
        with tempfile.TemporaryDirectory() as folder:
            planner=TaskPlanner(SkillRegistry(folder))
            plan=planner.plan({'placed':'unknown','release_attempted':'unknown'}, {'placed':True}, 'scene',
                              allowed_skills=['verify_place','reset_scene'],permissions=())
            self.assertTrue(plan['blocked']);self.assertEqual(plan['steps'],[])


class LearningTests(unittest.TestCase):
    @staticmethod
    def row(index,success):
        context=dict(object_width_m=.03,object_height_m=.02,distance_m=.25,visibility=.9,
                     depth_uncertainty_m=.003,target_distance_m=.5,softness=1.,scene_clutter=.1,previous_failures=0.)
        action=dict(approach_x=.1 if success else -.1,approach_y=0.,approach_z=.03,aperture_m=.04,
                    roll_rad=0.,base_shift_m=0.,clearance_m=.03 if success else .001,trajectory_cost=1. if success else 9.)
        return dict(episode_id='scene-'+str(index//5),context=context,action=action,outcome='success' if success else 'failure')

    def test_scorer_updates_checkpoint_and_changes_active_version(self):
        with tempfile.TemporaryDirectory() as folder:
            scorer=CandidateScorer(folder);samples=[self.row(i,i%2==0) for i in range(40)]
            checkpoint=scorer.train(samples,epochs=30)
            self.assertNotEqual(checkpoint['id'],'geometry-baseline')
            promoted=scorer.promote(checkpoint,minimum_margin=-1)
            self.assertTrue(promoted['accepted']);self.assertEqual(scorer.version,checkpoint['id'])

    def test_outcome_model_and_actor_have_real_parameter_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            transitions=[]
            for index in range(36):
                before=np.full(6,index/100);action=np.full(6,.01);delta=.5*action
                transitions.append(dict(state_before=before,action=action,observed_delta=delta,
                    observation_source='rgbd',confidence=.95))
            model=OutcomeModel(folder);checkpoint=model.train(transitions)
            self.assertGreater(checkpoint['samples'],10);self.assertEqual(len(model.predict([0]*6,[.01]*6)),6)
            rows=[]
            for index in range(40):
                obs=np.full(12,index/100);proposed=np.full(3,.1);executed=proposed if index%3 else np.zeros(3)
                rows.append(dict(observation=obs,proposed_action=proposed,executed_action=executed,
                    next_observation=obs+.01,reward=1. if index%2 else 0.,terminated=index%7==0,truncated=False,intervention=index%3==0))
            rl=InterventionActorLearner(folder);result=rl.update(rows,epochs=10)
            self.assertGreater(np.linalg.norm(np.asarray(result['actor'])),0)


class EpisodeAndDiscoveryTests(unittest.TestCase):
    def test_episode_is_transactional_and_restart_does_not_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            store=EpisodeStore(folder);item=store.begin('job','goal','scene','policy','reward')
            self.assertEqual(store.recover(),[item['id']]);self.assertEqual(store.read(item['id'])['state'],'interrupted')

    def test_segmentation_and_episode_level_anchors(self):
        events=[dict(at=1,event='approach_start',goal='sock',frame='map'),dict(at=2,event='gripper_close'),
                dict(at=3,event='lift_verified',outcome='success')]
        self.assertEqual([row['kind'] for row in segment(events)],['approach_start','gripper_close','lift_verified'])
        episodes=[dict(id='e'+str(i),task='a' if i<5 else 'b',outcome='success') for i in range(10)]
        split=replay_anchors(episodes,'b')
        self.assertFalse(set(split['training_episode_ids'])&set(split['validation_episode_ids']))


class SupervisorTests(unittest.TestCase):
    def test_idempotent_job_persists_without_asking_for_physical_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            world=lambda:dict(scene_version='scene-1',map_epoch='map-1',predicates={
                'scene_observed':'true','object_localized':'true','localization_valid':'true',
                'destination_localized':'true','base_aligned':'true','grasp_reachable':'true',
                'gripper_aligned':'true','grasp_attempted':'true','object_held':'true',
                'at_destination':'true','object_supported':'true','release_attempted':'true','placed':'unknown'})
            readiness=lambda:dict(hardware=dict(physical_execution_ready=False,blocked_by=['accept gripper retention']),
                                  permissions=dict(base_motion=True,arm_motion=True))
            supervisor=AutonomySupervisor(folder,world,readiness,lambda *args:dict(state='success'))
            spec=dict(goal='place sock',attempts=1,time_budget_s=60,target_predicates={'placed':True},
                      permissions=['base_motion','arm_motion'],object_query='sock',destination={'name':'basket'},task_type='delivery')
            first=supervisor.submit('stable-request-1234',spec);second=supervisor.submit('stable-request-1234',spec)
            self.assertEqual(first['id'],second['id'])
            deadline=time.time()+2;state=supervisor.get(first['id'])
            while state['state']!='blocked' and time.time()<deadline:
                time.sleep(.02);state=supervisor.get(first['id'])
            self.assertEqual(state['state'],'blocked');self.assertEqual(state['help'],[])
            self.assertIn('accept gripper retention',state['result']['blocked_by'])
            self.assertEqual(supervisor.cancel(first['id'])['cancelled'],[first['id']])
            self.assertEqual(supervisor.get(first['id'])['state'],'cancelled')
            supervisor.shutdown()

    def test_missing_goal_fields_produce_one_actionable_help_request(self):
        with tempfile.TemporaryDirectory() as folder:
            world=lambda:dict(scene_version='scene-1',predicates={
                'scene_observed':'true','object_localized':'true','localization_valid':'true',
                'destination_localized':'true','base_aligned':'true','grasp_reachable':'true',
                'gripper_aligned':'true','grasp_attempted':'true','object_held':'true',
                'at_destination':'true','object_supported':'true','release_attempted':'true','placed':'unknown'})
            readiness=lambda:dict(hardware=dict(physical_execution_ready=True),permissions={})
            supervisor=AutonomySupervisor(folder,world,readiness,lambda *args:dict(state='unknown'))
            spec=dict(goal='deliver something',attempts=1,time_budget_s=60,target_predicates={'placed':True},
                      permissions=[],task_type='delivery')
            item=supervisor.submit('missing-fields-1234',spec);deadline=time.time()+2;state=supervisor.get(item['id'])
            while state['state']!='blocked' and time.time()<deadline:
                time.sleep(.02);state=supervisor.get(item['id'])
            self.assertEqual(state['state'],'blocked');self.assertEqual(len(state['help']),1)
            self.assertIn('какой предмет',state['help'][0]['request']['question'])
            supervisor.shutdown()


if __name__=='__main__':unittest.main()
