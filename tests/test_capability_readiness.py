import json,tempfile,time,unittest
from pathlib import Path
from capability_readiness import CapabilityReadiness


class CapabilityReadinessTests(unittest.TestCase):
 def prepare(self,root):
  root=Path(root);(root/'data').mkdir();(root/'config').mkdir();now=time.time()
  (root/'data/status.json').write_text(json.dumps(dict(at=now,boot_id='b',stop_latched=True,velocity=[0,0,0],
      sensor_age={'odom':.1,'scan0':.1,'scan1':.1})))
  (root/'data/perception.json').write_text(json.dumps(dict(at=now,image_stamp=now)))
  (root/'data/power.json').write_text(json.dumps(dict(at=now,state='IDLE')))
  (root/'data/arm-state.json').write_text(json.dumps(dict(at=now,boot_id='b',phase='command_elapsed_observation_required')))
  (root/'config/handeye-accepted.json').write_text(json.dumps(dict(execution_authorized=True,physical_validation_record='record')))
  (root/'config/commissioning.json').write_text(json.dumps(dict(lidar_tf_validated=True,localization_verified=False)))
  (root/'data/navigation-health.json').write_text(json.dumps(dict(at=now,stage='active')))
  (root/'data/planning-health.json').write_text(json.dumps(dict(at=now,stage='active')))
  (root/'.venv-learning/bin').mkdir(parents=True);(root/'.venv-learning/bin/python').touch()
  return root
 def rows(self,root):return {row['id']:row for row in CapabilityReadiness(root).status()['capabilities']}
 def test_local_grasp_never_depends_on_localization_or_destination(self):
  with tempfile.TemporaryDirectory() as folder:
   root=self.prepare(folder);row=self.rows(root)['LEARN_GRASP_LOCAL']
   self.assertNotIn('localization',row['evidence_missing']);self.assertFalse(row['requires_localization'])
   self.assertFalse(row['requires_destination']);self.assertEqual(row['evidence_missing'],['local_grasp_profile'])
 def test_permission_profile_and_operate_acceptance_are_separate(self):
  with tempfile.TemporaryDirectory() as folder:
   root=self.prepare(folder);now=time.time()
   (root/'config/local-grasp-profile.json').write_text(json.dumps({'experimental_execution_authorized':True}))
   (root/'data/autonomy-permissions.json').write_text(json.dumps({'expires_at':now+100,'scopes':['arm_motion','target_contact']}))
   row=self.rows(root)['LEARN_GRASP_LOCAL'];self.assertTrue(row['experimental_ready']);self.assertIsNone(row['production_accepted'])
   self.assertIsNone(row['learned_policy_validated'])
   pick=self.rows(root)['PICK_LOCAL'];self.assertTrue(pick['capability_available']);self.assertFalse(pick['learned_policy_validated']);self.assertFalse(pick['production_accepted'])
 def test_missing_evidence_exposes_specific_workflow(self):
  with tempfile.TemporaryDirectory() as folder:
   row=self.rows(self.prepare(folder))['LEARN_PLACE_LOCAL']
   self.assertEqual([item['id'] for item in row['evidence_workflow']],['local_grasp_profile','local_reset_profile'])
 def test_persistent_day_profile_grants_only_listed_capability_scopes(self):
  with tempfile.TemporaryDirectory() as folder:
   root=self.prepare(folder);hour=time.localtime().tm_hour
   (root/'config/local-grasp-profile.json').write_text(json.dumps({'experimental_execution_authorized':True}))
   (root/'config/autonomous-day.json').write_text(json.dumps({'enabled':True,'start_hour':hour,
       'end_hour':min(24,hour+1),'allowed_capabilities':['LEARN_GRASP_LOCAL']}))
   row=self.rows(root)['LEARN_GRASP_LOCAL'];self.assertTrue(row['experimental_permission'])
   navigate=self.rows(root)['NAVIGATE'];self.assertFalse(navigate['experimental_permission'])


if __name__=='__main__':unittest.main()
