import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from experiments import Experiments,sequence_plan
from experiment_memory import ExperienceMemory
from arm_feedback import decode_position,describe
from telegram_pairing import Pairing


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'data').mkdir();(self.root/'config').mkdir()
        self.now=time.time()
        (self.root/'data/status.json').write_text(json.dumps(dict(at=self.now,stop_latched=True,velocity=[0,0,0],commissioning={},arm_command_state={})))
        (self.root/'data/perception.json').write_text(json.dumps(dict(at=self.now,image_stamp=self.now,objects=[dict(label='bottle',bbox=[10,20,30,40],confidence=.9)])))
        (self.root/'config/telegram.json').write_text(json.dumps(dict(allowed_user_ids=[],allowed_chat_ids=[])))
        self.lab=Experiments(self.root)

    def tearDown(self):self.temp.cleanup()

    def run_one(self,ident='E01',mode='observe',params=None,key='request-1234567890'):
        return self.lab.start(ident,mode,params or {},key)

    def test_catalog_and_motor_modes_are_denied(self):
        self.assertEqual(len(self.lab.catalog()['experiments']),18)
        for mode in ('physical','autonomous','observed_physical'):
            with self.assertRaises(ValueError):self.run_one(mode=mode)

    def test_idempotence_and_parameters_cannot_change(self):
        a=self.run_one();b=self.run_one();self.assertEqual(a['id'],b['id'])
        with self.assertRaises(ValueError):self.run_one(params={'query':'changed'})

    def test_replay_uses_same_input_and_survives_restart(self):
        a=self.run_one('E02');self.lab=Experiments(self.root)
        b=self.run_one('E02','replay',{'run_id':a['id']},'request-other-12345')
        self.assertEqual(a['input_sha256'],b['input_sha256'])
        self.assertEqual(a['result']['objects'],b['result']['objects'])
        self.assertFalse(b['motor_access'])

    def test_stale_sensor_data_is_not_a_detection(self):
        (self.root/'data/perception.json').write_text(json.dumps(dict(at=0,objects=[{'label':'bottle'}])))
        self.assertEqual(self.run_one('E02')['result']['objects'],[])

    def test_replay_curriculum_does_not_read_later_operator_labels(self):
        self.lab.memory.label('grasp','cloth','failure','empty gripper')
        before=self.run_one('E18')
        self.lab.memory.label('grasp','cloth','success','operator observed hold')
        after=self.run_one('E18','replay',{'run_id':before['id']},'replay-curriculum-001')
        self.assertEqual(before['result'],after['result'])

    def test_cancel_invalidates_result(self):
        original=self.lab.evaluate
        def slow(*args):
            self.lab.cancel();return original(*args)
        with patch.object(self.lab,'evaluate',side_effect=slow):
            self.assertEqual(self.run_one()['state'],'cancelled')

    def test_all_analysis_handlers_produce_persisted_result(self):
        for i in range(1,19):
            params={7:{"events":[]},8:{"before":{"x":0,"y":0,"yaw":0},"after":{"x":0,"y":0,"yaw":0},"duration_s":1,"velocity":[0,0,0]},12:{"samples":[{}]},13:{"before":[{}]*3,"after":[{}]*3}}.get(i,{})
            r=self.run_one(f'E{i:02}',params=params,key=f'run-all-{i:02}-123456789')
            self.assertEqual(r['state'],'completed',(i,r))
            self.assertIn('summary',r['result'])
            self.assertTrue((self.lab.folder/(r['id']+'.json')).exists())

    def test_required_inputs_do_not_create_empty_successful_runs(self):
        for ident in ("E07","E08","E12","E13"):
            self.assertFalse(self.lab.preflight(ident,"observe",{})["ready"])
            with self.assertRaises(ValueError):self.run_one(ident)
        self.assertEqual(self.lab.recent(),[])

    def test_replay_preserves_query_even_when_replayed_twice(self):
        original=self.run_one("E02",params={"query":"not-in-frame"})
        first=self.run_one("E02","replay",{"run_id":original["id"]},"replay-first-123456789")
        second=self.run_one("E02","replay",{"run_id":first["id"]},"replay-second-123456789")
        self.assertEqual(original["result"],first["result"])
        self.assertEqual(original["result"],second["result"])

    def test_mobile_demonstrations_and_live_training_are_visible(self):
        folder=self.root/"data/mobile-demonstrations/example";folder.mkdir(parents=True)
        record={"id":"example","name":"sock task","state":"complete","outcome":"failure","samples":42,
                "quality":{"usable":False,"reason":"frame gaps"}}
        (folder/"episode.json").write_text(json.dumps(record))
        self.lab.live_status=lambda:{"learning":{"jobs":[{"id":"actual-training"}]}}
        episode=self.run_one("E16")["result"]["episodes"][0]
        self.assertEqual(episode["steps"],42)
        self.assertFalse(episode["quality"]["usable"])
        job=self.run_one("E17",key="training-visible-123456")["result"]["learning"]["jobs"][0]
        self.assertEqual(job["id"],"actual-training")
        suggestions=self.run_one("E18",key="suggestions-visible-123456")["result"]["mobile_demonstration_suggestions"]
        self.assertEqual(suggestions[0]["reason"],"frame gaps")

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):self.lab.get('../../config/access_token')

    def test_memory_identical_labels_not_merged_or_removed_when_unseen(self):
        m=self.lab.memory;a=m.update('add','sock');b=m.update('add','sock')
        self.assertNotEqual(a['object_id'],b['object_id'])
        m.remember_view(dict(image_stamp=time.time(),objects=[]),'map-a')
        self.assertEqual(sum(x['active'] for x in m.objects()),2)
        with self.assertRaises(ValueError):m.update('remove',object_id=a['object_id'],source='detector')
        m.update('remove',object_id=a['object_id'],evidence={'reason':'operator confirmed'})
        self.assertEqual(len(m.history(a['object_id'])),2)
        self.assertEqual(len(ExperienceMemory(self.root).objects()),2)

    def test_memory_duplicate_views_and_retrieval_comparison(self):
        m=self.lab.memory;p=dict(image_stamp=self.now,objects=[dict(label='sock',bbox=[0,0,20,20])])
        self.assertTrue(m.remember_view(p,'m')['added']);self.assertFalse(m.remember_view(p,'m')['added'])
        q=m.retrieve('sock');self.assertEqual(q['retrieval'][0]['id'],q['recent_baseline'][0]['id'])

    def test_curriculum_uses_real_operator_labels(self):
        m=self.lab.memory;m.label('grasp','cloth','intervention','empty gripper')
        q=m.curriculum()['queue'];self.assertEqual(q[0]['interventions'],1)
        self.assertFalse(q[0]['auto_training'])

    def test_plan_orders_hold_before_carry_and_support_before_release(self):
        s=sequence_plan('принеси носок',{})['steps'];names=[x['skill'] for x in s]
        self.assertLess(names.index('verify_hold'),names.index('carry'))
        self.assertLess(names.index('support'),names.index('release'))
        self.assertIn('mcu_watchdog_verified',s[names.index('navigate')]['missing'])

    def test_joint_decoder_checks_checksum_id_and_error(self):
        packet=bytes([255,245,1,4,0,1,44]);packet+=bytes([(~sum(packet[2:]))&255])
        self.assertEqual(decode_position(packet,1)['raw_ticks'],300)
        with self.assertRaises(ValueError):decode_position(packet,2)
        with self.assertRaises(ValueError):decode_position(packet[:-1]+b'\x00',1)
        self.assertFalse(describe({'arm_feedback':[90]*6})['measured']['available'])

    def test_pairing_requires_correct_code_and_local_confirmation(self):
        p=Pairing(self.root);a=p.begin();code=a['command'].split()[1]
        self.assertFalse(p.claim('wrong',42,42,'owner'))
        self.assertTrue(p.claim(code,42,42,'owner'))
        self.assertFalse(p.claim(code,99,99,'stranger'))
        self.assertEqual(json.loads((self.root/'config/telegram.json').read_text())['allowed_user_ids'],[])
        with self.assertRaises(ValueError):p.confirm(99)
        p.confirm(42)
        self.assertEqual(json.loads((self.root/'config/telegram.json').read_text())['allowed_user_ids'],[42])
        with self.assertRaises(ValueError):p.begin()

    def test_expired_pairing_is_rejected(self):
        p=Pairing(self.root);a=p.begin()
        with patch('telegram_pairing.time.time',return_value=time.time()+301):
            self.assertFalse(p.claim(a['command'].split()[1],42,42,'owner'))

    def test_numeric_prediction_is_calculated_not_reported_success(self):
        params=dict(before=dict(x=0,y=0,yaw=0),after=dict(x=.09,y=0,yaw=0),velocity=[.1,0,0],duration_s=1)
        r=self.run_one('E08',params=params)['result']
        self.assertAlmostEqual(r['error_m'],.01);self.assertFalse(r['independent_observation'])

    def test_rgbd_geometry_reports_estimates_and_rejects_bad_depth(self):
        import numpy as np
        from lab_geometry import candidates
        depth=np.full((100,100),.4);k=np.array([[100,0,50],[0,100,50],[0,0,1]])
        r=candidates(depth,k,[40,30,60,70],10,10,10)
        self.assertAlmostEqual(r['candidates'][0]['bbox_extent_m'],.08)
        self.assertIsNone(r['candidates'][0]['feasible'])
        with self.assertRaises(ValueError):candidates(depth*0,k,[40,30,60,70],10,10,10)
        with self.assertRaises(ValueError):candidates(depth,k,[40,30,60,70],12,10,10)

    def test_capture_waits_for_next_real_frame_without_accepting_stale_depth(self):
        import numpy as np
        path=self.root/"data/rgbd-snapshot.npz"
        def write(stamp):
            np.savez(path,rgb=np.zeros((10,10,3),dtype=np.uint8),depth=np.ones((10,10)),k=np.eye(3),stamp=stamp)
        write(0)
        with patch("experiments.time.sleep",side_effect=lambda seconds:write(time.time())) as sleep:
            result=self.lab.capture()
        sleep.assert_called_once()
        with np.load(self.lab.capture_path(result["id"]),allow_pickle=False) as frame:
            self.assertGreater(float(frame["stamp"]),self.now)
        self.assertEqual((result["width"],result["height"]),(10,10))
