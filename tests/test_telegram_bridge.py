import unittest
from telegram_bridge import authorize,command

class TelegramTests(unittest.TestCase):
    def update(self):return {'message':{'chat':{'id':42,'type':'private'},'from':{'id':42,'is_bot':False},'date':100,'text':'/go kitchen'}}
    def test_allowlist_age_and_groups(self):
        config={'allowed_chat_ids':[42],'allowed_user_ids':[42]}
        self.assertTrue(authorize(self.update(),config,101))
        self.assertIsNone(authorize(self.update(),config,131))
        self.assertIsNone(authorize(self.update(),config,99))
        u=self.update();u['message']['from']['id']=13;self.assertIsNone(authorize(u,config,101))
        u=self.update();u['message']['chat']['type']='group';self.assertIsNone(authorize(u,config,101))
    def test_only_semantic_api_no_raw_motion_or_shell(self):
        self.assertEqual(command('/stop'),('control',{'op':'stop'}))
        self.assertEqual(command('/go kitchen'),('places/go',{'name':'kitchen'}))
        self.assertEqual(command('/survey kitchen, door'),('agents/survey',{'places':['kitchen','door'],'narrate':False}))
        for value in ('/exec rm file','/drive 1 1 1','/clear_stop','/arm 90 90'):
            with self.assertRaises(ValueError):command(value)
        self.assertEqual(command('что видно?')[0],'agent')
