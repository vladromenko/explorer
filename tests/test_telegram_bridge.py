import unittest
from telegram_bridge import authorize,command,format_status,format_experiment_catalog

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
        self.assertEqual(command('/mobile'),('mobile',None))
        for value in ('/exec rm file','/drive 1 1 1','/clear_stop','/arm 90 90'):
            with self.assertRaises(ValueError):command(value)
        self.assertEqual(command('что видно?')[0],'agent')

    def test_lights_are_allowlisted(self):
        self.assertEqual(command('/lights headlights'),('appearance',{'mode':'headlights'}))
        self.assertEqual(command('/lights'),('appearance',None))
        with self.assertRaises(ValueError):command('/lights laser')

    def test_delivery_command_has_explicit_object_and_destination(self):
        path,payload=command('/do носок -> корзина для белья')
        self.assertEqual(path,'autonomy/jobs')
        self.assertEqual(payload['object_query'],'носок')
        self.assertEqual(payload['destination_name'],'корзина для белья')
        with self.assertRaises(ValueError):command('/do просто задача')

    def test_status_contains_power_compute_sensors_and_autonomy(self):
        state={'mode':'MANUAL','stop_latched':True,'reason':'STOP','battery':11.8,
               'battery_gauge':{'available':True,'percent':55},'sensor_age':{'odom':.1,'scan0':.2,'scan1':2},
               'perception':{'stale':False},'controller':{'stale':False},
               'power_telemetry':{'state':'IDLE','battery_voltage_v':11.8,'resources':{'cpu_percent':25,'gpu_percent':4,'ram_available_mb':3500,'temperatures_c':{'cpu':51}}}}
        text=format_status(state,{'accepted':['localization'],'items':[{'state':'accepted'},{'state':'training','name':'Захват','next_action':'Показать'}]})
        for word in ('11.80 В','CPU 25%','GPU 4%','RAM','51.0°C','Датчики','Автономность','Следующее'):self.assertIn(word,text)

    def test_experiment_catalog_explains_effect(self):
        text=format_experiment_catalog([{'id':'E01','name':'Положение','implementation':'Читает состояние, не двигает'}])
        self.assertIn('без движения',text);self.assertIn('Читает состояние',text)
