from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from navigation_tasks import NavigationTasks

class RoomTasksTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.stops=[]
        self.missions=SimpleNamespace(status=lambda:{"active":None},cancel=lambda:self.stops.append("mission"))
        self.maps=SimpleNamespace(list_maps=lambda:[])
        self.tasks=NavigationTasks(self.root,self.missions,self.maps,lambda permit:None,lambda:self.stops.append("stop"))
    def wait(self):
        deadline=time.monotonic()+2
        while self.tasks.status()["busy"] and time.monotonic()<deadline:time.sleep(.01)
        self.assertFalse(self.tasks.status()["busy"])
    def test_prepare_failure_does_not_become_success(self):
        def fail(permit):raise ValueError("camera unavailable")
        self.tasks.prepare=fail
        self.tasks.start(dict(kind="map_room",map_name="test",observing=True));self.wait()
        result=self.tasks.status()["last"]
        self.assertEqual(result["state"],"failed");self.assertFalse(result["completed"])
        self.assertIn("camera unavailable",result["reason"]);self.assertEqual(self.stops,["stop"])
    def test_bad_map_name_and_operator_presence(self):
        for spec in ({"kind":"map_room","map_name":None,"observing":True},
                     {"kind":"map_room","map_name":"test"}):
            with self.assertRaises(ValueError):self.tasks.start(spec)
    def test_cancel_does_not_cancel_some_other_navigation_owner(self):
        self.missions.status=lambda:{"active":{"id":"someone_else"}}
        self.tasks.owned_mission="own";self.tasks.cancel()
        self.assertEqual(self.stops,["stop"])
    def test_map_job_observes_then_explores_returns_stops_and_saves(self):
        calls=[];latest={}
        def mission(kind,*args,**kwargs):
            identifier=str(len(calls));calls.append((kind,args,kwargs))
            latest.update(id=identifier,state="limit_reached" if kind=="explore" else "succeeded",details={"visited":1})
            return {"id":identifier}
        def observe(identifier,place,permit):
            permit();calls.append(("observe",place));return [{"id":"view"}]
        self.maps.pose=lambda:{"x":1.,"y":2.,"yaw":.3}
        self.maps.epoch=lambda:"epoch"
        self.maps.save=lambda name:(self.assertTrue(self.stops),{"name":name})[1]
        self.missions.state=lambda:{"stop_latched":True}
        self.missions.start=mission;self.missions.observe_views=observe
        self.missions.status=lambda:{"active":None,"last":latest}
        self.tasks.start(dict(kind="map_room",map_name="test",observing=True,max_goals=1));self.wait()
        result=self.tasks.status()["last"]
        self.assertEqual([call[0] for call in calls],["observe","explore","navigate"])
        self.assertEqual(calls[2][1],(1.,2.,.3))
        self.assertTrue(result["completed"])
        self.assertEqual(result["result"]["saved_map"],{"name":"test"})
        self.assertFalse(result["result"]["entire_apartment_verified"])
    def test_camera_pan_failure_prevents_exploration(self):
        self.maps.pose=lambda:{"x":0.,"y":0.,"yaw":0.};self.maps.epoch=lambda:"epoch"
        def observe(*args):raise ValueError("camera movement failed")
        self.missions.observe_views=observe
        self.tasks.start(dict(kind="survey_room",observing=True));self.wait()
        self.assertEqual(self.tasks.status()["last"]["state"],"failed")
        self.assertIn("camera movement failed",self.tasks.status()["last"]["reason"])
    def test_corrupt_record_does_not_break_restart_and_motion_never_resumes(self):
        folder=self.root/"data/navigation-tasks"
        (folder/"bad.json").write_text("{")
        (folder/"good.json").write_text('{"id":"good","state":"running","started":1}')
        restored=NavigationTasks(self.root,self.missions,self.maps,lambda permit:None,lambda:None)
        self.assertEqual(restored.status()["last"]["state"],"interrupted")
        self.assertFalse(restored.status()["busy"]);self.assertEqual(len(restored.recovery_errors),1)
