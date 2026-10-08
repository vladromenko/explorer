from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from navigation_tasks import NavigationTasks, return_waypoints

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
        with self.assertRaises(ValueError):
            self.tasks.start(dict(kind="navigate_current",x=1.,y=2.,yaw=0.,position_only=True,observing=True))
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

    def test_near_obstacle_departure_returns_to_clear_staging_pose(self):
        calls=[];latest={}
        original={"x":0.,"y":0.,"yaw":0.}
        clear={"x":.3,"y":0.,"yaw":0.}
        self.maps.pose=lambda:original
        self.maps.epoch=lambda:"epoch"
        self.maps.save=lambda name:{"name":name}
        self.missions.state=lambda:{"stop_latched":True}
        self.missions.observe_views=None
        def mission(kind,*args,**kwargs):
            identifier=str(len(calls));calls.append((kind,args))
            details={"visited":0,"staging":{"pose":clear,"moved_m":.3}} if kind=="explore" else clear
            latest.update(id=identifier,state="no_reachable_frontier" if kind=="explore" else "succeeded",details=details)
            return {"id":identifier}
        self.missions.start=mission
        self.missions.status=lambda:{"active":None,"last":latest}
        self.tasks.start(dict(kind="map_room",map_name="staged",observing=True,max_goals=1));self.wait()
        result=self.tasks.status()["last"]
        self.assertEqual(calls[1][1],(.3,0.,0.))
        self.assertTrue(result["result"]["navigation_executed"])
        self.assertTrue(result["result"]["return_target_was_safe_departure"])

    def test_failed_frontier_with_real_motion_is_reported_as_executed(self):
        state={"last":None}
        self.maps.pose=lambda:{"x":1.,"y":2.,"yaw":0.}
        self.maps.epoch=lambda:"epoch"
        self.maps.save=lambda name:{"name":name}
        self.missions.state=lambda:{"stop_latched":True}
        self.missions.observe_views=None
        def mission(kind,*args,**kwargs):
            identifier="explore" if kind=="explore" else "return"
            details={"visited":0,"physical_motion_m":.32,"failed_frontiers":[[1.3,2.]]} if kind=="explore" else self.maps.pose()
            state["last"]=dict(id=identifier,state="limit_reached" if kind=="explore" else "succeeded",details=details)
            return {"id":identifier}
        self.missions.start=mission
        self.missions.status=lambda:{"active":None,"last":state["last"]}
        self.tasks.start(dict(kind="map_room",map_name="motion",observing=True,max_goals=1));self.wait()
        result=self.tasks.status()["last"]["result"]
        self.assertTrue(result["navigation_executed"])
        self.assertEqual(result["coverage"]["visited"],0)

    def test_return_recovery_reverses_measured_corridor(self):
        trace=[{"x":x,"y":0.,"yaw":0.} for x in (0.,.2,.4,.6,.8,1.)]
        waypoints=return_waypoints(trace,{"x":.83,"y":0.,"yaw":.1},{"x":0.,"y":0.,"yaw":0.})
        self.assertEqual([(row["x"],row["yaw"]) for row in waypoints],[(.6,.1)])

    def test_failed_return_stops_and_saves_map_as_partial(self):
        state={"last":None}
        self.maps.pose=lambda:{"x":0.,"y":0.,"yaw":0.}
        self.maps.epoch=lambda:"epoch"
        self.maps.save=lambda name:{"name":name}
        self.missions.state=lambda:{"stop_latched":True}
        self.missions.observe_views=None
        def mission(kind,*args,**kwargs):
            identifier="explore" if kind=="explore" else "return"
            details={"visited":0,"travel_trace":[]} if kind=="explore" else {"reason":"obstacle"}
            state["last"]=dict(id=identifier,state="no_reachable_frontier" if kind=="explore" else "failed",details=details)
            return {"id":identifier}
        self.missions.start=mission
        self.missions.status=lambda:{"active":None,"last":state["last"]}
        self.tasks.start(dict(kind="map_room",map_name="partial",observing=True,max_goals=1));self.wait()
        final=self.tasks.status()["last"]
        self.assertEqual(final["state"],"failed")
        self.assertFalse(final["completed"])
        self.assertEqual(final["result"]["saved_partial_map"],{"name":"partial"})
    def test_corrupt_record_does_not_break_restart_and_motion_never_resumes(self):
        folder=self.root/"data/navigation-tasks"
        (folder/"bad.json").write_text("{")
        (folder/"good.json").write_text('{"id":"good","state":"running","started":1}')
        restored=NavigationTasks(self.root,self.missions,self.maps,lambda permit:None,lambda:None)
        self.assertEqual(restored.status()["last"]["state"],"interrupted")
        self.assertFalse(restored.status()["busy"]);self.assertEqual(len(restored.recovery_errors),1)
