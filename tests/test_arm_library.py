import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from arm_library import ArmLibrary,validate_scene


class ArmLibraryTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        (self.root/"src").mkdir();(self.root/"config").mkdir()
        (self.root/"src/camera_views.py").write_text("VIEWS = {\"forward\": [90, 90, 60, 15, 90]}\n")
        (self.root/"config/factory-arm-startup.json").write_text(json.dumps({"pose_deg":[90]*6}))
        self.model=Mock()
        self.model.path.return_value={"valid":True}
        self.model.fk.side_effect=lambda q,jaw:{"xyz":[(q[0]-90)/1000,(q[1]-90)/1000,0.3],"collision":False}
        self.model.ik.side_effect=lambda xyz,seed,jaw,**kwargs:{"solved":True,"collision":False,
            "servo_deg":[90+xyz[0]*1000,90+xyz[1]*1000,seed[2],seed[3],seed[4]]}
        self.planner=Mock()
        self.planner.plan.side_effect=lambda start,goal,jaw,boxes:{"planned":True,
            "servo_waypoints":[list(start),list(goal)],"joint_trajectory":{"names":["arm1_Joint"],"times":[0,1]}}
        self.library=ArmLibrary(self.root,lambda:self.model,self.planner)

    def tearDown(self):self.directory.cleanup()

    def test_builtin_camera_keeps_live_gripper_and_no_hardware_claim(self):
        camera=self.library.named_pose("camera_forward",[90,90,90,90,90,44])
        self.assertEqual(camera["resolved_servo_deg"],[90,90,60,15,90,44])
        self.assertFalse(camera["measured"])
        self.assertFalse(camera["current_hardware_acceptance"])
        self.assertFalse(self.library.catalog()["execution_allowed"])
        self.planner.plan.assert_not_called()

    def test_saves_only_collision_checked_pose_atomically(self):
        result=self.library.save_pose("Observed",[90]*6,True)
        self.assertEqual(self.model.path.call_count,5)
        self.assertTrue(result["operator_visual_confirmation"])
        self.assertFalse(result["measured"])
        self.assertFalse(result["executed"])
        loaded=ArmLibrary(self.root,lambda:self.model,self.planner).named_pose("Observed",[90]*6)
        self.assertEqual(loaded["resolved_servo_deg"],[90]*6)
        self.assertFalse(self.library.path.with_suffix(".tmp").exists())
        self.model.path.return_value={"valid":False}
        with self.assertRaises(ValueError):self.library.save_pose("Blocked",[90]*6)
        self.assertEqual(len(self.library.catalog()["poses"]),3)

    def test_saved_world_scene_sent_to_existing_planner(self):
        obstacle={"id":"shelf","center":[0.3,0.1,0.2],"size":[0.1,0.1,0.4]}
        self.library.save_scene("Shelf",[obstacle])
        result=self.library.preview_named("camera_forward",[90]*6,"Shelf")
        self.assertEqual(self.planner.plan.call_args.args[3],[obstacle])
        self.assertTrue(result["planned"])
        self.assertFalse(result["execution_allowed"])
        self.assertFalse(result["executed"])

    def test_corrupt_scene_and_non_numeric_input_rejected(self):
        for obstacles in ([{"center":[0,0,0],"size":[1,1,0]}],
                [{"center":[True,0,0],"size":[1,1,1]}],
                [{"id":"same","center":[0,0,0],"size":[1,1,1]}]*2):
            with self.assertRaises(ValueError):validate_scene(obstacles)
        with self.assertRaises(ValueError):self.library.save_pose("Bad",[float("nan")]*6)
        with self.assertRaises(ValueError):self.library.save_pose("camera_forward",[90]*6)

    def test_cartesian_preview_preserves_jaw_and_checks_every_segment(self):
        result=self.library.preview_cartesian([90]*5+[47],[0.015,0.0,0.3])
        self.assertTrue(result["planned"])
        self.assertEqual(result["completed_fraction"],1)
        self.assertEqual(len(result["segments"]),2)
        self.assertEqual(result["servo_waypoints"][-1],[105,90,90,90,90,47])
        self.assertTrue(all(point[5]==47 for point in result["servo_waypoints"]))
        self.assertFalse(result["executed"])
        self.assertFalse(result["arbitrary_6D_orientation_supported"])

    def test_cartesian_does_not_accept_ompl_detour_or_unreachable_point(self):
        self.planner.plan.side_effect=lambda start,goal,*args:{"planned":True,
            "servo_waypoints":[start,[start[0],start[1]+20,*start[2:]],goal]}
        result=self.library.preview_cartesian([90]*6,[0.005,0.0,0.3])
        self.assertFalse(result["planned"])
        self.assertEqual(result["completed_fraction"],0)
        self.assertIn("коридора",result["reason"])
        self.model.ik.side_effect=None
        self.model.ik.return_value={"solved":False,"collision":False}
        result=self.library.preview_cartesian([90]*6,[0.005,0.0,0.3])
        self.assertFalse(result["planned"])

    def test_random_is_seeded_local_reachable_preview_only(self):
        first=self.library.preview_random([90]*5+[42],seed=11)
        second=self.library.preview_random([90]*5+[42],seed=11)
        self.assertEqual(first["accepted_count"],3)
        self.assertEqual([p["requested_goal_deg"] for p in first["previews"]],
            [p["requested_goal_deg"] for p in second["previews"]])
        self.assertTrue(all(p["requested_goal_deg"][5]==42 for p in first["previews"]))
        self.assertFalse(first["execution_allowed"])
        self.model.fk.side_effect=lambda *args:{"xyz":[0,0,0.3],"collision":True}
        rejected=self.library.preview_random([90]*6,count=2,attempts=3)
        self.assertEqual(rejected["attempted"],3)
        self.assertEqual(rejected["accepted_count"],0)


if __name__=="__main__":unittest.main()
