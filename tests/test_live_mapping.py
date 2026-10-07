"""Real timestamp, inverted TF, rotated map and non-disruptive service recovery."""
import math
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import patch
from lidar_view import LidarView
from map_coordinates import pose_pixel
from mapping_services import MappingServices


def scan(ranges, stamp=100, frame="laser"):
    return NS(header=NS(stamp=NS(sec=stamp,nanosec=0),frame_id=frame),ranges=ranges,
              angle_min=0,angle_increment=math.pi/2,range_min=.1,range_max=10)


class LidarViewTests(unittest.TestCase):
    def test_inverted_sensor_uses_full_quaternion(self):
        view=LidarView(lambda frame:([.2,.1,.3],[1,0,0,0]),lambda:100.1)
        view.receive("scan0",scan([1,2,float("nan"),float("inf"),0,11]))
        result=view.status()["sensors"][0]
        self.assertTrue(result["fresh"])
        self.assertTrue(result["transform_valid"])
        self.assertEqual(result["valid_returns"],2)
        self.assertEqual(result["points_m"],[[1.2,.1],[.2,-1.9]])
        self.assertFalse(view.status()["all_fresh"])

    def test_recent_receipt_cannot_hide_old_source_stamp(self):
        view=LidarView(lambda frame:([0,0,0],[0,0,0,1]),lambda:105)
        view.receive("scan0",scan([1]))
        self.assertFalse(view.status()["sensors"][0]["fresh"])

    def test_receipt_expiry_and_display_bound(self):
        view=LidarView(lambda frame:([0,0,0],[0,0,0,1]),lambda:100,display_limit=360)
        with patch("lidar_view.time.monotonic",return_value=10):view.receive("scan0",scan([1]*1500))
        with patch("lidar_view.time.monotonic",return_value=11):
            result=view.status()["sensors"][0]
        self.assertFalse(result["fresh"])
        self.assertLessEqual(len(result["points_m"]),360)
        self.assertEqual(result["valid_returns"],1500)

    def test_missing_tf_never_fabricates_base_points(self):
        def missing(frame):raise ValueError("TF missing")
        view=LidarView(missing,lambda:100)
        view.receive("scan0",scan([1]))
        result=view.status()["sensors"][0]
        self.assertFalse(result["transform_valid"])
        self.assertEqual(result["points_m"],[])
        self.assertEqual(result["reason"],"TF missing")

    def test_actual_sensor_rate_and_both_sensors(self):
        view=LidarView(lambda frame:([0,0,0],[0,0,0,1]),lambda:100)
        with patch("lidar_view.time.monotonic",return_value=10):view.receive("scan0",scan([1]))
        with patch("lidar_view.time.monotonic",return_value=10.1):
            view.receive("scan0",scan([1]));view.receive("scan1",scan([1]))
            result=view.status()
        self.assertTrue(result["all_fresh"])
        self.assertAlmostEqual(result["sensors"][0]["rate_hz"],10)


class MapCoordinatesTests(unittest.TestCase):
    def test_rotated_origin_and_image_flip(self):
        meta={"origin":[2,3],"origin_yaw":math.pi/2,"resolution":.1,"height":100}
        result=pose_pixel({"x":1.7,"y":3.2,"yaw":math.pi/2},meta,2)
        self.assertAlmostEqual(result["x"],4)
        self.assertAlmostEqual(result["y"],192)
        self.assertAlmostEqual(result["yaw"],0)

    def test_bad_resolution_rejected(self):
        with self.assertRaises(ValueError):pose_pixel({}, {"resolution":0})


class MappingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.commands=[];self.values=["active","active","active","active"]
        def run(command,**kwargs):
            self.commands.append(command)
            if "is-active" in command:return NS(stdout="\n".join(self.values),returncode=0,stderr="")
            self.values[3]="active"
            return NS(stdout="",returncode=0,stderr="")
        self.services=MappingServices(run)
        self.state={"at":time.time(),"stop_latched":True,"velocity":[0,0,0],"mission":None}

    def test_active_slam_is_never_restarted(self):
        result=self.services.recover(self.state)
        self.assertEqual(result["started"],[])
        self.assertTrue(all("is-active" in command for command in self.commands))
        self.assertFalse(result["motor_commands_sent"])

    def test_only_missing_mapview_is_started(self):
        self.values[3]="failed"
        self.assertEqual(self.services.recover(self.state)["started"],["mapview"])
        starts=[command for command in self.commands if "start" in command]
        self.assertEqual(starts,[["systemctl","--user","start","explorer-mapview.service"]])

    def test_motion_mission_stale_state_and_unlatched_stop_block_recovery(self):
        for changes in ({"velocity":[.1,0,0]},{"mission":"running"},{"at":0},{"stop_latched":False}):
            with self.assertRaises(ValueError):self.services.recover(dict(self.state,**changes))
        self.assertEqual(self.commands,[])
