"""Bounded map and path operations; never publishes actuator commands."""
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid
from stored_records import records
import numpy as np
from PIL import Image
import yaml
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import ComputePathToPose
from slam_toolbox.srv import SerializePoseGraph, DeserializePoseGraph
from tf2_ros import Buffer, TransformListener, TransformException

ROOT=Path('/home/vlad/Explorer')

def wait(future,seconds=10):
    event=threading.Event();future.add_done_callback(lambda _:event.set())
    if not event.wait(seconds):raise TimeoutError('ROS operation timed out')
    return future.result()

class MapTools:
    def __init__(self,node):
        self.node=node;self.grid=None;self.lock=threading.Lock()
        self.maps=ROOT/'data/maps';self.maps.mkdir(exist_ok=True)
        self.tf=Buffer();self.listener=TransformListener(self.tf,node)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.subscription=node.create_subscription(OccupancyGrid,'/map',self.receive,qos)
        self.plan_client=ActionClient(node,ComputePathToPose,'/compute_path_to_pose')
        self.save_client=node.create_client(SerializePoseGraph,'/slam_toolbox/serialize_map')
        self.load_client=node.create_client(DeserializePoseGraph,'/slam_toolbox/deserialize_map')

    def epoch(self):return json.loads((ROOT/'data/map_session.json').read_text())['id']

    def receive(self,m):self.grid=(m,time.monotonic())

    def pose(self):
        try:t=self.tf.lookup_transform('map','base_footprint',Time())
        except TransformException as exc:raise ValueError('Map pose unavailable') from exc
        age=(self.node.get_clock().now().nanoseconds-Time.from_msg(t.header.stamp).nanoseconds)/1e9
        if not -.5<age<1:raise ValueError('Map pose stale')
        q=t.transform.rotation;p=t.transform.translation
        try:verified=json.loads((ROOT/'config/commissioning.json').read_text()).get('localization_verified') is True
        except (OSError,ValueError,TypeError):verified=False
        return dict(x=p.x,y=p.y,yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),
                    frame='map',source='slam_toolbox_lidar_map_tf',age_s=max(0.,age),
                    localization_verified=verified,provisional=not verified)

    def preview(self,x,y):
        if not all(math.isfinite(v) for v in (x,y)) or max(abs(x),abs(y))>100:raise ValueError('Invalid map coordinate')
        if not self.grid or time.monotonic()-self.grid[1]>15:raise ValueError('Fresh map unavailable')
        if not self.plan_client.wait_for_server(timeout_sec=2):raise ValueError('Planner unavailable')
        goal=ComputePathToPose.Goal();goal.goal.header.frame_id='map';goal.goal.header.stamp=self.node.get_clock().now().to_msg()
        goal.goal.pose.position.x=float(x);goal.goal.pose.position.y=float(y);goal.goal.pose.orientation.w=1.
        goal.planner_id='GridBased';goal.use_start=False
        handle=wait(self.plan_client.send_goal_async(goal),3)
        if not handle.accepted:raise ValueError('Planner rejected request')
        try:result=wait(handle.get_result_async(),8)
        except TimeoutError:
            handle.cancel_goal_async();raise
        if result.status!=4 or result.result.error_code:raise ValueError('No safe path: '+str(result.result.error_code)+' '+str(result.result.error_msg))
        points=[[p.pose.position.x,p.pose.position.y] for p in result.result.path.poses]
        return dict(frame='map',points=points,provisional=True,executed=False,pose=self.pose())

    def list_maps(self):
        paths=[p for p in sorted(self.maps.glob("*/metadata.json")) if not p.parent.name.startswith(".")]
        rows,self.record_errors=records(paths,("name",))
        return [row for _,row in rows]

    def require_stopped(self):
        s=json.loads((ROOT/'data/status.json').read_text())
        if time.time()-s['at']>1 or not s['stop_latched'] or any(s['velocity']):raise ValueError('Map operation requires fresh stop-latched controller')

    def save(self,name):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,48}',name):raise ValueError('Use 1..48 letters, digits, dash or underscore for map name')
        with self.lock:
            self.require_stopped()
            if (self.maps/name).exists():raise ValueError('A map with this name already exists')
            if not self.grid or time.monotonic()-self.grid[1]>15:raise ValueError('Fresh map unavailable')
            if not self.save_client.wait_for_service(timeout_sec=2):raise ValueError('Map serialization unavailable')
            grid=self.grid[0];pose=self.pose()
            staging=self.maps/('.pending-'+uuid.uuid4().hex);staging.mkdir()
            # Use a unique staging path so a late ROS write cannot overwrite a saved map.
            req=SerializePoseGraph.Request();req.filename=str(staging/'graph')
            response=wait(self.save_client.call_async(req),15)
            if response.result!=0:raise ValueError('SLAM graph serialization failed')
            if not (staging/'graph.posegraph').is_file() or not (staging/'graph.data').is_file():raise ValueError('SLAM graph files missing')
            w,h=grid.info.width,grid.info.height
            occupancy=np.array(grid.data).reshape(h,w)
            pixels=np.full((h,w),205,dtype=np.uint8);pixels[occupancy==0]=254;pixels[occupancy>50]=0
            Image.fromarray(np.flipud(pixels)).save(staging/'map.pgm')
            p=grid.info.origin.position;q=grid.info.origin.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            (staging/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',mode='trinary',resolution=grid.info.resolution,origin=[p.x,p.y,yaw],negate=0,occupied_thresh=.65,free_thresh=.25)))
            metadata=dict(name=name,map_id=self.epoch(),saved_at=time.time(),pose=pose,provisional=True,width=w,height=h,resolution=grid.info.resolution)
            (staging/'metadata.json').write_text(json.dumps(metadata,indent=2));staging.rename(self.maps/name)
            return metadata

    def load(self,name):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,48}',name):raise ValueError('Invalid map name')
        with self.lock:
            self.require_stopped()
            directory=self.maps/name
            if not (directory/'metadata.json').is_file():raise ValueError('Saved map not found')
            if not self.load_client.wait_for_service(timeout_sec=2):raise ValueError('Map deserialization unavailable')
            req=DeserializePoseGraph.Request();req.filename=str(directory/'graph')
            # Continue near the saved location; operator must localize if robot was moved.
            metadata=json.loads((directory/'metadata.json').read_text());pose=metadata['pose']
            req.match_type=DeserializePoseGraph.Request.START_AT_GIVEN_POSE
            req.initial_pose.x=pose['x'];req.initial_pose.y=pose['y'];req.initial_pose.theta=pose['yaw']
            wait(self.load_client.call_async(req),15)
            epoch=metadata.get('map_id',f"legacy-{name}-{metadata['saved_at']}")
            temp=ROOT/'data/map_session.tmp';temp.write_text(json.dumps(dict(id=epoch,at=time.time())));temp.replace(ROOT/'data/map_session.json')
            return dict(name=name,load_request_completed=True,localization_verified=False,motion_enabled=False)
