"""Persist map previews for the local UI. No actuator interface."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid

ROOT=Path('/home/vlad/Explorer')
class MapView(Node):
    def __init__(self):
        super().__init__('explorer_map_view')
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid,'/map',self.receive,qos)
    def receive(self,m):
        w,h=m.info.width,m.info.height
        if not w or not h or w*h>16_000_000:return
        grid=np.array(m.data,dtype=np.int16).reshape(h,w)
        pixels=np.full((h,w,3),[89,107,111],dtype=np.uint8)
        pixels[grid==0]=[233,241,230];pixels[grid>50]=[18,32,33]
        pixels[(grid>0)&(grid<=50)]=[149,174,164]
        image=Image.fromarray(np.flipud(pixels));tmp=ROOT/'data/map.tmp.png';image.save(tmp);tmp.replace(ROOT/'data/map.png')
        p=m.info.origin.position
        status=dict(at=time.time(),frame=m.header.frame_id,resolution=m.info.resolution,width=w,height=h,origin=[p.x,p.y],provisional=True,known_cells=int((grid>=0).sum()))
        tmp=ROOT/'data/map.tmp.json';tmp.write_text(json.dumps(status));tmp.replace(ROOT/'data/map.json')

rclpy.init();node=MapView()
try:rclpy.spin(node)
except (KeyboardInterrupt,ExternalShutdownException):pass
finally:node.destroy_node();rclpy.try_shutdown()
