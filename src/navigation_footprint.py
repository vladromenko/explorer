"""Apply a camera-pose footprint to all existing costmaps; restore after STOP."""
import json
import math
from pathlib import Path
import time
from geometry_msgs.msg import Polygon,PolygonStamped,Point32


def matches(message,polygon,padding=.02):
    # Published footprints are in map/odom, rotated and padded. Translation and
    # rotation preserve the sorted edge lengths of this rectangle.
    points=message.polygon.points
    if len(points)!=len(polygon):return False
    edges=sorted(math.hypot(a.x-b.x,a.y-b.y) for a,b in zip(points,list(points[1:])+[points[0]]))
    expected=sorted(math.hypot(a[0]-b[0],a[1]-b[1])+2*padding for a,b in zip(polygon,polygon[1:]+polygon[:1]))
    return all(abs(a-b)<.005 for a,b in zip(edges,expected))


class NavigationFootprint:
    def __init__(self,root,node):
        self.node=node;self.profile=json.loads((Path(root)/"config/navigation-footprint.json").read_text())
        self.publishers={name:node.create_publisher(Polygon,"/"+name+"/footprint",10) for name in ("local_costmap","global_costmap")}
        self.observed={};self.since=0.;self.active=False
        for name in self.publishers:
            node.create_subscription(PolygonStamped,"/"+name+"/published_footprint",
                lambda message,info,key=name:self.receive(key,message,info),10)
        node.create_timer(1.,self.maintain)

    def maintain(self):
        if self.active:self.publish(self.profile["polygon"])
        else:self.restore()

    def guard(self):
        now=time.monotonic()
        for name in self.publishers:
            count=self.node.count_publishers("/"+name+"/published_footprint")
            observed=[(identity,stamp) for (key,identity),stamp in self.observed.items() if key==name and now-stamp<2.]
            if not self.active or count<1 or len(observed)<count or (count>1 and any(identity==b"single_publisher_only" for identity,_ in observed)):
                raise ValueError("Camera-pose costmap footprint confirmation expired")

    def receive(self,key,message,info):
        if self.active and matches(message,self.profile["polygon"]):
            gid=info.get("publisher_gid") if isinstance(info,dict) else getattr(info,"publisher_gid",None)
            identity=bytes(gid) if gid is not None else b"single_publisher_only"
            self.observed[(key,identity)]=time.monotonic()

    def publish(self,polygon):
        message=Polygon(points=[Point32(x=float(x),y=float(y),z=0.) for x,y in polygon])
        for publisher in self.publishers.values():publisher.publish(message)

    def apply(self,permit):
        self.observed={};self.since=time.monotonic();self.active=True
        deadline=self.since+4
        while time.monotonic()<deadline:
            permit();self.publish(self.profile["polygon"])
            expected={name:self.node.count_publishers("/"+name+"/published_footprint") for name in self.publishers}
            if all(count>0 and (count==1 or (name,b"single_publisher_only") not in self.observed) and sum(key[0]==name and stamp>=self.since for key,stamp in self.observed.items())>=count
                    for name,count in expected.items()):return
            time.sleep(.1)
        self.restore();raise ValueError("Costmaps did not confirm the camera-pose footprint")

    def restore(self):
        self.active=False
        # Circumscribed polygon retains the existing full-reach 0.38 m envelope
        # after padding, for callers without the camera-pose constraints.
        radius=.36/math.cos(math.pi/16)
        polygon=[[radius*math.cos(i*math.pi/8),radius*math.sin(i*math.pi/8)] for i in range(16)]
        self.publish(polygon)
