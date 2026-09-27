"""Deterministic navigation missions. No direct actuator publisher or model execution."""
import json, math, threading, time, uuid, sqlite3, re
from pathlib import Path
import numpy as np
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from frontiers import candidates
from map_tools import wait
ROOT=Path('/home/vlad/Explorer')
FLAGS=('base_commissioned','lidar_tf_validated','mcu_watchdog_verified','localization_verified')

def readiness(s,now):
    reasons=[]
    if now-s.get('at',0)>.9 or now<s.get('at',0):reasons.append('controller_stale')
    if s.get('stop_latched',True):reasons.append('stop_latched')
    if s.get('mode')!='AUTONOMOUS':reasons.append('autonomous_mode_not_selected')
    for flag in FLAGS:
        if not s.get('commissioning',{}).get(flag,False):reasons.append(flag)
    for sensor,ttl in [('imu',.5),('odom',.5),('scan0',.6),('scan1',.6),('battery',3)]:
        if s.get('sensor_age',{}).get(sensor,1e9)>ttl:reasons.append(sensor+'_stale')
    battery=s.get('battery');limit=s.get('commissioning',{}).get('battery_stop_voltage',10.8)
    if battery is None or not math.isfinite(battery) or battery<=limit:reasons.append('battery')
    if s.get('reason') in ('OBSTACLE','SENSOR OR BATTERY FAULT'):reasons.append(s['reason'])
    return reasons

class Missions:
    def __init__(self,node,maps):
        self.node=node;self.maps=maps;self.lock=threading.RLock();self.active=None;self.last=None
        self.client=ActionClient(node,NavigateToPose,'/navigate_to_pose')
        self.pub=node.create_publisher(String,'/explorer/request',1)
        self.timer=node.create_timer(.1,self.monitor)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS missions(id TEXT PRIMARY KEY, started REAL, ended REAL, kind TEXT, state TEXT, details TEXT)')
            db.execute("UPDATE missions SET ended=?,state='interrupted' WHERE ended IS NULL",(time.time(),))
            db.execute('CREATE TABLE IF NOT EXISTS visits(id INTEGER PRIMARY KEY, seen REAL, x REAL, y REAL, yaw REAL, frame TEXT, provisional INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS places(name TEXT PRIMARY KEY, saved REAL, x REAL,y REAL,yaw REAL, map_epoch TEXT, provisional INTEGER)')

    def db(self):return sqlite3.connect(ROOT/'data/world.sqlite3',timeout=2)
    def state(self):
        try:return json.loads((ROOT/'data/status.json').read_text())
        except (OSError,ValueError):return {}
    def emit(self,op,**kw):self.pub.publish(String(data=json.dumps(dict(op=op,id=uuid.uuid4().hex,at=time.monotonic(),**kw))))
    def status(self):
        with self.lock:
            return dict(active={k:v for k,v in self.active.items() if k not in ('handle',)} if self.active else None,
                        last=self.last,blocked_by=readiness(self.state(),time.time()))
    def require_ready(self):
        reasons=readiness(self.state(),time.time())
        if reasons:raise ValueError('Navigation unavailable: '+', '.join(reasons))
        self.maps.pose()
        if not self.maps.grid or time.monotonic()-self.maps.grid[1]>15:raise ValueError('Map stale')

    def record(self,mission,state,details):
        with self.db() as db:db.execute('INSERT OR REPLACE INTO missions VALUES(?,?,?,?,?,?)',
            (mission['id'],mission['started'],None if state=='running' else time.time(),mission['kind'],state,json.dumps(details)))

    def finish(self,mid,state,details):
        with self.lock:
            if not self.active or self.active['id']!=mid:return
            m=self.active;self.active=None;self.last=dict(id=mid,state=state,details=details,at=time.time())
            if m.get('handle'):m['handle'].cancel_goal_async()
            self.emit('stop')
            self.record(m,state,details)

    def cancel(self):
        with self.lock:
            if self.active:self.finish(self.active['id'],'cancelled',{'reason':'operator'})
            else:self.emit('stop')
        return dict(stopped=True)

    def monitor(self):
        with self.lock:
            if not self.active:return
            mid=self.active['id']
            try:
                self.require_ready()
                if self.maps.epoch()!=self.active['map_epoch']:raise ValueError('Map frame changed')
                if time.time()>self.active['deadline']:raise ValueError('Mission time limit')
                self.emit('autonomy_lease',mission=mid)
            except (OSError,ValueError,KeyError) as e:self.finish(mid,'interrupted',{'reason':str(e)})

    def frontiers(self):
        if not self.maps.grid or time.monotonic()-self.maps.grid[1]>15:raise ValueError('Map stale')
        g=self.maps.grid[0];q=g.info.origin.orientation
        if abs(q.z)>.001 or abs(q.w-1)>.001:raise ValueError('Rotated map grid unsupported')
        p=self.maps.pose();a=np.asarray(g.data).reshape(g.info.height,g.info.width)
        points=candidates(a,g.info.resolution,[g.info.origin.position.x,g.info.origin.position.y],[p['x'],p['y']])
        return dict(candidates=points,known_area_m2=float((a>=0).sum()*g.info.resolution**2),frame='map',provisional=True,executed=False)

    def places(self):
        epoch=self.maps.epoch()
        with self.db() as db:
            db.row_factory=sqlite3.Row
            return [dict(r,compatible_map=r['map_epoch']==epoch) for r in db.execute('SELECT * FROM places ORDER BY saved DESC LIMIT 100')]

    def save_place(self,name):
        if not isinstance(name,str) or not re.fullmatch(r'[\w -]{1,48}',name):raise ValueError('Invalid place name')
        with self.maps.lock:
            self.maps.require_stopped();pose=self.maps.pose();epoch=self.maps.epoch()
            with self.db() as db:db.execute('INSERT OR REPLACE INTO places VALUES(?,?,?,?,?,?,?)',(name,time.time(),pose['x'],pose['y'],pose['yaw'],epoch,1))
        return dict(name=name,pose=pose,provisional=True)

    def go_place(self,name):
        p=next((p for p in self.places() if p['name']==name),None)
        if p is None:raise ValueError('Place has not been saved')
        if not p['compatible_map']:raise ValueError('Place belongs to a different map frame; load and localize on its map first')
        return self.start('navigate',p['x'],p['y'],p['yaw'])

    def start(self,kind,x=None,y=None,yaw=0.):
        if kind not in ('navigate','explore'):raise ValueError('Unsupported mission')
        if kind=='navigate' and (not all(isinstance(v,(float,int)) and math.isfinite(v) for v in (x,y,yaw)) or max(abs(x),abs(y))>100):raise ValueError('Invalid target')
        with self.lock:
            self.require_ready()
            if self.active:raise ValueError('A mission is already active')
            mid=uuid.uuid4().hex;self.active=dict(id=mid,kind=kind,map_epoch=self.maps.epoch(),started=time.time(),deadline=time.time()+(600 if kind=='explore' else 120),phase='planning')
            self.record(self.active,'running',dict(x=x,y=y,yaw=yaw))
        threading.Thread(target=self.worker,args=(mid,kind,x,y,yaw),daemon=True).start()
        return dict(id=mid,accepted=True,completed=False)

    def go(self,mid,x,y,yaw):
        self.maps.preview(x,y)
        self.require_ready()
        if not self.client.wait_for_server(timeout_sec=2):raise ValueError('Navigator unavailable')
        goal=NavigateToPose.Goal();goal.pose.header.frame_id='map';goal.pose.header.stamp=self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x=float(x);goal.pose.pose.position.y=float(y)
        goal.pose.pose.orientation.z=math.sin(yaw/2);goal.pose.pose.orientation.w=math.cos(yaw/2)
        with self.lock:
            if not self.active or self.active['id']!=mid:raise ValueError('Mission cancelled')
            future=self.client.send_goal_async(goal)
        try:handle=wait(future,3)
        except TimeoutError:
            future.add_done_callback(lambda f:f.result().cancel_goal_async() if f.result().accepted else None);raise
        with self.lock:
            if not self.active or self.active['id']!=mid:
                if handle.accepted:handle.cancel_goal_async()
                raise ValueError('Mission cancelled')
            if not handle.accepted:raise ValueError('Navigator rejected goal')
            self.active.update(handle=handle,phase='navigating',target=dict(x=x,y=y,yaw=yaw))
        result=wait(handle.get_result_async(),90)
        if result.status!=4:raise ValueError('Navigation did not succeed: '+str(result.status))
        pose=self.maps.pose()
        if math.hypot(pose['x']-x,pose['y']-y)>.15:raise ValueError('Goal result disagrees with live pose')
        with self.db() as db:db.execute('INSERT INTO visits(seen,x,y,yaw,frame,provisional) VALUES(?,?,?,?,?,?)',(time.time(),pose['x'],pose['y'],pose['yaw'],'map',1))
        return pose

    def worker(self,mid,kind,x,y,yaw):
        try:
            if kind=='navigate':
                result=self.go(mid,x,y,yaw);self.finish(mid,'succeeded',result);return
            visited=[]
            for _ in range(20):
                self.require_ready()
                options=[p for p in self.frontiers()['candidates'] if all(math.hypot(p['x']-x0,p['y']-y0)>.35 for x0,y0 in visited)]
                if not options:
                    self.finish(mid,'no_reachable_frontier',{'visited':len(visited),'complete_room_coverage_verified':False});return
                p=options[0];self.go(mid,p['x'],p['y'],p['yaw']);visited.append((p['x'],p['y']))
                time.sleep(2.) # let new lidar/image observations enter the persistent world model
            self.finish(mid,'limit_reached',{'visited':len(visited)})
        except (ValueError,OSError,TimeoutError,KeyError) as exc:self.finish(mid,'failed',{'reason':str(exc)})
