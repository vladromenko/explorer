"""Deterministic navigation missions. No direct actuator publisher or model execution."""
import json, math, threading, time, uuid, sqlite3, re
from pathlib import Path
import numpy as np
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from frontiers import candidates
from map_tools import wait
from survey import SurveyStore
ROOT=Path('/home/vlad/Explorer')
FLAGS=('base_commissioned','lidar_tf_validated','mcu_watchdog_verified','localization_verified')

def navigation_attained(pose,x,y,yaw):
    values=[pose.get('x'),pose.get('y'),pose.get('yaw'),x,y,yaw]
    if not all(type(value) in (int,float) and math.isfinite(value) for value in values):return False
    angle=abs(math.atan2(math.sin(pose['yaw']-yaw),math.cos(pose['yaw']-yaw)))
    return math.hypot(pose['x']-x,pose['y']-y)<=.15 and angle<=.15

def readiness(s,now):
    reasons=[]
    if now-s.get('at',0)>.9 or now<s.get('at',0):reasons.append('controller_stale')
    if s.get('stop_latched',True):reasons.append('stop_latched')
    if s.get('mode')!='AUTONOMOUS':reasons.append('autonomous_mode_not_selected')
    for flag in FLAGS:
        if not s.get('commissioning',{}).get(flag,False):reasons.append(flag)
    for sensor,ttl in [('imu',.5),('odom',.5),('scan0',.6),('scan1',.6),('battery',3)]:
        age=s.get('sensor_age',{}).get(sensor,math.inf)
        if not isinstance(age,(int,float)) or not math.isfinite(age) or not 0<=age<ttl:reasons.append(sensor+'_stale')
    battery=s.get('battery');limit=s.get('commissioning',{}).get('battery_stop_voltage',10.8)
    if battery is None or not math.isfinite(battery) or battery<=limit:reasons.append('battery')
    if s.get('reason')=='SENSOR OR BATTERY FAULT':reasons.append(s['reason'])
    return reasons

class Missions:
    def __init__(self,node,maps):
        self.surveys=SurveyStore(ROOT);self.speak=None;self.compound_guard=None;self.compound_cancel=None
        self.node=node;self.maps=maps;self.lock=threading.RLock();self.active=None;self.last=None
        self.client=ActionClient(node,NavigateToPose,'/navigate_to_pose')
        self.pub=node.create_publisher(String,'/explorer/request',10)
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
    def emit(self,op,**kw):self.pub.publish(String(data=json.dumps(dict(op=op,id=uuid.uuid4().hex,at=time.monotonic(),initiator='missions',**kw))))
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
            errors=[]
            if m.get('handle'):
                try:m['handle'].cancel_goal_async()
                except Exception as exc:errors.append('Nav2 cancel: '+str(exc))
            try:self.emit('finish_mission' if state=='succeeded' else 'cancel_mission',mission=mid)
            except Exception as exc:errors.append('Base stop: '+str(exc))
            if errors:self.last['details']=dict(details,stop_errors=errors)
            self.record(m,state,self.last['details'])
            if errors:raise ValueError('; '.join(errors))

    def cancel(self):
        with self.lock:
            if self.active:self.finish(self.active['id'],'cancelled',{'reason':'operator'})
            else:self.emit('hold_base')
        return dict(cancel_requested=True,physical_stop_confirmed=False)

    def monitor(self):
        with self.lock:
            if not self.active:return
            mid=self.active['id']
            try:
                self.require_ready()
                if self.maps.epoch()!=self.active['map_epoch']:raise ValueError('Map frame changed')
                if time.monotonic()>self.active['deadline_monotonic']:raise ValueError('Mission time limit')
                status=self.state()
                if time.monotonic()-self.active['started_monotonic']>1 and status.get('mission')!=mid:
                    raise ValueError('Mission permission revoked')
                self.active['waiting_for_obstacle']=status.get('reason')=='OBSTACLE'
                if self.active.get('kind')=='delivery' and self.compound_guard:self.compound_guard(mid)
                self.emit('autonomy_lease',mission=mid)
            except Exception as e:
                try:self.finish(mid,'interrupted',{'reason':str(e)})
                except Exception:pass  # Recorded stop errors; never renew this mission's lease.

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
            mid=uuid.uuid4().hex;self.active=dict(id=mid,kind=kind,map_epoch=self.maps.epoch(),started=time.time(),started_monotonic=time.monotonic(),deadline_monotonic=time.monotonic()+(600 if kind=='explore' else 120),phase='planning')
            self.record(self.active,'running',dict(x=x,y=y,yaw=yaw))
            self.emit('begin_mission',mission=mid)
        threading.Thread(target=self.worker,args=(mid,kind,x,y,yaw),daemon=True).start()
        return dict(id=mid,accepted=True,completed=False)

    def start_survey(self,names,narrate=False):
        if not isinstance(names,list) or not 1<=len(names)<=12 or any(not isinstance(n,str) for n in names):
            raise ValueError('Выберите от 1 до 12 сохранённых мест')
        with self.lock:
            self.require_ready()
            if self.active:raise ValueError('Другая поездка уже выполняется')
            places={p['name']:p for p in self.places()}
            if any(n not in places or not places[n]['compatible_map'] for n in names):
                raise ValueError('Место неизвестно или относится к другой карте')
            mid=uuid.uuid4().hex
            self.active=dict(id=mid,kind='survey',map_epoch=self.maps.epoch(),started=time.time(),started_monotonic=time.monotonic(),
                             deadline_monotonic=time.monotonic()+min(1200,120*len(names)),phase='planning',route=names,visited=0)
            self.record(self.active,'running',dict(route=names,narrate=narrate))
            self.emit('begin_mission',mission=mid)
            route=[dict(places[n]) for n in names]
        threading.Thread(target=self.survey_worker,args=(mid,route,narrate),daemon=True).start()
        return dict(id=mid,accepted=True,completed=False)

    def begin_compound(self,mid,kind,timeout,permit=lambda:None):
        with self.lock:
            permit()
            self.require_ready()
            if self.active:raise ValueError('Другая миссия уже выполняется')
            self.active=dict(id=mid,kind=kind,map_epoch=self.maps.epoch(),started=time.time(),
                started_monotonic=time.monotonic(),deadline_monotonic=time.monotonic()+timeout,phase='preparing')
            self.record(self.active,'running',dict(local_execution=True))
            self.emit('begin_mission',mission=mid)
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:
            permit()
            if self.state().get('mission')==mid:return
            time.sleep(.02)
        self.finish(mid,'failed',dict(reason='Controller did not acknowledge mission ownership'))
        raise ValueError('Контроллер не подтвердил владение миссией')

    def survey_permit(self,mid):
        with self.lock:
            if not self.active or self.active['id']!=mid:raise ValueError('Осмотр отменён')
            self.require_ready()
            if self.maps.epoch()!=self.active['map_epoch']:raise ValueError('Карта изменилась')

    def survey_worker(self,mid,route,narrate):
        observations=[]
        try:
            for target in route:
                self.survey_permit(mid)
                pose=self.go(mid,target['x'],target['y'],target['yaw'])
                arrived=time.time()
                with self.lock:
                    self.survey_permit(mid);self.active['phase']='observing'
                    epoch=self.active['map_epoch']
                record=self.surveys.capture(mid,target['name'],pose,epoch,arrived,lambda:self.survey_permit(mid))
                observations.append(record['id'])
                with self.lock:
                    self.survey_permit(mid);self.active['visited']=len(observations)
                if narrate and self.speak:
                    try:self.speak(record['summary'])
                    except (OSError,ValueError):pass
            self.finish(mid,'succeeded',dict(visited=len(observations),observations=observations,
                                             complete_room_coverage_verified=False))
        except Exception as exc:
            self.finish(mid,'failed',dict(reason=str(exc),observations=observations))

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
            self.emit('resume_base',mission=mid)
        try:
            result_future=handle.get_result_async();deadline=time.monotonic()+90
            while not result_future.done():
                self.survey_permit(mid)
                if time.monotonic()>=deadline:raise TimeoutError('Navigation timed out')
                time.sleep(.05)
            self.survey_permit(mid)
            result=result_future.result()
        except Exception:
            try:handle.cancel_goal_async()
            finally:self.emit('hold_base',mission=mid)
            raise
        if result.status!=4:raise ValueError('Navigation did not succeed: '+str(result.status))
        self.hold_base(mid)
        pose=self.maps.pose()
        if not navigation_attained(pose,x,y,yaw):
            raise ValueError('Goal result disagrees with live position/orientation')
        with self.db() as db:db.execute('INSERT INTO visits(seen,x,y,yaw,frame,provisional) VALUES(?,?,?,?,?,?)',(time.time(),pose['x'],pose['y'],pose['yaw'],'map',1))
        return pose

    def hold_base(self,mid):
        """Arrival is complete only after fresh measured base velocity settles."""
        with self.lock:
            if not self.active or self.active['id']!=mid:raise ValueError('Mission cancelled')
            self.active['phase']='settling'
            self.emit('hold_base',mission=mid)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            with self.lock:
                if not self.active or self.active['id']!=mid:raise ValueError('Mission cancelled')
            s=self.state()
            if 0<=time.time()-s.get('at',0)<.9 and s.get('mission')==mid and s.get('base_hold_confirmed'):
                return
            time.sleep(.05)
        raise ValueError('Base did not confirm stationary hold')

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
        except Exception as exc:self.finish(mid,'failed',{'reason':str(exc)})
