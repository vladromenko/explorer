"""Deterministic navigation missions. No direct actuator publisher or model execution."""
import json, math, threading, time, uuid, sqlite3, re
from pathlib import Path
import numpy as np
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose, DriveOnHeading
from rclpy.action import ActionClient
from navigation_scope import scope_blockers
from frontiers import candidates, departure_candidate
from navigation_heading import departure_heading, reverse_without_turn
from map_tools import wait
from survey import SurveyStore
ROOT=Path('/home/vlad/Explorer')
FLAGS=('base_commissioned','lidar_tf_validated','mcu_watchdog_verified','localization_verified')

def navigation_attained(pose,x,y,yaw):
    values=[pose.get('x'),pose.get('y'),pose.get('yaw'),x,y,yaw]
    if not all(type(value) in (int,float) and math.isfinite(value) for value in values):return False
    angle=abs(math.atan2(math.sin(pose['yaw']-yaw),math.cos(pose['yaw']-yaw)))
    return math.hypot(pose['x']-x,pose['y']-y)<=.15 and angle<=.15


def navigation_position_attained(pose,x,y):
    values=[pose.get("x"),pose.get("y"),x,y]
    return all(type(value) in (int,float) and math.isfinite(value) for value in values) and math.hypot(pose["x"]-x,pose["y"]-y)<=.15

def readiness(s,now,scope="localized",health=None):
    reasons=[]
    if now-s.get('at',0)>.9 or now<s.get('at',0):reasons.append('controller_stale')
    if s.get('stop_latched',True):reasons.append('stop_latched')
    if s.get('mode')!='AUTONOMOUS':reasons.append('autonomous_mode_not_selected')
    reasons.extend(scope_blockers(s.get("commissioning",{}),scope,health,now))
    for sensor,ttl in [('imu',.5),('odom',.5),('scan0',.6),('scan1',.6),('battery',3)]:
        age=s.get('sensor_age',{}).get(sensor,math.inf)
        if not isinstance(age,(int,float)) or not math.isfinite(age) or not 0<=age<ttl:reasons.append(sensor+'_stale')
    battery=s.get('battery');limit=s.get('commissioning',{}).get('battery_stop_voltage',10.8)
    if battery is None or not math.isfinite(battery) or battery<=limit:reasons.append('battery')
    if s.get('reason')=='SENSOR OR BATTERY FAULT':reasons.append(s['reason'])
    return reasons

class Missions:
    def __init__(self,node,maps):
        self.surveys=SurveyStore(ROOT);self.speak=None;self.observe_views=None;self.camera_guard=None;self.mapping_polygon=None;self.compound_guard=None;self.compound_cancel=None
        self.node=node;self.maps=maps;self.lock=threading.RLock();self.active=None;self.last=None
        self.client=ActionClient(node,NavigateToPose,'/navigate_to_pose')
        self.departure_client=ActionClient(node,DriveOnHeading,"/drive_on_heading")
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
    def require_ready(self,scope=None):
        scope=scope or (self.active or {}).get("scope","localized")
        health={}
        for component in ("navigation","planning"):
            try:health[component]=json.loads((ROOT/"data"/(component+"-health.json")).read_text())
            except (OSError,ValueError):health[component]={}
        reasons=readiness(self.state(),time.time(),scope,health)
        if reasons:raise ValueError('Navigation unavailable: '+', '.join(reasons))
        if scope=="mapping" and self.camera_guard:self.camera_guard(self.state())
        self.maps.pose()
        if not self.maps.grid or time.monotonic()-self.maps.grid[1]>15:raise ValueError('Map stale')

    def record(self,mission,state,details):
        with self.db() as db:db.execute('INSERT OR REPLACE INTO missions VALUES(?,?,?,?,?,?)',
            (mission['id'],mission['started'],None if state=='running' else time.time(),mission['kind'],state,json.dumps(details)))

    def finish(self,mid,state,details):
        with self.lock:
            if not self.active or self.active['id']!=mid:return
            m=self.active;self.active=None;self.last=dict(id=mid,state=state,details=details,at=time.time())
            if m.get("kind")=="explore" and isinstance(details,dict):
                self.last["details"]=dict(details,travel_trace=m.get("travel_trace",[]))
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
        origin_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        p=self.maps.pose();a=np.asarray(g.data).reshape(g.info.height,g.info.width)
        points=candidates(a,g.info.resolution,[g.info.origin.position.x,g.info.origin.position.y],[p['x'],p['y']],origin_yaw=origin_yaw,
            footprint=self.mapping_polygon if (self.active or {}).get("scope")=="mapping" else None,heading=p["yaw"])
        departure=None
        if not points and self.mapping_polygon and (self.active or {}).get("scope")=="mapping":
            departure=departure_candidate(a,g.info.resolution,[g.info.origin.position.x,g.info.origin.position.y],
                [p["x"],p["y"]],self.mapping_polygon,p["yaw"],origin_yaw)
        return dict(candidates=points,departure=departure,known_area_m2=float((a>=0).sum()*g.info.resolution**2),frame='map',provisional=True,executed=False)

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

    def start(self,kind,x=None,y=None,yaw=0.,scope=None,max_goals=20,position_only=False,holonomic=False):
        if kind not in ("navigate","explore"):raise ValueError("Unsupported mission")
        scope=scope or ("mapping" if kind=="explore" else "localized")
        if kind=='navigate' and (not all(isinstance(v,(float,int)) and math.isfinite(v) for v in (x,y,yaw)) or max(abs(x),abs(y))>100):raise ValueError('Invalid target')
        if type(max_goals) is not int or not 1<=max_goals<=20:raise ValueError("Exploration goal budget: 1..20")
        if type(position_only) is not bool or type(holonomic) is not bool:raise ValueError("Invalid navigation mode")
        if position_only and not holonomic:raise ValueError("Position-only navigation requires holonomic travel")
        with self.lock:
            self.require_ready(scope)
            if self.active:raise ValueError('A mission is already active')
            mid=uuid.uuid4().hex;self.active=dict(id=mid,kind=kind,scope=scope,map_epoch=self.maps.epoch(),started=time.time(),started_monotonic=time.monotonic(),deadline_monotonic=time.monotonic()+(600 if kind=='explore' else 120),phase='planning',max_goals=max_goals,position_only=position_only,holonomic=holonomic)
            self.record(self.active,'running',dict(x=x,y=y,yaw=yaw))
            self.emit("begin_mission",mission=mid,kind="mapping" if scope=="mapping" else kind)
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
            self.emit('begin_mission',mission=mid,kind=kind)
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
        pose=self.maps.pose()
        distance=math.hypot(pose["x"]-x,pose["y"]-y)
        if (self.active or {}).get("scope")=="mapping" and distance>.15:
            preview=self.maps.preview(x,y)
            heading=departure_heading(preview["points"],pose)
            difference=abs(math.atan2(math.sin(heading-pose["yaw"]),math.cos(heading-pose["yaw"])))
            if difference>.35 and not reverse_without_turn(distance,difference) and not (self.active or {}).get("holonomic"):
                self.go_goal(mid,pose["x"],pose["y"],heading,rotation_only=True)
                # Once the base faces the route, keep that heading at the
                # frontier instead of demanding a second turn beside furniture.
                yaw=heading
        return self.go_goal(mid,x,y,yaw)

    def go_departure(self,mid,candidate):
        """One short Nav2 collision-checked exit from a mapped near-obstacle start."""
        self.require_ready("mapping")
        before=self.maps.pose()
        distance=float(candidate["distance_m"])
        if not .20<=distance<=.40 or not self.departure_client.wait_for_server(timeout_sec=2):
            raise ValueError("Safe departure unavailable")
        goal=DriveOnHeading.Goal()
        goal.target.x=distance;goal.speed=.05;goal.time_allowance.sec=12
        handle=wait(self.departure_client.send_goal_async(goal),3)
        if not handle.accepted:raise ValueError("Safe departure rejected by navigator")
        with self.lock:
            if not self.active or self.active["id"]!=mid:
                handle.cancel_goal_async();raise ValueError("Mission cancelled")
            self.active.update(handle=handle,phase="safe_departure",target=candidate)
            self.emit("resume_base",mission=mid)
        try:
            future=handle.get_result_async();deadline=time.monotonic()+13
            while not future.done():
                self.survey_permit(mid)
                if time.monotonic()>deadline:raise TimeoutError("Safe departure timed out")
                time.sleep(.05)
            result=future.result()
        except Exception:
            try:handle.cancel_goal_async()
            finally:self.emit("hold_base",mission=mid)
            raise
        self.hold_base(mid)
        if result.status!=4 or result.result.error_code:
            raise ValueError("Safe departure stopped: "+str(result.result.error_code)+" "+result.result.error_msg)
        after=self.maps.pose()
        moved=math.hypot(after["x"]-before["x"],after["y"]-before["y"])
        if moved<.15:raise ValueError("Safe departure did not move the chassis")
        return dict(pose=after,moved_m=moved,policy=candidate["policy"])

    def go_goal(self,mid,x,y,yaw,rotation_only=False):
        pose=self.maps.pose()
        if math.hypot(pose["x"]-x,pose["y"]-y)<.03 and abs(math.atan2(math.sin(pose["yaw"]-yaw),math.cos(pose["yaw"]-yaw)))<.05:
            self.hold_base(mid)
            return self.maps.pose()
        if not rotation_only:self.maps.preview(x,y)
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
            progress_pose=pose;last_progress=time.monotonic();progress_check_due=0.
            self.trace_position(mid,pose)
            while not result_future.done():
                self.survey_permit(mid)
                if time.monotonic()>=deadline:raise TimeoutError('Navigation timed out')
                if time.monotonic()>=progress_check_due:
                    progress_check_due=time.monotonic()+.5
                    current=self.maps.pose()
                    self.trace_position(mid,current)
                    travelled=math.hypot(current["x"]-progress_pose["x"],current["y"]-progress_pose["y"])
                    turned=abs(math.atan2(math.sin(current["yaw"]-progress_pose["yaw"]),math.cos(current["yaw"]-progress_pose["yaw"])))
                    if travelled>.035 or turned>.05:progress_pose=current;last_progress=time.monotonic()
                    if time.monotonic()-last_progress>8. and self.state().get("reason")=="OBSTACLE":
                        raise ValueError("Navigator stalled at a real obstacle")
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
        position_only=(self.active or {}).get("position_only") is True
        if not (navigation_position_attained(pose,x,y) if position_only else navigation_attained(pose,x,y,yaw)):
            raise ValueError('Goal result disagrees with live position/orientation')
        with self.db() as db:db.execute('INSERT INTO visits(seen,x,y,yaw,frame,provisional) VALUES(?,?,?,?,?,?)',(time.time(),pose['x'],pose['y'],pose['yaw'],'map',1))
        return dict(pose,goal_orientation_verified=not position_only or
            abs(math.atan2(math.sin(pose["yaw"]-yaw),math.cos(pose["yaw"]-yaw)))<=.15)

    def trace_position(self,mid,pose):
        with self.lock:
            if not self.active or self.active["id"]!=mid or self.active.get("kind")!="explore":return
            trace=self.active.setdefault("travel_trace",[])
            if not trace or math.hypot(pose["x"]-trace[-1]["x"],pose["y"]-trace[-1]["y"])>=.10:
                trace.append({"x":float(pose["x"]),"y":float(pose["y"]),"yaw":float(pose["yaw"])})
                if len(trace)>500:trace.pop(1)

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
            visited=[];failed=[];failed_attempts=[];observations=[]
            budget=self.active.get("max_goals",20)
            for _ in range(budget):
                self.survey_permit(mid)
                selection=self.frontiers()
                options=[p for p in selection["candidates"] if all(
                    math.hypot(p["x"]-x0,p["y"]-y0)>.35 for x0,y0 in visited+failed)]
                for _departure in range(3):
                    total=float((self.active or {}).get("departure_total_m",0.))
                    available=not options and not visited and selection.get("departure") and total<.65
                    if available:
                        with self.lock:self.active.update(phase="safe_departure")
                        staging=self.go_departure(mid,selection["departure"])
                        with self.lock:self.active.update(staging=staging,departure_total_m=total+staging["moved_m"])
                        selection=self.frontiers()
                        options=[p for p in selection["candidates"] if all(
                            math.hypot(p["x"]-x0,p["y"]-y0)>.35 for x0,y0 in visited+failed)]
                if not options:
                    self.finish(mid,"no_reachable_frontier",dict(visited=len(visited),failed_frontiers=failed,
                        failed_attempts=failed_attempts,physical_motion_m=sum(row["moved_m"] for row in failed_attempts),
                        observations=observations,staging=(self.active or {}).get("staging"),complete_room_coverage_verified=False));return
                p=options[0]
                before=self.maps.pose()
                try:self.go(mid,p["x"],p["y"],p["yaw"])
                except (ValueError,TimeoutError) as exc:
                    self.survey_permit(mid)
                    self.hold_base(mid)
                    failed.append((p["x"],p["y"]))
                    after=self.maps.pose()
                    failed_attempts.append(dict(target=[p["x"],p["y"]],reason=str(exc),
                        moved_m=math.hypot(after["x"]-before["x"],after["y"]-before["y"])))
                    with self.lock:self.active.update(phase="retry_other_frontier",last_failure=str(exc))
                else:
                    visited.append((p["x"],p["y"]))
                    if self.observe_views:
                        records=self.observe_views(mid,"frontier_"+str(len(visited)),lambda:self.survey_permit(mid))
                        observations.extend(row["id"] for row in records)
                    else:
                        row=self.surveys.capture(mid,"frontier_"+str(len(visited)),self.maps.pose(),
                            self.maps.epoch(),time.time(),lambda:self.survey_permit(mid))
                        observations.append(row["id"])
                    with self.lock:self.active.update(visited=len(visited),observations=list(observations))
            self.finish(mid,"limit_reached",dict(visited=len(visited),failed_frontiers=failed,
                failed_attempts=failed_attempts,physical_motion_m=sum(row["moved_m"] for row in failed_attempts),
                observations=observations,staging=(self.active or {}).get("staging"),complete_room_coverage_verified=False))
        except Exception as exc:self.finish(mid,'failed',{'reason':str(exc)})
