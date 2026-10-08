"""Persistent, deterministic room jobs over the existing Nav2 mission owner."""
import copy
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid
from lerobot_bridge import write_json
from stored_records import records

CATALOG=[
 {"id":"map_room","name":"Медленно построить карту","description":"Исследовать доступные границы, вернуться к старту и сохранить карту. Бортовая камера смотрит вперёд при поездке и осматривается на остановках.","parameters":["map_name"]},
 {"id":"survey_room","name":"Осмотреть комнату","description":"Исследовать доступные границы и сохранять наблюдения камеры в достигнутых точках; вернуться к старту.","parameters":[]},
 {"id":"navigate_current","name":"Поехать к точке текущей карты","description":"Медленно проехать по безопасному пути к указанным x/y. Использует непрерывный SLAM.","parameters":["x","y","yaw"]},
 {"id":"patrol","name":"Обойти сохранённые места","description":"Поехать по выбранным местам этой карты, остановиться и сохранить наблюдение в каждом.","parameters":["places"]},
 {"id":"find_object","name":"Найти предмет без захвата","description":"Проверить камеру здесь и в выбранных сохранённых местах. Остановиться при обнаружении; захват не запускается.","parameters":["object_query","places"]}]


def return_waypoints(trace,current,start,limit=4):
    """Reverse a measured outbound corridor after a direct return was blocked."""
    valid=[point for point in trace if isinstance(point,dict) and all(
        type(point.get(key)) in (int,float) and math.isfinite(point[key]) for key in ("x","y","yaw"))]
    if not valid:return []
    nearest=min(range(len(valid)),key=lambda i:math.hypot(valid[i]["x"]-current["x"],valid[i]["y"]-current["y"]))
    selected=[];anchor=current
    for point in reversed(valid[:nearest+1]):
        separation=math.hypot(point["x"]-anchor["x"],point["y"]-anchor["y"])
        remaining=math.hypot(point["x"]-start["x"],point["y"]-start["y"])
        if separation>=.22 and remaining>=.25 and len(selected)<limit:
            selected.append(dict(x=point["x"],y=point["y"],yaw=current["yaw"]))
            anchor=point
    return selected


class NavigationTasks:
    def __init__(self,root,missions,maps,prepare,stop,finder=None,view=None,release=lambda:None):
        self.root=Path(root);self.folder=self.root/"data/navigation-tasks";self.folder.mkdir(parents=True,exist_ok=True)
        self.missions=missions;self.maps=maps;self.prepare=prepare;self.stop=stop;self.finder=finder;self.view=view;self.release=release
        self.lock=threading.RLock();self.active=None;self.last=None;self.worker=None;self.cancelled=threading.Event()
        saved,self.recovery_errors=records(sorted(self.folder.glob("*.json")),("id","state","started"))
        for path,row in saved:
            if row["state"]=="running":row.update(state="interrupted",reason="Process restarted; no automatic motion resume");write_json(path,row)
        if saved:self.last=max((row for _,row in saved),key=lambda row:row["started"])

    def status(self):
        with self.lock:return {"active":copy.deepcopy(self.active),"last":copy.deepcopy(self.last),
            "busy":self.worker is not None,"catalog":CATALOG,"recovery_errors":self.recovery_errors,
            "mapping_speed_limit_m_s":.10,"arm_motion":"stationary_camera_views","browser_required":False}

    def save(self):
        if self.active:write_json(self.folder/(self.active["id"]+".json"),self.active)

    def start(self,spec):
        spec=dict(spec);kind=spec.get("kind")
        if kind not in {row["id"] for row in CATALOG}:raise ValueError("Unknown room task")
        if spec.get("observing") is not True:raise ValueError("Подтвердите свободный участок и наблюдение за началом поездки")
        if kind=="map_room" and (not isinstance(spec.get("map_name"),str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,48}",spec["map_name"])):raise ValueError("Введите имя карты")
        if kind=="map_room" and any(row["name"]==spec["map_name"] for row in self.maps.list_maps()):raise ValueError("Карта уже существует; выберите новое имя")
        if kind=="navigate_current":
            import math
            if any(type(spec.get(key)) not in (int,float) or not math.isfinite(spec[key]) for key in ("x","y","yaw")):
                raise ValueError("Нужны конечные координаты x, y, yaw")
            if type(spec.get("holonomic",False)) is not bool or type(spec.get("position_only",False)) is not bool:
                raise ValueError("Неверный режим движения к точке")
            if spec.get("position_only",False) and not spec.get("holonomic",False):
                raise ValueError("Проверка только x/y требует движения без разворота")
        if kind in ("patrol","find_object"):
            places=spec.get("places",[])
            if not isinstance(places,list) or len(places)>12 or any(not isinstance(name,str) for name in places):raise ValueError("Выберите до 12 мест")
            accepted={row["name"]:row for row in self.missions.places()}
            if any(name not in accepted or not accepted[name]["compatible_map"] for name in places):raise ValueError("Место не принадлежит текущей карте")
            if kind=="patrol" and not places:raise ValueError("Выберите места маршрута")
            if kind=="find_object" and (not self.finder or not str(spec.get("object_query","")).strip()):raise ValueError("Укажите предмет для поиска")
        if type(spec.get("max_goals",20)) is not int or not 1<=spec.get("max_goals",20)<=20:raise ValueError("Goal budget: 1..20")
        with self.lock:
            if self.worker is not None or self.missions.status().get("active"):raise ValueError("Другая поездка ещё выполняется")
            self.cancelled.clear();self.owned_mission=None;identifier=uuid.uuid4().hex
            self.active={"id":identifier,"state":"running","phase":"preparing","started":time.time(),"spec":copy.deepcopy(spec),"events":[],"completed":False}
            self.save();self.worker=identifier
            try:threading.Thread(target=self.run,args=(identifier,),daemon=True,name="explorer-room-task").start()
            except Exception:
                self.active.update(state="failed",reason="Worker did not start");self.save();self.last=self.active;self.active=None;self.worker=None;raise
            return {"id":identifier,"accepted":True,"completed":False}

    def permit(self,identifier):
        if self.cancelled.is_set() or not self.active or self.active["id"]!=identifier:raise ValueError("Room task cancelled")
        if time.time()-self.active["started"]>1200:raise ValueError("Room task time budget expired")

    def event(self,identifier,phase,result=None):
        with self.lock:
            self.permit(identifier);self.active["phase"]=phase
            self.active["events"].append({"at":time.time(),"phase":phase,"result":result});self.save()

    def await_mission(self,identifier,request):
        mid=request["id"];self.owned_mission=mid
        while True:
            self.permit(identifier);state=self.missions.status()
            if not state.get("active"):
                last=state.get("last") or {}
                if last.get("id")!=mid:raise ValueError("Navigation result identity changed")
                if last["state"] not in ("succeeded","no_reachable_frontier","limit_reached"):
                    raise ValueError("Поездка не завершилась: "+str(last.get("details")))
                return last
            if state["active"]["id"]!=mid:raise ValueError("Navigation ownership changed")
            if self.cancelled.wait(.1):self.permit(identifier)

    def search(self,identifier,query):
        found=False
        for view in ("forward","left","right") if self.view else ("forward",):
            if not found:
                self.permit(identifier)
                if self.view:self.view.move(view,lambda:self.permit(identifier))
                request=self.finder.start(query);deadline=time.monotonic()+110
                try:
                    while self.finder.status().get("busy"):
                        self.permit(identifier)
                        if time.monotonic()>deadline:raise ValueError("Object search timed out")
                        self.cancelled.wait(.1)
                except Exception:
                    self.finder.cancel(request["id"]);raise
                result=self.finder.status();self.event(identifier,"object_search",dict(view=view,search=result))
                if result.get("phase")=="error":raise ValueError("Поиск не выполнен: "+str(result.get("error")))
                found=result.get("phase")=="ready" and bool((result.get("result") or {}).get("objects"))
        if self.view:self.view.move("forward",lambda:self.permit(identifier))
        return found

    def run(self,identifier):
        outcome="failed";reason=None;result={}
        try:
            self.permit(identifier);spec=copy.deepcopy(self.active["spec"]);kind=spec["kind"]
            self.prepare(lambda:self.permit(identifier));self.permit(identifier);start=self.maps.pose();epoch=self.maps.epoch()
            self.event(identifier,"start_pose",dict(start,map_epoch=epoch))
            if kind in ("map_room","survey_room"):
                if self.missions.observe_views:
                    initial=self.missions.observe_views(identifier,"start",lambda:self.permit(identifier))
                    self.event(identifier,"initial_observations",initial)
                navigation=self.await_mission(identifier,self.missions.start("explore",scope="mapping",max_goals=spec.get("max_goals",20)))
                self.event(identifier,"exploration",navigation)
                staging=(navigation.get("details") or {}).get("staging")
                return_pose=staging["pose"] if staging else start
                if staging:self.event(identifier,"safe_departure_start",staging)
                try:
                    self.await_mission(identifier,self.missions.start("navigate",return_pose["x"],return_pose["y"],return_pose["yaw"],scope="mapping"))
                except ValueError as direct_error:
                    current=self.maps.pose()
                    waypoints=return_waypoints((navigation.get("details") or {}).get("travel_trace",[]),current,return_pose)
                    self.event(identifier,"return_recovery",dict(reason=str(direct_error),waypoints=waypoints))
                    if not waypoints:raise
                    for waypoint in waypoints:
                        self.permit(identifier)
                        self.await_mission(identifier,self.missions.start("navigate",waypoint["x"],waypoint["y"],waypoint["yaw"],scope="mapping",position_only=True,holonomic=True))
                        self.event(identifier,"return_waypoint",self.maps.pose())
                    self.await_mission(identifier,self.missions.start("navigate",return_pose["x"],return_pose["y"],return_pose["yaw"],scope="mapping",position_only=True,holonomic=True))
                returned=self.maps.pose()
                heading_error=abs(math.atan2(math.sin(returned["yaw"]-return_pose["yaw"]),math.cos(returned["yaw"]-return_pose["yaw"])))
                self.event(identifier,"returned_to_start",dict(pose=returned,return_target=return_pose,
                    return_orientation_verified=heading_error<=.15))
                result["coverage"]=navigation["details"];result["entire_apartment_verified"]=False
                result["return_orientation_verified"]=heading_error<=.15
                result["navigation_executed"]=bool((navigation.get("details") or {}).get("visited",0) or staging or
                    (navigation.get("details") or {}).get("physical_motion_m",0)>.05)
                result["return_target_was_safe_departure"]=bool(staging)
            elif kind=="navigate_current":
                result=self.await_mission(identifier,self.missions.start("navigate",spec["x"],spec["y"],spec["yaw"],scope="mapping",
                    position_only=spec.get("position_only",False),holonomic=spec.get("holonomic",False)))
            else:
                found=self.search(identifier,spec["object_query"]) if kind=="find_object" else False
                for name in spec.get("places",[]):
                    if not found:
                        self.permit(identifier)
                        place=next(row for row in self.missions.places() if row["name"]==name and row["compatible_map"])
                        self.await_mission(identifier,self.missions.start("navigate",place["x"],place["y"],place["yaw"],scope="mapping"))
                        self.event(identifier,"visited",name)
                        observation=self.missions.observe_views(identifier,name,lambda:self.permit(identifier)) if self.missions.observe_views else self.missions.surveys.capture(identifier,name,self.maps.pose(),epoch,time.time(),lambda:self.permit(identifier))
                        self.event(identifier,"observation",observation)
                        if kind=="find_object":found=self.search(identifier,spec["object_query"])
                result={"found":found,"object_query":spec.get("object_query"),"grasp_executed":False}
                if kind=="find_object" and not found:raise ValueError("Предмет не найден в проверенных местах")
            self.permit(identifier)
            if self.maps.epoch()!=epoch:raise ValueError("Map frame changed during task")
            self.stop();deadline=time.monotonic()+3
            while time.monotonic()<deadline and not self.missions.state().get("stop_latched"):self.cancelled.wait(.05)
            self.permit(identifier)
            if kind=="map_room":result["saved_map"]=self.maps.save(spec["map_name"])
            outcome="succeeded"
        except Exception as exc:reason=str(exc)
        finally:
            try:
                if not self.cancelled.is_set():
                    active=self.missions.status().get("active")
                    if active and active["id"]==getattr(self,"owned_mission",None):self.missions.cancel()
                    self.stop()
            except Exception as exc:reason=(reason or "")+"; stop error: "+str(exc);outcome="failed"
            try:self.release()
            except Exception as exc:reason=(reason or "")+"; footprint restore: "+str(exc);outcome="failed"
            if outcome=="failed" and not self.cancelled.is_set() and self.active and self.active["id"]==identifier:
                spec=self.active["spec"]
                if spec["kind"]=="map_room" and any(row["phase"]=="exploration" for row in self.active["events"]):
                    try:result["saved_partial_map"]=self.maps.save(spec["map_name"])
                    except Exception as exc:reason=(reason or "")+"; partial map save: "+str(exc)
            with self.lock:
                if self.active and self.active["id"]==identifier:
                    self.active.update(state="cancelled" if self.cancelled.is_set() else outcome,
                        ended=time.time(),phase="cancelled" if self.cancelled.is_set() else ("finished" if outcome=="succeeded" else "failed"),
                        reason=reason,result=result,completed=outcome=="succeeded" and not self.cancelled.is_set())
                    self.save();self.last=self.active;self.active=None
                self.worker=None

    def cancel(self):
        self.cancelled.set()
        active=self.missions.status().get("active")
        if active and active["id"]==getattr(self,"owned_mission",None):self.missions.cancel()
        self.stop()
        return {"cancel_requested":True,"physical_stop_confirmed":False}
