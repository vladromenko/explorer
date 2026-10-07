"""Jetson-only bindings for the accepted, bounded sock delivery scenario.

Task coordinates/calibrations are physical acceptance artifacts, never guessed
from a detector label. MoveIt plans each actual measured pose to its next goal.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from grasp_verification import verify_lift, verify_place
from delivery_vision import FeatureObject, reacquire_candidate
from mobile_alignment import correction
from semantic_world import guarded_closure
from learning_stack import CandidateScorer

SETUP_ARTIFACTS={'config/delivery.json','config/handeye-accepted.json','config/gripper-accepted.json',
                 'config/explorer.urdf','config/explorer.srdf'}


def validate_delivery_goal(goal, settings):
    """The accepted physical executor currently handles socks at one destination."""
    query=str(goal.get("object_query", "")).strip().lower()
    if query not in {"sock", "socks", "носок", "носки"}:
        raise ValueError("Принятый сценарий доставки поддерживает только носок; запрос не выполнен")
    destination=(goal.get("destination") or {}).get("name")
    if not isinstance(destination,str) or destination.strip()!=settings["destination"]:
        raise ValueError("Место назначения команды не совпадает с принятым сценарием доставки")


def load_settings(root, source_sha, calibration_sha, evidence=None):
    root=Path(root)
    if evidence is None:evidence=json.loads((root/'config/delivery-acceptance.json').read_text())
    if evidence.get('accepted') is not True or not evidence.get('physical_test_records'):
        raise ValueError('Нет результатов приёмки контроллера, руки и геометрии для первого задания')
    if evidence.get('firmware_source_sha256')!=source_sha or evidence.get('controller_calibration_sha256')!=calibration_sha:
        raise ValueError('Приёмка доставки относится к другой прошивке или калибровке')
    required=SETUP_ARTIFACTS
    artifacts=evidence.get('artifacts',{})
    if not required.issubset(artifacts):raise ValueError('Неполный список артефактов приёмки доставки')
    if set(artifacts)&set(evidence['physical_test_records']):raise ValueError('Настройки и журналы приёмки пересекаются')
    verified={}
    for name,digest in dict(artifacts,**evidence['physical_test_records']).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Недопустимый путь артефакта: '+name)
        verified[name]=path.read_bytes()
        if hashlib.sha256(verified[name]).hexdigest()!=digest:
            raise ValueError('Артефакт приёмки изменён: '+name)
    kinds=set()
    for name in evidence['physical_test_records']:
        record=json.loads(verified[name])
        if (record.get('hardware_executed') is not True or record.get('simulation') is True or
            record.get('outcome')!='passed' or record.get('firmware_source_sha256')!=source_sha or
            record.get('controller_calibration_sha256')!=calibration_sha):
            raise ValueError('Файл не подтверждает физическую приёмку текущей машины: '+name)
        kinds.add(record.get('kind'))
    if not {'controller','arm','geometry'}.issubset(kinds):
        raise ValueError('Не завершены физические проверки контроллера, руки и геометрии')
    config=json.loads(verified['config/delivery.json'])
    handeye=json.loads(verified['config/handeye-accepted.json'])
    gripper=json.loads(verified['config/gripper-accepted.json'])
    if handeye.get('execution_authorized') is not True or not (handeye.get('measured_joint_positions') is True or
            handeye.get('joint_state_source')=='command_estimate' and handeye.get('physical_validation_record')):
        raise ValueError('Нет физически проверенной калибровки камеры для текущего источника позы')
    if gripper.get('execution_authorized') is not True or gripper.get('aperture_mm_calibrated') is not True:
        raise ValueError('Нет принятой калибровки захвата')
    transform=np.asarray(handeye['camera_to_mount_reference'],dtype=float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]):
        raise ValueError('Неверное преобразование камеры')
    if handeye.get('reference_mount')!='arm4' or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(transform[:3,:3])<.999:
        raise ValueError('Неверная система координат камеры')
    config.update(camera_to_mount=transform.tolist(),open_deg=gripper['open_deg'],close_deg=gripper['sock_close_deg'])
    for key in ('transport_deg','search_deg'):
        value=np.asarray(config[key],dtype=float)
        if value.shape!=(6,) or not np.isfinite(value).all():raise ValueError('Нет принятой позы '+key)
    plane=np.asarray(config['floor_plane_base']);quat=np.asarray(config['grasp_quaternion_xyzw'])
    if plane.shape!=(4,) or not np.isfinite(plane).all() or abs(np.linalg.norm(plane[:3])-1)>.001:
        raise ValueError('Нет измеренной плоскости пола')
    if quat.shape!=(4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)>.001:
        raise ValueError('Нет принятой ориентации захвата')
    if not isinstance(config['search_places'],list) or not 1<=len(config['search_places'])<=5 or any(not isinstance(v,str) for v in config['search_places']):
        raise ValueError('Нужны от одного до пяти принятых мест поиска')
    if not isinstance(config['destination'],str) or not config['destination']:raise ValueError('Нет места доставки')
    if not isinstance(config.get('map_epoch'),str) or not config['map_epoch']:
        raise ValueError('Настройки доставки не привязаны к принятой карте')
    places=config.get('place_poses',{})
    for name in set(config['search_places']+[config['destination']]):
        pose=places.get(name,{})
        if not all(type(pose.get(key)) in (int,float) and math.isfinite(pose[key]) for key in ('x','y','yaw')):
            raise ValueError('Нет принятой позы места: '+name)
    for key,lo,hi in [('open_deg',30,170),('close_deg',30,170),('approach_height_m',.03,.10),
                      ('lift_height_m',.04,.12),('grasp_tcp_offset_m',-.02,.04),('gripper_linkage_rad',-1.54,0)]:
        value=config[key]
        if type(value) not in (float,int) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Параметр доставки вне принятого диапазона: '+key)
    if config['close_deg']<=config['open_deg']:raise ValueError('Неверное направление захвата')
    zone=config['drop_zone'];center=np.asarray(zone['center_xyz'])
    if center.shape!=(3,) or not np.isfinite(center).all() or np.linalg.norm(center)>1 or not .03<=zone['radius_m']<=.3 or not .005<=zone['support_tolerance_m']<=.03:
        raise ValueError('Нет принятой доступной области размещения')
    return config


def seal_setup(root, source_sha, calibration_sha, record_paths, write=True):
    """Link already accepted physical prerequisites; never claim a delivered sock.

    No motion or inferred calibration. The records must already bind the exact
    settings they physically checked, so editing geometry cannot reuse old logs.
    """
    root=Path(root).resolve()
    profile=json.loads((root/'config/controller-profile.json').read_text())
    if profile.get('firmware_source_sha256')!=source_sha or profile.get('calibration_sha256')!=calibration_sha:
        raise ValueError('Профиль контроллера изменился до оформления приёмки')
    artifacts={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(SETUP_ARTIFACTS)}
    records={};bound={}
    for name in record_paths:
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Недопустимый журнал приёмки')
        relative=path.relative_to(root).as_posix();raw=path.read_bytes();record=json.loads(raw)
        records[relative]=hashlib.sha256(raw).hexdigest()
        for artifact,digest in record.get('artifacts',{}).items():
            if artifact in artifacts and digest!=artifacts[artifact]:
                raise ValueError('Журнал относится к прежним настройкам: '+artifact)
            bound[artifact]=digest
    if any(bound.get(name)!=digest for name,digest in artifacts.items()):
        raise ValueError('Физические журналы не привязаны ко всем текущим настройкам доставки')
    evidence=dict(accepted=True,scope='delivery_prerequisites',physical_delivery_verified=False,
                  firmware_source_sha256=source_sha,controller_calibration_sha256=calibration_sha,
                  artifacts=artifacts,physical_test_records=records)
    load_settings(root,source_sha,calibration_sha,evidence)
    path=root/'config/delivery-acceptance.json'
    if write:
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(evidence,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        temporary.replace(path)
    return dict(prerequisites_accepted=True,physical_delivery_verified=False,record=str(path),saved=bool(write))


def placement_zone(zone, saved_pose, actual_pose):
    """Keep the taught placement point fixed in map when Nav2 stops with error."""
    x,y,z=np.asarray(zone['center_xyz'],dtype=float)
    sx,sy,sa=(float(saved_pose[k]) for k in ('x','y','yaw'))
    ax,ay,aa=(float(actual_pose[k]) for k in ('x','y','yaw'))
    if not np.isfinite([x,y,z,sx,sy,sa,ax,ay,aa]).all():raise ValueError('Неверная локализация размещения')
    wx=sx+math.cos(sa)*x-math.sin(sa)*y;wy=sy+math.sin(sa)*x+math.cos(sa)*y
    dx,dy=wx-ax,wy-ay
    return dict(zone,center_xyz=[math.cos(aa)*dx+math.sin(aa)*dy,-math.sin(aa)*dx+math.cos(aa)*dy,float(z)])


def validated_places(settings, places, epoch):
    if epoch!=settings['map_epoch']:raise ValueError('Карта отличается от принятого сценария доставки')
    available={p['name']:p for p in places}
    for name,accepted in settings['place_poses'].items():
        current=available.get(name)
        if not current or not current['compatible_map']:raise ValueError('Нет принятого места: '+name)
        angle=math.atan2(math.sin(current['yaw']-accepted['yaw']),math.cos(current['yaw']-accepted['yaw']))
        if max(abs(current['x']-accepted['x']),abs(current['y']-accepted['y']),abs(angle))>1e-6:
            raise ValueError('Сохранённое место изменилось после приёмки: '+name)
    return available


class DeliveryRobot:
    def __init__(self, root, missions, arm, trajectory, finder, vision, model):
        self.root,self.missions,self.arm,self.trajectory=Path(root),missions,arm,trajectory
        self.finder,self.vision,self.model=finder,vision,model
        self.scorer=CandidateScorer(self.root)
        self.mid=None;self.settings=None;self.tracking=False;self.grasp_xyz=None;self.boot=None;self.owner_check=lambda:None;self.search_id=None
        self.drop_zone=None;self.destination_pose=None;self.grasp_selection=None

    def select_grasp(self,point,observation):
        extent=observation.get('object_extent_xyz_m',[.05,.04,.025])
        uncertainty=float(observation.get('object_position_uncertainty_m',.01))
        candidates=[]
        for lateral in (0.,-.01,.01):
            candidate=np.asarray(point,dtype=float)+[0,lateral,0]
            above=candidate+[0,0,self.settings['approach_height_m']]
            solved=self.model().ik(above,self.arm.reference()['servo_deg'][:5],self.settings['gripper_linkage_rad'],self.settings['grasp_quaternion_xyzw'])
            if solved.get('solved') and not solved.get('collision'):
                candidates.append(dict(approach_x=float(candidate[0]),approach_y=float(candidate[1]),approach_z=float(candidate[2]),
                    aperture_m=min(.07,max(.01,float(extent[1])*1.15)),roll_rad=0.,base_shift_m=0.,
                    clearance_m=max(.001,float(self.settings['approach_height_m'])-uncertainty),
                    trajectory_cost=float(solved.get('residual',0))+abs(lateral)*10,point=candidate.tolist()))
        if not candidates:raise ValueError('Нет достижимого кандидата захвата')
        context=dict(object_width_m=float(extent[0]),object_height_m=float(extent[2]),distance_m=float(np.linalg.norm(point)),
            visibility=float(observation.get('confidence',0)),depth_uncertainty_m=uncertainty,target_distance_m=1.,
            softness=1. if str(self.settings.get('object_kind','soft')).startswith('soft') else 0.,scene_clutter=float(observation.get('scene_clutter',0)),
            previous_failures=0.)
        ranked=self.scorer.choose(context,candidates,exploration=float(self.settings.get('grasp_exploration',0)))
        ranked['context']=context
        chosen=ranked['rows'][ranked['selected']]['candidate'];self.grasp_selection=ranked
        return np.asarray(chosen['point'],dtype=float)

    def blockers(self):
        reasons=[]
        if self.finder.status().get('busy'):reasons.append('Дождитесь завершения текущего распознавания')
        if not getattr(self.arm,'native',False):
            flags={}
            try:flags=json.loads((self.root/'config/commissioning.json').read_text())
            except (OSError,ValueError):pass
            if flags.get('camera_tf_validated') is not True:
                reasons.append('Не принята привязка камеры к руке robotio')
            if flags.get('localization_verified') is not True:
                reasons.append('Не принята локализация и повторное определение позы на сохранённой карте')
            try:gripper=json.loads((self.root/'config/gripper-accepted.json').read_text())
            except (OSError,ValueError):gripper={}
            if not (gripper.get('execution_authorized') is True and gripper.get('aperture_mm_calibrated') is True):
                reasons.append('Не приняты раскрытие, контакт и удержание предмета захватом')
            reasons.append('До первого запуска нужен принятый сценарий robotio с местами поиска и размещения')
        else:
            status=self.arm.status()
            if status.get('blocked_by'):reasons.append(status['blocked_by'])
            if status.get('busy'):reasons.append('Рука уже выполняет движение')
            if self.trajectory.status().get('busy'):reasons.append('Исполнитель пути руки занят')
            try:
                profile=self.arm.profile
                load_settings(self.root,profile['firmware_source_sha256'],profile['calibration_sha256'])
            except FileNotFoundError as exc:reasons.append('Не завершена приёмка: '+Path(exc.filename).name)
            except (OSError,ValueError,KeyError,TypeError) as exc:reasons.append(str(exc))
        try:self.missions.require_ready()
        except (OSError,ValueError,KeyError) as exc:reasons.append(str(exc))
        return list(dict.fromkeys(reasons))

    def begin(self, mid, check, goal=None):
        check()
        self.owner_check=check
        reasons=self.blockers()
        if reasons:raise ValueError('; '.join(reasons))
        self.settings=load_settings(self.root,self.arm.profile['firmware_source_sha256'],self.arm.profile['calibration_sha256'])
        if goal is not None:validate_delivery_goal(goal,self.settings)
        validated_places(self.settings,self.missions.places(),self.missions.maps.epoch())
        self.boot=self.arm.reference()['boot_id']
        self.mid=mid;self.tracking=False;self.drop_zone=None;self.destination_pose=None
        self.missions.begin_compound(mid,'delivery',900,check)
        check()
        self.hold()
        self.arm.prepare_geometry()
        self._move(self.settings['transport_deg'])

    def permit(self, mid):
        self.owner_check()
        if self.mid!=mid:raise ValueError('Исполнитель доставки отменён')
        self.missions.survey_permit(mid)
        self.arm.reference(expected_boot=self.boot)
        if self.tracking:self.vision.latest()

    def hold(self):
        self.owner_check()
        self.missions.hold_base(self.mid)
        return dict(base_stationary_confirmed=True)

    def _move(self, goal):
        self.permit(self.mid);self.hold()
        if self.arm.status().get('busy') or self.trajectory.status().get('busy'):
            raise ValueError('Другой исполнитель ещё управляет рукой')
        actual=self.arm.reference()['servo_deg']
        if np.allclose(actual,goal,atol=.3,rtol=0):
            state=self.arm._state();stamp=state.get('monotonic_ns');controller=state.get('controller',{})
            if (not isinstance(stamp,int) or not 0<=time.monotonic_ns()-stamp<350_000_000
                    or not state.get('telemetry_fresh') or controller.get('arm_enabled') is not False
                    or controller.get('arm_cancel_pending') is not False):
                raise ValueError('Рука ещё исполняет или отменяет предыдущую команду')
            estimated=getattr(self.arm,'profile',{}).get('manual_reference_version')==1
            return dict(attained=not estimated,measured=not estimated,command_completed=True,already_at_command_goal=estimated)
        plan=self.trajectory.plan([float(v) for v in goal]);mid=self.mid
        started=self.trajectory.start_local(plan['plan_id'],lambda:self.permit(mid))
        session=started['session'];end=time.monotonic()+70
        while time.monotonic()<end:
            self.permit(mid);state=self.trajectory.status()
            if state.get('session')!=session:raise ValueError('Исполнитель пути заменён')
            if not state['busy']:
                measured=self.arm.reference()['servo_deg']
                completed=(state.get('phase')=='reached' and state.get('reached') is True or
                           state.get('phase')=='command_completed' and state.get('command_completed') is True)
                if (not completed or
                    not np.allclose(measured,goal,atol=.5,rtol=0)):
                    raise ValueError('Поза не достигнута: '+str(state.get('reason')))
                estimated=getattr(self.arm,'profile',{}).get('manual_reference_version')==1
                return dict(attained=not estimated,measured=not estimated,command_completed=True,session=session,
                            measured_deg=None if estimated else measured,q_estimated_deg=measured if estimated else None)
            time.sleep(.03)
        self.trajectory.stop();raise ValueError('Истёк срок пути руки')

    def _xyz(self, point, grip):
        current=self.arm.reference()['servo_deg']
        solution=self.model().ik(point,current[:5],self.settings['gripper_linkage_rad'],self.settings['grasp_quaternion_xyzw'])
        if not solution.get('solved') or solution.get('collision'):raise ValueError('Нет доступного положения захвата без столкновения')
        return self._move(solution['servo_deg']+[grip])

    def find(self):
        places=validated_places(self.settings,self.missions.places(),self.missions.maps.epoch())
        for name in self.settings['search_places']:
            self.permit(self.mid)
            p=places.get(name)
            if not p or not p['compatible_map']:raise ValueError('Место поиска не принадлежит текущей карте: '+name)
            self.missions.go(self.mid,p['x'],p['y'],p['yaw']);self.hold()
            self._move(self.settings['search_deg'])
            found=self._detect_current(name)
            if found:return found
            self._move(self.settings['transport_deg'])
        raise ValueError('В принятых местах не найден доступный носок')

    def _detect_current(self,place):
            arrived=time.time();deadline=time.monotonic()+2
            while float(self.vision.snapshot()['stamp'])<arrived:
                self.permit(self.mid)
                if time.monotonic()>deadline:raise ValueError('Нет кадра после достижения поисковой позы')
                time.sleep(.03)
            before=np.asarray(self.arm.reference()['servo_deg'],dtype=float);base=self.missions.maps.pose()
            request=self.finder.start('sock');self.search_id=request['id'];end=time.monotonic()+105
            try:self.permit(self.mid)
            except Exception:
                self.finder.cancel(request['id']);raise
            while self.finder.status().get('busy') and time.monotonic()<end:
                self.permit(self.mid);time.sleep(.1)
            found=self.finder.status()
            if found.get('id')!=request['id'] or found.get('phase')!='ready':
                raise ValueError('Поиск предмета не завершился: '+str(found.get('error','timeout')))
            objects=[o for o in found['result']['objects'] if o.get('position') and o.get('confidence',0)>=.35]
            if len(objects)>1:raise ValueError('Несколько похожих предметов: выбор для принятого сценария неоднозначен')
            if objects:
                current=np.asarray(self.arm.reference()['servo_deg'],dtype=float);pose=self.missions.maps.pose()
                if np.max(np.abs(before-current))>.15 or max(abs(pose[k]-base[k]) for k in ('x','y','yaw'))>.003:
                    raise ValueError('Камера сместилась во время распознавания')
                with np.load(self.root/'data/object-searches'/request['id']/'rgbd.npz',allow_pickle=False) as f:initial=dict(f)
                fresh=self.vision.snapshot()
                candidate=reacquire_candidate(initial,fresh,objects[0]['bbox'])
                tracker=FeatureObject(fresh,candidate['bbox'])
                self.vision.start_tracker(tracker,self.settings)
                end=time.monotonic()+2
                while not self.vision.frames and time.monotonic()<end:
                    self.permit(self.mid);time.sleep(.03)
                target=self.vision.latest();self.tracking=True
                return dict(place=place,target=target,semantic_label='sock',semantic_identity_verified=False,reacquisition=candidate)
            return None

    def find_aligned(self):
        found=self._detect_current('base_aligned')
        if not found:raise ValueError('После коррекции базы носок не найден; движение к старой координате запрещено')
        return found

    def approach(self, target):
        # The accepted bounded release searches saved base poses with a reachable
        # foreground object. It never drives blind using optical-frame coordinates.
        self.hold();observation=self.vision.latest();point=np.asarray(observation['object_xyz'])
        alignment=correction(point)
        if alignment['reobserve_required']:
            pose=self.missions.maps.pose();bearing=pose['yaw']+alignment['bearing_rad']
            world_x=pose['x']+math.cos(pose['yaw'])*point[0]-math.sin(pose['yaw'])*point[1]
            world_y=pose['y']+math.sin(pose['yaw'])*point[0]+math.cos(pose['yaw'])*point[1]
            goal_x=world_x-alignment['desired_distance_m']*math.cos(bearing)
            goal_y=world_y-alignment['desired_distance_m']*math.sin(bearing)
            self.vision.stop();self.tracking=False
            self.missions.go(self.mid,goal_x,goal_y,bearing);self.hold()
            return dict(reachable=False,base_repositioned=True,reobserve_required=True,
                        alignment=alignment,goal_map=dict(x=goal_x,y=goal_y,yaw=bearing))
        point[2]+=self.settings['grasp_tcp_offset_m'];point=self.select_grasp(point,observation)
        above=point+[0,0,self.settings['approach_height_m']]
        solved=self.model().ik(above,self.arm.reference()['servo_deg'][:5],self.settings['gripper_linkage_rad'],self.settings['grasp_quaternion_xyzw'])
        if not solved.get('solved') or solved.get('collision'):raise ValueError('Предмет недоступен из принятой точки поиска')
        self.grasp_xyz=point.tolist()
        return dict(reachable=True,base_stationary_confirmed=True,grasp_xyz=self.grasp_xyz,alignment=alignment,
                    grasp_selection=self.grasp_selection)

    def reobserve(self, target):
        current=self.vision.latest()
        if np.linalg.norm(np.asarray(current['object_xyz'])-target['target']['object_xyz'])>.015:
            raise ValueError('Предмет сдвинулся до захвата')
        return current
    def inspect(self):
        return self.vision.observe(lambda:self.permit(self.mid),duration=.4)
    def align(self, target):
        """Recompute the pregrasp from fresh tracking until the target settles."""
        corrections=[]
        for _ in range(3):
            self.permit(self.mid);current=self.vision.latest()
            point=np.asarray(current['object_xyz'],dtype=float)
            point[2]+=self.settings['grasp_tcp_offset_m']
            error=float(np.linalg.norm(point-np.asarray(self.grasp_xyz)))
            if error<=.008:return dict(aligned=True,corrections=corrections,error_m=error)
            if error>.04:raise ValueError('Цель сместилась за предел локальной визуальной коррекции')
            self.grasp_xyz=point.tolist()
            above=(point+[0,0,self.settings['approach_height_m']]).tolist()
            corrections.append(dict(error_m=error,command=self._xyz(above,self.settings['open_deg'])))
        current=np.asarray(self.vision.latest()['object_xyz'],dtype=float)
        residual=float(np.linalg.norm(current-np.asarray(self.grasp_xyz)))
        if residual>.008:raise ValueError('Визуальное выравнивание не сошлось')
        return dict(aligned=True,corrections=corrections,error_m=residual)
    def pregrasp(self, target):return self._xyz((np.asarray(self.grasp_xyz)+[0,0,self.settings['approach_height_m']]).tolist(),self.settings['open_deg'])
    def observe(self):return self.vision.observe(lambda:self.permit(self.mid))
    def grasp(self, target):
        self._xyz(self.grasp_xyz,self.settings['open_deg'])
        if self.settings.get('guarded_closure_enabled') is not True:
            return self._xyz(self.grasp_xyz,self.settings['close_deg'])
        observations=self.vision.observe(lambda:self.permit(self.mid),duration=.25)
        angle=float(self.settings['open_deg']);commands=[]
        while angle<self.settings['close_deg']:
            decision=guarded_closure(observations,soft=self.settings.get('object_kind','soft')=='soft')
            if decision['action'] in ('hold','stop'):
                return dict(guarded=True,decision=decision,commands=commands,final_deg=angle)
            step=float(decision.get('step_deg',1));angle=min(float(self.settings['close_deg']),angle+step)
            commands.append(self._xyz(self.grasp_xyz,angle))
            observations=self.vision.observe(lambda:self.permit(self.mid),duration=.25)
        return dict(guarded=True,decision={'action':'hold','reason':'accepted_close_limit'},commands=commands,final_deg=angle)
    def regrasp(self,target):
        """One bounded retry after a visually proven empty grasp."""
        self._xyz((np.asarray(self.grasp_xyz)+[0,0,self.settings['approach_height_m']]).tolist(),self.settings['open_deg'])
        current=self.vision.latest();point=np.asarray(current['object_xyz'],dtype=float)
        point[2]+=self.settings['grasp_tcp_offset_m'];self.grasp_xyz=point.tolist()
        self.align(target)
        return self.grasp(target)
    def lift(self):return self._xyz((np.asarray(self.grasp_xyz)+[0,0,self.settings['lift_height_m']]).tolist(),self.settings['close_deg'])
    def verify_hold(self, before):return verify_lift(before,self.observe(),True)
    def transport(self):return self._move(self.settings['transport_deg'][:5]+[self.settings['close_deg']])
    def carry(self):
        places=validated_places(self.settings,self.missions.places(),self.missions.maps.epoch());p=places.get(self.settings['destination'])
        if not p or not p['compatible_map']:raise ValueError('Место доставки относится к другой карте')
        self.destination_pose=dict(p)
        # Navigation heartbeat also checks tracking through the compound guard.
        return self.missions.go(self.mid,p['x'],p['y'],p['yaw'])
    def support(self):
        self.permit(self.mid);self.hold()
        if self.destination_pose is None:raise ValueError('Нет принятого места доставки')
        self.drop_zone=placement_zone(self.settings['drop_zone'],self.destination_pose,self.missions.maps.pose())
        return self._xyz(self.drop_zone['center_xyz'],self.settings['close_deg'])
    def release(self):return self._xyz(self.drop_zone['center_xyz'],self.settings['open_deg'])
    def withdraw(self):return self._xyz((np.asarray(self.drop_zone['center_xyz'])+[0,0,.08]).tolist(),self.settings['open_deg'])
    def verify_place(self, before):return verify_place(before,self.observe(),self.drop_zone,True)
    def _end(self, mid, state, details):
        if self.mid==mid:
            search_id=self.search_id;self.mid=None;self.tracking=False;self.search_id=None
            operations=[self.trajectory.stop,lambda:self.missions.finish(mid,state,details),self.vision.stop]
            if search_id:operations.append(lambda:self.finder.cancel(search_id))
            errors=[]
            for operation in operations:
                try:operation()
                except Exception as exc:errors.append(str(exc))
            if errors:raise ValueError('Ошибки завершения доставки: '+'; '.join(errors))
    def stop(self, mid):self._end(mid,'cancelled',dict(reason='Delivery cancelled'))
    def finish(self, mid, success):self._end(mid,'succeeded' if success else 'failed',dict(delivery_verified=success))
