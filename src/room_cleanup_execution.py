"""Explicit observed ROSMASTER room trials, isolated from ROS/model runtimes.

No constructor, package import, status request or preflight can move hardware.
The existing mission/arm/STOP owners and a two-second observer lease remain in
charge. A room package cannot inherit admission from a local grasp report.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from observed_skill_execution import ObservedSkillExecution,FactoryObservedTransport
from room_cleanup_rpc import RobotRPC,save


class RoomCleanupExecution(ObservedSkillExecution):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from simulator.room_policy import RUNTIME_FILES
        self.loaded_runtime={name:hashlib.sha256((self.root/name).read_bytes()).hexdigest()
            for name in RUNTIME_FILES if (self.root/name).is_file()}

    def candidate(self,identifier):
        if not isinstance(identifier,str) or len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid full-room candidate')
        from simulator.room_transfer import validate_room_folder
        path=self.root/'data/room-cleanup-candidates'/identifier
        return path,validate_room_folder(path,check_runtime=True)

    def preflight(self,identifier):
        reasons=[];checked=None
        try:
            _,checked=self.candidate(identifier)
            if checked['runtime_files']!=self.loaded_runtime:reasons.append('После установки этого пакета web-служба ещё не перезапущена')
            reasons+=self.robot.blockers()
            settings=self.robot._settings()
            if abs(settings['open_deg']-30)>.5 or abs(settings['close_deg']-170)>.5:
                reasons.append('Нужна принятая калибровка раскрытия: модель 30–170° не совпадает с захватом робота')
            plane=np.asarray(settings.get('floor_plane_base'),float)
            if plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:2])>.03 or plane[2]<.999:
                reasons.append('Нужна принятая плоскость пола для полного сценария')
            status=json.loads((self.root/'data/status.json').read_text())
            if time.time()-status['at']>1 or status.get('stop_latched') is not False or status.get('mode')!='AUTONOMOUS':
                reasons.append('Выберите автономный режим и явно снимите STOP перед наблюдаемой проверкой')
            if self.robot.missions.status().get('active') or self.robot.arm.status().get('busy') or self.robot.trajectory.status().get('busy'):
                reasons.append('Другой исполнитель уже использует робота')
            pose=self.robot.missions.maps.pose()
            if pose.get('localization_verified') is not True:reasons.append('Локализация на карте ещё не принята')
            self.robot.missions.frontiers()
            scans=self.scans()
            if len(scans)!=2 or any(time.monotonic()-s['acquired_monotonic']>.6 for s in scans):reasons.append('Нужны два свежих полных скана лидара')
            model_lock=json.loads((self.root/'config/grounding-model.json').read_text())
            for name,digest in model_lock['files'].items():
                p=self.model_root/'models/grounding-dino-tiny'/name
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:reasons.append('Не установлены проверенные веса зрения: '+name)
            if not (self.root/'.venv-learning/bin/python').is_file():reasons.append('Не установлена среда ACT на Jetson')
            if not (self.root/'data/env-vision/bin/python').is_file():reasons.append('Не установлена среда распознавания')
            runtime=json.loads((self.root/'data/room-learning-runtime.json').read_text())
            if (runtime.get('ready') is not True or runtime.get('hardware_accessed') is not False
                    or runtime.get('device')!='cuda' or runtime.get('model_sha256')!=checked['bundle']['checkpoint_sha256']['model.safetensors']
                    or runtime.get('mission_contract_sha256')!=checked['mission_contract_sha256']):
                reasons.append('Проверка загрузки и расчёта именно этого ACT-пакета на Jetson не пройдена')
        except (OSError,ValueError,KeyError,TypeError) as error:reasons.append(str(error))
        return dict(ready=not reasons,candidate=identifier,blocked_by=list(dict.fromkeys(reasons)),
            simulation_ready=bool(checked),physical_ready=False,observed_trial_required=True)

    def stop(self):
        if getattr(self,'session_folder',None) is not None:(self.session_folder/'cancel.request').touch()
        return super().stop()

    def _run(self,identifier):
        folder=self.root/'data/room-cleanup-runs'/self.session;folder.mkdir(parents=True)
        self.session_folder=folder;child=None;drive=None
        try:
            self._lease();path,checked=self.candidate(identifier)
            self.expected_contract=checked['mission_contract_sha256']
            self.robot.settings=self.robot._settings();self.robot.owner_check=self._lease
            self.robot.mid=self.session;self.robot.boot=self.robot.arm.reference()['boot_id'];self.robot.tracking=False
            self.robot.missions.begin_compound(self.session,'mobile_policy',1200,self._lease)
            self.robot.hold();drive=threading.Thread(target=self._drive,daemon=True);drive.start()
            save(folder/'config.json',dict(package=str(path),session=self.session,
                mesh_dir=str(path/'assets/meshes'),semantic_root=str(self.model_root)))
            transport=FactoryRoomTransport(self);rpc=RobotRPC(transport,folder,self.session)
            with (folder/'worker.log').open('ab') as log:
                child=subprocess.Popen([str(self.root/'.venv-learning/bin/python'),str(self.root/'bin/room-cleanup-worker.py'),str(folder)],
                    cwd=self.root,stdout=log,stderr=log,start_new_session=True)
            deadline=time.monotonic()+1200
            while child.poll() is None:
                self._lease()
                if time.monotonic()>deadline:raise TimeoutError('Полный сценарий превысил конечный бюджет')
                rpc.serve_next()
                if (folder/'progress.json').exists():
                    progress=json.loads((folder/'progress.json').read_text())
                    self.state.update(phase=progress.get('stage',progress.get('phase','running')),progress=progress)
                time.sleep(.01)
            if child.returncode or not (folder/'result.json').is_file():raise ValueError('Исполнитель завершился с ошибкой; журнал сохранён')
            result=json.loads((folder/'result.json').read_text())
            self.state.update(phase='completed' if result['state']=='succeeded' else 'stopped',result=result,
                reason=result.get('reason'),visual_outcome_confirmed=result.get('placed_verified') is True,
                physical_ready=False,physical_success_verified=False)
        except Exception as error:self.state.update(phase='stopped',reason=str(error),physical_ready=False)
        finally:
            self.stop()
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(5)
                except subprocess.TimeoutExpired:child.kill();child.wait(5)
            if drive is not None:drive.join(1)
            try:
                if self.robot.mid==self.session:
                    self.robot.missions.finish(self.session,'succeeded' if self.state.get('phase')=='completed' else 'interrupted',
                        dict(reason=self.state.get('reason')));self.robot.mid=None
            finally:
                try:save(folder/'owner-result.json',self.state)
                finally:self.lock.release()


class FactoryRoomTransport(FactoryObservedTransport):
    def _base(self,status):
        # The inherited local-grasp port uses wheel odometry. A Nav2 mission
        # must explicitly use map TF; odom coordinates are not map goals.
        pose=self.robot.missions.maps.pose()
        if pose.get('localization_verified') is not True or pose['age_s']>.075:raise ValueError('Нет свежей принятой локализации для RGB-D и карты')
        def transform(value):
            yaw=value['yaw'];c,s=np.cos(yaw),np.sin(yaw);t=np.eye(4)
            t[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];t[:2,3]=[value['x'],value['y']];return t
        mapped=transform(pose);odom=transform(status['raw_pose']);correction=mapped@np.linalg.inv(odom)
        previous=getattr(self,'map_odom_anchor',None)
        if previous is not None:
            delta=np.linalg.inv(previous)@correction
            if np.linalg.norm(delta[:2,3])>.05 or abs(np.arctan2(delta[1,0],delta[0,0]))>.05:
                raise ValueError('Локализация изменилась относительно одометрии; нужен новый наблюдаемый запуск')
        else:self.map_odom_anchor=correction
        if self.origin is None:self.origin=np.linalg.inv(mapped)
        plane=self.robot.settings['floor_plane_base'];mapped[2,3]=plane[3]/plane[2]
        self.map_acquired=time.monotonic()-pose['age_s']
        return self.origin@mapped

    def state(self):
        state=super().state();state['acquired_monotonic']=min(state['acquired_monotonic'],self.map_acquired)
        return state

    def preflight(self,contract):
        self.permit()
        if contract!=self.owner.expected_contract:raise ValueError('Полный сценарий не соответствует проверенному пакету')
        return dict(ready=True,physical_ready=False)

    def plan(self,goal):
        self.permit();self.state()
        # ObservedRobotEnv rebases map coordinates to the initial robot pose.
        anchor=np.linalg.inv(self.origin);xy=anchor@np.array([goal['x'],goal['y'],0.,1.])
        try:plan=self.robot.missions.maps.preview(float(xy[0]),float(xy[1]))
        except ValueError:return dict(reachable=False)
        if not plan.get('points'):return dict(reachable=False)
        points=np.asarray(plan['points'],float)
        points=(self.origin@np.c_[points,np.zeros(len(points)),np.ones(len(points))].T).T[:,:2]
        return dict(reachable=True,points=points.tolist(),cost=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))

    def frontiers(self):
        self.permit();self.state();value=self.robot.missions.frontiers();result=[]
        yaw=float(np.arctan2(self.origin[1,0],self.origin[0,0]))
        for point in value['candidates']:
            xy=self.origin@np.array([point['x'],point['y'],0.,1.])
            result.append(dict(x=float(xy[0]),y=float(xy[1]),yaw=float(point.get('yaw',0.)+yaw)))
        return result

    def apply(self,action,*,reference,period,action_kind='view'):
        from mobile_policy_contract import bound_action
        self.permit();current=np.asarray(self.state()['servo_deg'])
        if np.max(abs(current-np.asarray(reference)[:6]))>.6:raise ValueError('Устаревшая исходная команда руки')
        bounded=bound_action(reference,action)
        issued=np.r_[bounded['issued_joint_goal_deg'],bounded['issued_body_velocity']]
        if action_kind=='policy' and not np.allclose(action,issued,atol=1e-5):raise ValueError('Действие отличается от ограничений полного сценария')
        if action_kind not in ('policy','view') or not 30<=action[5]<=170:raise ValueError('Недопустимый контактный диапазон')
        return super().apply(action,reference=reference,period=period)
