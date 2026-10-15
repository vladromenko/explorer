"""Observed, bounded manipulation primitives for the shared task executor.

Only rendered depth, camera/arm kinematics and odometry guide actions. Scene
truth remains in the independent outcome evaluator and never enters this port.
These geometric candidates do not establish semantic identity of a real sock.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .environment import PERIOD


class SimulatorSkillPort:
    def __init__(self,env,folder,cancelled=lambda:False,progress=lambda value:None):
        self.env=env;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.cancelled=cancelled;self.progress=progress;self.sequence=0
        self.target=None;self.destination=None;self.grasp_xyz=None;self.before_hold=None;self.release_xyz=None
        self.destination_views=[];self.surveys=[]
        self.transport_poses=[]
        self.frames=[];self.handle=(self.folder/'commands.jsonl').open('w')

    def check(self,ctx=None):
        if self.cancelled():raise InterruptedError('Проверка навыков остановлена')
        if self.env.last_error:raise ValueError(self.env.last_error)
        if ctx is not None:ctx.check()

    def tick(self,arm,base=(0.,0.,0.),ctx=None):
        self.check(ctx)
        action=np.r_[arm,base]
        self.handle.write(json.dumps(dict(simulation_time=float(self.env.data.time),
            phase=self.env.phase,command=action.tolist()))+'\n')
        self.env.step(action)
        if self.env.steps%5==0:self.progress(dict(phase=self.env.phase,
            simulation_time=float(self.env.data.time),command=self.env.command.tolist()))
        self.check(ctx)

    def move(self,xyz,gripper,ctx,downward=True,duration=1.5):
        goal=self.env.solve_ik(xyz,self.env.command[:5],gripper=gripper,downward=downward)
        return self.move_angles(goal,ctx,duration)

    def move_angles(self,goal,ctx,duration=1.5,*,motion_check=None):
        start=self.env.command[:6].copy()
        count=max(round(duration/PERIOD),round(np.max(abs(goal-start)/self.env.velocity_limit)/PERIOD)+10)
        for n in range(count):
            u=(n+1)/count;blend=u*u*u*(10.-15.*u+6.*u*u)
            self.tick(start+(goal-start)*blend,ctx=ctx)
            if motion_check is not None:motion_check()
        for _ in range(12):
            self.tick(goal,ctx=ctx)
            if motion_check is not None:motion_check()
        return goal

    def look_at(self,point,ctx,*,gripper=None,distance_m=.24,min_camera_height_m=.12):
        """Aim the calibrated wrist camera at an already observed location."""
        from .inspection_planning import plan_camera_view,CameraViewUnavailable
        from .manipulation_planning import hand_rotation
        if self.before_hold is not None and not hasattr(self,'hold_rotation'):
            raise CameraViewUnavailable('Нет исходной ориентации подтверждённого захвата')
        held_reference=self.hold_rotation.T@np.array([0.,0.,1.]) if self.before_hold is not None else None
        plan=plan_camera_view(self.env,point,
            gripper=float(self.env.command[5] if gripper is None else gripper),distance_m=distance_m,
            min_camera_height_m=min_camera_height_m,check=lambda:self.check(ctx),
            maximum_hand_tilt_deg=2. if held_reference is not None else None,
            reference_hand_gravity=held_reference)
        def check_tilt():
            gravity=hand_rotation(self.env).T@np.array([0.,0.,1.])
            tilt=float(np.arccos(np.clip(held_reference@gravity,-1.,1.)))
            if not np.isfinite(tilt) or tilt>np.deg2rad(3.):
                raise CameraViewUnavailable('Во время обзора превышен наклон удерживаемого предмета')
        self.move_angles(plan.command,ctx,motion_check=check_tilt if held_reference is not None else None)
        # Finite torque and joint tracking can lower a physically settled
        # camera below the kinematic prediction. Do not silently accept it.
        camera=self.env.model.camera('wrist_depth').id
        if float(self.env.data.cam_xpos[camera,2])<min_camera_height_m-.008:
            raise CameraViewUnavailable('После движения камера не достигла рассчитанной высоты обзора')
        if held_reference is not None:check_tilt()

    def observe(self,ctx):
        from src.task_executor import Observation
        from .rgbd_observation import observe_environment
        self.check(ctx);self.sequence+=1
        data=observe_environment(self.env)
        self.frames.append(data)
        return Observation(sequence=self.sequence,captured_at=time.monotonic(),
            base_revision=ctx.base_revision,data=data)

    def scene(self,ctx):
        return self.observe(ctx).data

    def estimate(self,ctx,kind):
        from .rgbd_observation import estimate_target,ObservationRefused
        frame=self.scene(ctx)
        try:item=estimate_target(frame,kind,now_simulation_time=float(self.env.data.time))
        except ObservationRefused:
            noun='Корзина' if kind=='container' else 'Предмет'
            raise ObservationRefused(noun+' не выделена однозначно по свежему изображению глубины') from None
        return dict(item,frame_id=frame['frame_id'],simulation_time=frame['simulation_time'])

    def inspect(self,ctx):
        base=np.asarray(self.env.base_pose())
        direction=np.array([math.cos(base[2]),math.sin(base[2])])
        self.move([*(base[:2]+.23*direction),.15],30.,ctx)

    def survey_destination(self,ctx,*,max_views=8):
        """A finite robot-relative survey; partial geometry only guides camera aim.

        Cached world geometry is an aiming hint. At least one new complete
        depth frame must agree with a separate complete camera viewpoint before
        the destination can be used. Gripper state is preserved throughout.
        """
        from .rgbd_observation import combine_container_views,ObservationRefused
        self.env.phase='skill_find_destination'
        pose=np.asarray(self.env.base_pose());direction=np.array([math.cos(pose[2]),math.sin(pose[2])])
        lateral=np.array([-direction[1],direction[0]])
        aims=[]
        if self.destination is not None:
            aims.append((np.asarray(self.destination['center']),.45))
        for forward,side,height in ((.40,.24,.15),(.40,-.24,.15),(.50,.35,.20),
                                    (.50,-.35,.20),(.35,.28,.25),(.35,-.28,.25)):
            aims.append((np.r_[pose[:2]+forward*direction+side*lateral,height],.40))
        visited=[];notes=[];new_complete=0
        while aims and len(visited)<max_views:
            self.check(ctx)
            point,distance=aims.pop(0)
            if any(np.linalg.norm(point-old[0])<.025 and abs(distance-old[1])<.025 for old in visited):
                continue
            visited.append((point.copy(),distance))
            try:
                self.look_at(point,ctx,distance_m=distance,min_camera_height_m=.30)
                frame=self.scene(ctx)
            except ValueError as error:
                notes.append(dict(aim=point.tolist(),reason=str(error)));continue
            complete=frame.get('containers',[])
            notes.append(dict(aim=point.tolist(),frame_id=frame['frame_id'],complete_candidates=len(complete)))
            if len(complete)==1:
                self.destination_views.append(dict(candidate=complete[0],frame_id=frame['frame_id'],
                    simulation_time=frame['simulation_time'],
                    camera_position=[row[3] for row in frame['optical_to_world'][:3]]))
                new_complete+=1
                try:
                    box=combine_container_views(self.destination_views)
                    if box['frame_id']==frame['frame_id'] and float(box['uncertainty_m'])<=.02:
                        self.destination=box
                        self.surveys.append(dict(outcome='success',views=notes,destination=box))
                        return dict(outcome='success',destination=box,evidence=dict(source='bounded_rgbd_survey',
                            complete_views=len(self.destination_views),new_complete_views=new_complete))
                except ObservationRefused:pass
                aims.insert(0,(np.asarray(complete[0]['center']),.45))
            elif len(complete)>1:
                self.surveys.append(dict(outcome='unknown',views=notes,reason='Несколько полных корзин'))
                return dict(outcome='unknown',reason='Камера видит несколько корзин; назначение неоднозначно')
            elif len(frame.get('container_hypotheses',[]))==1:
                # A wall fragment may steer a new camera view, never a release.
                hint=frame['container_hypotheses'][0]
                aims.insert(0,(np.asarray(hint['center']),.45))
        self.surveys.append(dict(outcome='unknown',views=notes))
        needs_new_view=self.before_hold is not None and new_complete==0
        reason=('Не найден обзор корзины с сохранением удержания; нужна другая точка обзора'
                if needs_new_view else 'За ограниченный обзор не получены два полных согласованных изображения отверстия корзины')
        return dict(outcome='unknown',reason=reason,
                    evidence=dict(source='bounded_rgbd_survey',views=len(notes),new_complete_views=new_complete,
                        held_object_motion_guard=self.before_hold is not None,needs_new_viewpoint=needs_new_view))

    def find(self,ctx):
        self.env.phase='skill_find';self.inspect(ctx)
        self.target=self.estimate(ctx,'object')
        survey=self.survey_destination(ctx)
        if survey['outcome']!='success':return survey
        self.inspect(ctx)
        self.target=self.estimate(ctx,'object')
        return dict(outcome='success',target=self.target,
            evidence=dict(source='rendered_rgbd',semantic_identity='unverified',single_target_assumption=True))

    def approach(self,ctx):
        import mujoco
        import copy
        self.env.phase='skill_approach'
        if self.target is None:return dict(outcome='unknown',reason='Нет наблюдаемой цели')
        target=np.asarray(self.target['center'])
        pose=np.asarray(self.env.base_pose());direction=np.array([math.cos(pose[2]),math.sin(pose[2])])
        destination=None
        # Select a reachable stance with the robot's own kinematics before
        # driving. Only the measured target supplies environment coordinates.
        branch=copy.copy(self.env);branch.data=mujoco.MjData(self.env.model)
        branch.data.qpos[:]=self.env.data.qpos
        address=self.env.model.joint('base_free').qposadr[0]
        grasp=[*target[:2],max(.017,float(target[2])+.010)]
        for stand_off in (.20,.22,.24,.26,.18):
            self.check(ctx)
            candidate=target[:2]-stand_off*direction
            branch.data.qpos[address:address+2]=candidate
            mujoco.mj_forward(branch.model,branch.data)
            try:
                branch.solve_ik(grasp,gripper=30.)
                branch.solve_ik(grasp,gripper=170.)
                try:branch.solve_ik([*target[:2],.18],gripper=170.)
                except ValueError:branch.solve_ik([*target[:2],.18],gripper=170.,downward=False)
                destination=candidate;break
            except ValueError:continue
        if destination is None:return dict(outcome='failure',reason='Для измеренной точки не найден достижимый захват')
        travelled=0.
        for _ in range(160):
            pose=np.asarray(self.env.base_pose());delta=destination-pose[:2]
            if np.linalg.norm(delta)<.008:break
            if travelled>.22:return dict(outcome='failure',reason='Превышен предел локального подъезда',base_moved=True)
            ranges=self.env.lidar_scan()['ranges']
            # Stop on a near scan return rather than assuming a clear room.
            if any(r is not None and r<.12 for r in ranges):
                return dict(outcome='unknown',reason='Лидар не подтвердил свободный проход',base_moved=travelled>0)
            c,s=math.cos(pose[2]),math.sin(pose[2]);body=np.array([[c,s],[-s,c]])@delta
            speed=np.clip(body*1.2,-.06,.06)
            self.tick(self.env.command[:6],(*speed,0.),ctx);travelled+=float(np.linalg.norm(speed))*PERIOD
        else:return dict(outcome='failure',reason='Подъезд не сошёлся',base_moved=travelled>0)
        for _ in range(12):self.tick(self.env.command[:6],ctx=ctx)
        return dict(outcome='success',base_moved=travelled>0,evidence=dict(local_travel_m=travelled))

    def reobserve(self,ctx):
        from .rgbd_observation import ObservationRefused
        self.env.phase='skill_reobserve'
        new=None;last_error=None
        # A base approach changes the wrist view. Reframe the last observed
        # location with bounded camera motions instead of accepting a crop.
        for reframe in (False,True):
            if reframe and self.target is not None:
                self.look_at(self.target['center'],ctx)
            try:new=self.estimate(ctx,'object');break
            except ObservationRefused as error:last_error=error
        if new is None:return dict(outcome='unknown',reason='Не удалось получить полный свежий обзор: '+str(last_error))
        if self.target and np.linalg.norm(np.asarray(new['center'])-self.target['center'])>.10:
            return dict(outcome='unknown',reason='Свежая цель не совпадает с предыдущим наблюдением')
        self.target=new;self.grasp_xyz=np.asarray(new['center'],float)
        return dict(outcome='success',target=new,evidence=dict(frame_id=new['frame_id']))

    def pregrasp(self,ctx):
        self.env.phase='skill_pregrasp'
        if self.grasp_xyz is None:return dict(outcome='unknown',reason='Нет свежей точки захвата')
        self.move([*self.grasp_xyz[:2],max(.10,float(self.grasp_xyz[2])+.08)],30.,ctx)
        return dict(outcome='success')

    def grasp(self,ctx):
        self.env.phase='skill_grasp'
        # The measured visible surface supplies height; no cloth-node truth.
        # Geometric pad-centre offset above the visible cloth surface. It is
        # explicitly estimated, pending the same real gripper calibration as
        # the existing simulator contact geometry; it is not cloth truth.
        xyz=self.grasp_xyz.copy();xyz[2]=max(.017,float(xyz[2])+.010)
        self.move(xyz,30.,ctx)
        self.move(xyz,170.,ctx,duration=2.)
        return dict(outcome='success',evidence=dict(command_only=True,holding_not_yet_confirmed=True))

    def lift(self,ctx):
        self.env.phase='skill_lift'
        xyz=[*self.grasp_xyz[:2],.18]
        try:goal=self.env.solve_ik(xyz,self.env.command[:5],gripper=170.)
        except ValueError:goal=self.env.solve_ik(xyz,self.env.command[:5],gripper=170.,downward=False)
        self.move_angles(goal,ctx)
        return dict(outcome='success',evidence=dict(command_only=True))

    def verify_hold(self,ctx,probe=True):
        self.env.phase='skill_verify_hold';frame=self.scene(ctx)
        jaw=np.asarray(self.env.jaw_center())
        reference=(self.target or {}).get('appearance',{}).get('rgb_median')
        def associated(candidate):
            appearance=candidate.get('appearance',{}).get('rgb_median')
            # A held cloth may be cropped and much darker in the wrist's
            # shadow. Whole-object localisation confidence is not a tracking
            # proof; require a visible non-self patch, chromatic agreement and
            # then a separate measured motion probe below.
            return (reference is not None and appearance is not None
                and candidate.get('evidence',{}).get('self_occlusion_not_resolved') is False
                and candidate.get('evidence',{}).get('visible_pixels',0)>=100
                and float(candidate.get('uncertainty_m',1.))<=.02
                and np.linalg.norm(np.asarray(appearance)/max(.05,sum(appearance))-
                    np.asarray(reference)/max(.05,sum(reference)))<.08)
        matches=[c for c in frame.get('object_candidates',[]) if np.linalg.norm(np.asarray(c['center'])-jaw)<.08 and associated(c)]
        if len(matches)!=1:return dict(outcome='unknown',reason='Камера не подтверждает удержание предмета у пальцев')
        if float(matches[0]['bounds_min'][2])<.045:
            return dict(outcome='failure',reason='Предмет не поднят над полом')
        first=matches[0]
        if not probe:
            if self.before_hold is None or not hasattr(self,'hold_jaw') or not hasattr(self,'hold_rotation'):
                return dict(outcome='unknown',reason='Нет исходного подтверждения удержания')
            from .manipulation_planning import hand_rotation
            # The held patch rotates with the hand during a camera survey.
            # Compare both offsets in their own hand frames, never as fixed
            # world vectors. Keep the same 25 mm slip gate.
            relative=hand_rotation(self.env).T@(np.asarray(first['center'])-jaw)
            previous=self.hold_rotation.T@(np.asarray(self.before_hold['center'])-self.hold_jaw)
            displacement=float(np.linalg.norm(relative-previous))
            if not np.isfinite(displacement) or displacement>.025:
                return dict(outcome='unknown',reason='Положение предмета относительно захвата изменилось')
            return dict(outcome='success',evidence=dict(confirmed=True,source='tracked_rgbd_geometry',
                frame_id=frame['frame_id'],comparison_frame='arm5',relative_displacement_m=displacement))
        from .manipulation_planning import plan_hold_probe,hand_rotation
        from scipy.spatial.transform import Rotation
        probe_plan=plan_hold_probe(self.env,check=lambda:self.check(ctx))
        self.move_angles(probe_plan.command,ctx)
        second=self.scene(ctx);new_jaw=np.asarray(self.env.jaw_center())
        candidates=[c for c in second.get('object_candidates',[]) if associated(c)
            and np.linalg.norm(np.asarray(c['center'])-new_jaw)<.08]
        if len(candidates)!=1:return dict(outcome='unknown',reason='Повторный кадр не подтверждает удержание')
        moved=np.asarray(candidates[0]['center'])-first['center'];hand_moved=new_jaw-jaw
        rotation_change=Rotation.from_matrix(probe_plan.reference_rotation.T@hand_rotation(self.env)).magnitude()
        if (second['frame_id']==frame['frame_id'] or np.linalg.norm(hand_moved)<.015
            or float(candidates[0]['bounds_min'][2])<.045 or rotation_change>np.deg2rad(3.)
            or np.linalg.norm(moved-hand_moved)>.015):
            return dict(outcome='unknown',reason='Предмет не подтверждён совместным движением с захватом')
        self.before_hold=candidates[0]
        self.hold_jaw=new_jaw
        self.hold_rotation=hand_rotation(self.env)
        return dict(outcome='success',evidence=dict(confirmed=True,source='two_frame_rgbd_motion_association',
            frames=[frame['frame_id'],second['frame_id']],probe_direction=probe_plan.direction,
            measured_translation_m=float(np.linalg.norm(hand_moved)),measured_rotation_deg=float(np.rad2deg(rotation_change)),
            semantic_identity='unverified_single_geometric_target'))

    def verify_stationary_hold(self,ctx):
        """Observe a full second of stationary holding before chassis motion."""
        frames=[]
        for _ in range(4):
            for _ in range(5):self.tick(self.env.command[:6],ctx=ctx)
            held=self.verify_hold(ctx,probe=False)
            if held['outcome']!='success':return held
            frame=held.get('evidence',{}).get('frame_id')
            if not frame or frame in frames:
                return dict(outcome='unknown',reason='Нет свежих кадров выдержки удержания')
            frames.append(frame)
        return dict(outcome='success',evidence=dict(confirmed=True,
            source='stationary_multi_frame_rgbd_hold',duration_s=20*PERIOD,frames=frames))

    def carry(self,ctx):
        self.env.phase='skill_find_destination'
        # Missing or partial container geometry cannot be replaced by the
        # simulated bin pose or an assumed pre-recorded destination.
        survey=self.survey_destination(ctx,max_views=4)
        if survey['outcome']!='success':return survey
        held=self.verify_hold(ctx,probe=False)
        if held['outcome']!='success':return held
        box=self.destination
        if float(box['uncertainty_m'])>.02:return dict(outcome='unknown',reason='Граница корзины измерена недостаточно точно')
        from .manipulation_planning import plan_transport_pose,transport_pose_satisfied,hand_rotation
        from scipy.spatial.transform import Rotation
        rim=float(box['bounds_max'][2])
        transport=plan_transport_pose(self.env,max(.18,rim+.09),check=lambda:self.check(ctx))
        transport_evidence=dict(simulation_time=float(self.env.data.time),retained=transport.retained,
            before_jaw=np.asarray(self.env.jaw_center()).tolist(),command=transport.command.tolist(),
            minimum_height_m=transport.minimum_height_m,maximum_reach_m=transport.maximum_reach_m)
        self.transport_poses.append(transport_evidence)
        def check_transport_rotation():
            angle=Rotation.from_matrix(transport.reference_rotation.T@hand_rotation(self.env)).magnitude()
            if not np.isfinite(angle) or angle>np.deg2rad(3.):
                raise ValueError('Во время подготовки к переносу изменился наклон захвата')
        if not transport.retained:
            self.move_angles(transport.command,ctx,motion_check=check_transport_rotation)
        rotation=Rotation.from_matrix(transport.reference_rotation.T@hand_rotation(self.env)).magnitude()
        transport_evidence.update(after_jaw=np.asarray(self.env.jaw_center()).tolist(),
            measured_rotation_deg=float(np.rad2deg(rotation)))
        if (not transport_pose_satisfied(self.env,transport.minimum_height_m,transport.maximum_reach_m)
            or rotation>np.deg2rad(3.)):
            return dict(outcome='unknown',reason='Измеренная транспортная поза не сохранила высоту, компактность или ориентацию захвата')
        held=self.verify_stationary_hold(ctx)
        transport_evidence['stationary_hold']=held
        if held['outcome']!='success':return held
        self.env.phase='skill_carry'
        half=(np.asarray(box['bounds_max'])-box['bounds_min'])/2
        destination=np.asarray(box['center'])[:2]-[float(half[0])+.17,0.]
        moved=False
        for n in range(400):
            pose=np.asarray(self.env.base_pose());delta=destination-pose[:2]
            if np.linalg.norm(delta)<.01:break
            if np.linalg.norm(delta)>.8:return dict(outcome='unknown',reason='Для дальней корзины нужна карта и маршрут',base_moved=moved)
            ranges=self.env.lidar_scan()['ranges']
            if any(r is not None and r<.12 for r in ranges):
                return dict(outcome='unknown',reason='Проход к корзине не подтверждён лидаром',base_moved=moved)
            c,s=math.cos(pose[2]),math.sin(pose[2]);body=np.array([[c,s],[-s,c]])@delta
            self.tick(self.env.command[:6],(*np.clip(body,-.04,.04),0.),ctx);moved=True
            if n%20==0:
                held=self.verify_hold(ctx,probe=False)
                if held['outcome']!='success':return dict(held,base_moved=moved)
        else:return dict(outcome='failure',reason='Перенос не сошёлся за ограниченное время',base_moved=moved)
        for _ in range(12):self.tick(self.env.command[:6],ctx=ctx)
        return dict(outcome='success',base_moved=moved,evidence=dict(destination=box,route='local_rgbd_odometry_lidar'))

    def support(self,ctx):
        self.env.phase='skill_support'
        previous=self.destination
        survey=self.survey_destination(ctx,max_views=4)
        if survey['outcome']!='success':return survey
        box=self.destination
        if float(box['uncertainty_m'])>.02:return dict(outcome='unknown',reason='Граница корзины перед отпусканием недостаточно точна')
        if previous and np.linalg.norm(np.asarray(box['center'])-previous['center'])>.06:
            return dict(outcome='unknown',reason='Корзина не совпала после подъезда')
        self.destination=box
        lo,hi=np.asarray(box['bounds_min']),np.asarray(box['bounds_max'])
        if min(hi[:2]-lo[:2])<.10:return dict(outcome='unknown',reason='Отверстие корзины слишком мало или не видно')
        self.release_xyz=np.array([lo[0]+min(.09,(hi[0]-lo[0])/2),(lo[1]+hi[1])/2,hi[2]+.035])
        self.move(self.release_xyz,170.,ctx,downward=False)
        return dict(outcome='success',evidence=dict(frame_id=box['frame_id'],rim_height_m=float(hi[2])))

    def release(self,ctx):
        self.env.phase='skill_release'
        if self.release_xyz is None:return dict(outcome='unknown',reason='Нет проверенной точки отпускания')
        held=self.verify_hold(ctx,probe=False)
        if held['outcome']!='success':return held
        self.env.phase='skill_release'
        self.before_release=self.scene(ctx)
        self.release_jaw=np.asarray(self.env.jaw_center()).copy()
        self.move(self.release_xyz,30.,ctx,downward=False,duration=2.)
        return dict(outcome='success',evidence=dict(command_only=True))

    def withdraw(self,ctx):
        self.env.phase='skill_withdraw'
        if self.release_xyz is None:return dict(outcome='unknown',reason='Нет точки отвода')
        self.move(self.release_xyz+[-.035,0.,.07],30.,ctx,downward=False)
        for _ in range(40):self.tick(self.env.command[:6],ctx=ctx)
        return dict(outcome='success')

    def verify_place(self,ctx):
        from .placement_evidence import verify_released_target
        self.env.phase='skill_verify_place'
        if self.destination is None:return dict(outcome='unknown',reason='Нет наблюдаемой корзины')
        if not hasattr(self,'before_release') or self.before_hold is None:
            return dict(outcome='unknown',reason='Нет последовательности наблюдаемого удержания и отпускания')
        first=self.scene(ctx)
        for _ in range(10):self.tick(self.env.command[:6],ctx=ctx)
        second=self.scene(ctx)
        return verify_released_target(first,second,destination=self.destination,
            held_reference=self.before_hold,release_xyz=self.release_xyz,jaw_xyz=self.env.jaw_center(),
            before_release=self.before_release,release_jaw_xyz=self.release_jaw)

    def stop_motion(self):
        """Release base commands even during cancellation, physics or I/O errors."""
        self.env.pending_command[6:]=0.;self.env.command[6:]=0.
        self.env.data.ctrl[self.env.wheel_ids]=0.

    def close(self):
        self.stop_motion()
        try:self.handle.close()
        finally:
            (self.folder/'destination-surveys.json').write_text(json.dumps(self.surveys,ensure_ascii=False,indent=2))
            (self.folder/'transport-poses.json').write_text(json.dumps(self.transport_poses,ensure_ascii=False,indent=2))
            (self.folder/'observations.json').write_text(json.dumps(self.frames,ensure_ascii=False,indent=2))

    def operations(self):
        return {name:getattr(self,name) for name in ('find','approach','reobserve','pregrasp','grasp',
            'lift','verify_hold','carry','support','release','withdraw','verify_place')}
