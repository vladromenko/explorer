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
    def __init__(self,env,folder,cancelled=lambda:False,progress=lambda value:None,*,parameters=None,repair_program=None,experience=None):
        from .skill_parameters import parameters as checked_parameters
        from .failure_repair import program
        self.controller_program=program(repair_program);self.experience=experience or {}
        if set(self.experience)-{'sha256','ik_seeds'}:
            raise ValueError('Actor historical input must contain only audited robot joint priors')
        self.parameters=checked_parameters(parameters)
        self.env=env;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.cancelled=cancelled;self.progress=progress;self.sequence=0;self.base_motion_sequence=0
        self.target=None;self.destination=None;self.grasp_xyz=None;self.before_hold=None;self.release_xyz=None
        self.destination_views=[];self.surveys=[]
        self.transport_poses=[]
        self.grasp_recoveries=[];self.grasp_failures=0
        self.repair_events=[]
        self.env.observed_ik_seeds=(self.experience.get('ik_seeds',[]) if self.repair_enabled('history_ik_seeds') else [])
        self.env.observed_grasp_axis=None
        self.depth_records=[];self.depth_manifest=[]
        (self.folder/'depth').mkdir()
        self.frames=[];self.handle=(self.folder/'commands.jsonl').open('w')

    def repair_enabled(self,name):return name in self.controller_program['operators']

    def move_pick_feedback(self,point,gripper,ctx):
        """Bounded own-FK feedback; no task truth or unverified contact input."""
        from .grasp_planning import solve_pick_ik
        target=np.asarray(point,float);aim=target.copy()
        for attempt in range(3):
            self.check(ctx);goal=solve_pick_ik(self.env,aim,gripper,check=lambda:self.check(ctx))
            self.move_angles(goal,ctx)
            error=target-np.asarray(self.env.jaw_center(),float)
            self.repair_events.append(dict(operator='cartesian_feedback',attempt=attempt,
                desired=target.tolist(),observed_error_m=error.tolist(),source='own_robot_FK'))
            if np.linalg.norm(error)<=.003:return aim
            if np.linalg.norm(error)>.012:break
            correction=np.clip(aim+error-target,-.008,.008)
            # Never compensate into the floor beyond the originally checked
            # pad target. The unchanged IK cone and final 3 mm gate still hold.
            aim=target+correction;aim[2]=max(target[2],aim[2])
        raise ValueError('Обратная связь не подтвердила точность захвата 3 мм')

    def check(self,ctx=None):
        if self.cancelled():raise InterruptedError('Проверка навыков остановлена')
        if self.env.last_error:raise ValueError(self.env.last_error)
        if ctx is not None:ctx.check()

    def tick(self,arm,base=(0.,0.,0.),ctx=None):
        self.check(ctx)
        action=np.r_[arm,base]
        if np.any(np.asarray(base)!=0.):self.base_motion_sequence+=1
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

    def look_at(self,point,ctx,*,gripper=None,distance_m=.24,min_camera_height_m=.12,transport_required=True,height_reserve_m=0.):
        """Aim the calibrated wrist camera at an already observed location."""
        from .inspection_planning import plan_camera_view,CameraViewUnavailable,ObservedMotionViolation,CameraViewTrackingFailure
        from .manipulation_planning import hand_rotation
        if self.before_hold is not None and not hasattr(self,'hold_rotation'):
            raise CameraViewUnavailable('Нет исходной ориентации подтверждённого захвата')
        held_reference=(self.hold_rotation.T@np.array([0.,0.,1.])
            if self.before_hold is not None and not getattr(self,'release_commanded',False) else None)
        minimum_jaw=(float(self.destination['bounds_max'][2])+.10
            if held_reference is not None and self.destination is not None and transport_required else None)
        if not np.isfinite(height_reserve_m) or not 0<=height_reserve_m<=.06:raise ValueError('Invalid camera tracking reserve')
        plan=plan_camera_view(self.env,point,
            gripper=float(self.env.command[5] if gripper is None else gripper),distance_m=distance_m,
            min_camera_height_m=min_camera_height_m+height_reserve_m,check=lambda:self.check(ctx),
            maximum_hand_tilt_deg=2. if held_reference is not None else None,
            reference_hand_gravity=held_reference,minimum_jaw_height_m=minimum_jaw)
        def check_tilt():
            gravity=hand_rotation(self.env).T@np.array([0.,0.,1.])
            tilt=float(np.arccos(np.clip(held_reference@gravity,-1.,1.)))
            if not np.isfinite(tilt) or tilt>np.deg2rad(3.):
                raise ObservedMotionViolation('Во время обзора превышен наклон удерживаемого предмета')
        self.move_angles(plan.command,ctx,motion_check=check_tilt if held_reference is not None else None)
        # Finite torque and joint tracking can lower a physically settled
        # camera below the kinematic prediction. Do not silently accept it.
        camera=self.env.model.camera('wrist_depth').id
        if float(self.env.data.cam_xpos[camera,2])<min_camera_height_m-.008:
            if (held_reference is None and float(self.env.data.cam_xpos[camera,2])>=.12
                    and self.env.jaw_center()[2]>=.06):
                raise CameraViewTrackingFailure('Свободная камера не достигла высоты обзора; захват остаётся над полом')
            raise ObservedMotionViolation('После движения камера не достигла рассчитанной высоты обзора')
        if held_reference is not None:check_tilt()
        if minimum_jaw is not None and self.env.jaw_center()[2]<minimum_jaw-.008:
            raise ObservedMotionViolation('Во время обзора захват опустился ниже высоты переноса')

    def held_yaw_view(self,point,offset_deg,ctx,*,transport_required=True):
        """Change only arm azimuth, preserving the settled high grasp posture."""
        from .inspection_planning import check_observed_box_path,ObservedMotionViolation,CameraViewUnavailable
        from .manipulation_planning import hand_rotation
        camera=self.env.model.camera('wrist_depth').id
        forward=-self.env.data.cam_xmat[camera].reshape(3,3)[:,2]
        direction=np.asarray(point,float)[:2]-self.env.data.cam_xpos[camera,:2]
        delta=math.atan2(direction[1],direction[0])-math.atan2(forward[1],forward[0])
        delta=math.atan2(math.sin(delta),math.cos(delta))
        command=self.env.command[:6].copy();command[0]+=-math.degrees(delta)+offset_deg
        if not 0.<=command[0]<=180.:raise CameraViewUnavailable('Азимут обзора вне диапазона основания руки')
        check_observed_box_path(self.env,command,self.destination,check=lambda:self.check(ctx))
        reference=self.hold_rotation.T@np.array([0.,0.,1.])
        def guard():
            gravity=hand_rotation(self.env).T@np.array([0.,0.,1.])
            if math.acos(float(np.clip(reference@gravity,-1.,1.)))>math.radians(3.):
                raise ObservedMotionViolation('При повороте обзора изменился наклон удерживаемого предмета')
            minimum=(float(self.destination['bounds_max'][2])+.09 if transport_required else .18)
            if self.env.jaw_center()[2]<minimum:
                raise ObservedMotionViolation('При повороте обзора потеряна высота переноса')
        self.move_angles(command,ctx,duration=3.,motion_check=guard)

    def held_view_baseline(self,ctx,*,transport_required=True):
        """A measured lateral shift supplies a baseline without losing reach."""
        from .observed_navigation import TranslationFeedback
        from .inspection_planning import ObservedMotionViolation
        from .manipulation_planning import hand_rotation,transport_pose_satisfied
        minimum=max(.18,float(self.destination['bounds_max'][2])+.09) if transport_required else .18
        if not transport_pose_satisfied(self.env,minimum):
            raise ObservedMotionViolation('Нет компактной высокой позы для второго ракурса')
        start=np.asarray(self.env.base_pose())[:2]
        away=start-np.asarray(self.destination['center'])[:2];away/=np.linalg.norm(away)
        # Retreating again after the destination approach moved the robot
        # beyond the arm's placement workspace. A lateral Mecanum baseline
        # supplies independent depth while preserving radial stand-off.
        tangent=np.array([-away[1],away[0]])
        goal=start+self.parameters['view_baseline_m']*tangent;feedback=TranslationFeedback(maximum_speed=.03)
        reference=self.hold_rotation.T@np.array([0.,0.,1.])
        for n in range(120):
            pose=np.asarray(self.env.base_pose());delta=goal-pose[:2]
            if np.linalg.norm(delta)<.002:break
            if np.linalg.norm(pose[:2]-start)>.08:
                raise ObservedMotionViolation('Превышен предел второго ракурса')
            gravity=hand_rotation(self.env).T@np.array([0.,0.,1.])
            if math.acos(float(np.clip(reference@gravity,-1.,1.)))>math.radians(3.):
                raise ObservedMotionViolation('Во время смены ракурса изменился наклон захвата')
            if any(r is not None and r<.12 for r in self.env.lidar_scan()['ranges']):
                raise ObservedMotionViolation('Лидар не подтвердил проход для второго ракурса')
            speed=feedback.update(delta,float(pose[2]),PERIOD)
            self.tick(self.env.command[:6],(*speed,0.),ctx)
        else:raise ObservedMotionViolation('Второй ракурс не достигнут за ограниченное время')
        for _ in range(12):self.tick(self.env.command[:6],ctx=ctx)
        if np.linalg.norm(np.asarray(self.env.base_pose())[:2]-start)<.025:
            raise ObservedMotionViolation('Измеренное смещение не создало отдельный ракурс')
        held=self.verify_hold(ctx,probe=False)
        if held['outcome']!='success':raise ObservedMotionViolation(held['reason'])

    def observe(self,ctx):
        from src.task_executor import Observation
        from .rgbd_observation import observe_environment
        self.check(ctx);self.sequence+=1
        raw={};data=observe_environment(self.env,raw_output=raw)
        if raw.get('robot_depth') is not None:
            record=dict(frame=data,depth=raw['depth'],robot_depth=raw['robot_depth'],rgb=raw.get('rgb'),
                command=self.env.command[:6].copy(),base_pose=np.asarray(self.env.base_pose()).copy())
            if self.target is not None and self.target.get('appearance',{}).get('rgb_median') is not None:
                from .rgbd_observation import held_material_patches
                data['held_candidates']=held_material_patches(record,self.env.jaw_center(),self.target['appearance']['rgb_median'])
            self.depth_records.append(record);self.depth_records=self.depth_records[-24:]
            path=self.folder/'depth'/f'{self.sequence:04d}.npz'
            np.savez_compressed(path,depth=raw['depth'],robot_depth=raw['robot_depth'],
                rgb=raw.get('rgb'),
                optical_to_world=np.asarray(data['optical_to_world']),intrinsics_json=json.dumps(data['intrinsics']),
                frame_id=data['frame_id'],simulation_time=data['simulation_time'],command=record['command'],
                base_pose=record['base_pose'])
            self.depth_manifest.append(dict(frame_id=data['frame_id'],file=str(path)))
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
        self.move([*(base[:2]+self.parameters['inspection_reach_m']*direction),self.parameters['inspection_height_m']],30.,ctx)

    def survey_destination(self,ctx,*,max_views=8,transport_required=True):
        """A finite robot-relative survey; partial geometry only guides camera aim.

        Cached world geometry is an aiming hint. At least one new complete
        depth frame must agree with a separate complete camera viewpoint before
        the destination can be used. Gripper state is preserved throughout.
        """
        from .rgbd_observation import combine_container_views,ObservationRefused
        from .inspection_planning import ObservedMotionViolation
        from .multiview_container import combine_registered_container,opening_refresh_hint
        self.env.phase='skill_find_destination'
        pose=np.asarray(self.env.base_pose());direction=np.array([math.cos(pose[2]),math.sin(pose[2])])
        lateral=np.array([-direction[1],direction[0]])
        aims=[]
        if self.destination is not None:
            center=np.asarray(self.destination['center'],float)
            if self.before_hold is not None:
                # A verified high hold can point the camera horizontally.
                # Forcing its optical axis through the basket centre would
                # require tipping the hand, although the opening can be seen
                # below the axis within the measured image's field of view.
                camera=self.env.model.camera('wrist_depth').id
                current_height=float(self.env.data.cam_xpos[camera,2])
                forward_z=-float(self.env.data.cam_xmat[camera].reshape(3,3)[2,2])
                for distance in (.65,.70,.60,.75):
                    aims.append((np.r_[center[:2],current_height+forward_z*distance],distance))
            else:aims.append((center,.45))
        for forward,side,height in ((.40,.24,.15),(.40,-.24,.15),(.50,.35,.20),
                                    (.50,-.35,.20),(.35,.28,.25),(.35,-.28,.25)):
            aims.append((np.r_[pose[:2]+forward*direction+side*lateral,height],.40))
        visited=[];notes=[];new_complete=0;refreshed=False;yaw_attempts=0
        if self.before_hold is not None and self.destination is not None:
            for index,offset in enumerate((0.,0.)):
                if yaw_attempts>=max_views:break
                yaw_attempts+=1
                try:
                    if index:self.held_view_baseline(ctx,transport_required=transport_required)
                    self.held_yaw_view(self.destination['center'],offset,ctx,transport_required=transport_required)
                    frame=self.scene(ctx)
                    notes.append(dict(yaw_offset_deg=offset,frame_id=frame['frame_id'],
                        complete_candidates=len(frame.get('containers',[]))))
                    box=combine_registered_container(self.depth_records,now_simulation_time=float(self.env.data.time))
                    self.destination=box
                    self.surveys.append(dict(outcome='success',views=notes,destination=box))
                    return dict(outcome='success',destination=box,evidence=dict(
                        source='registered_rgbd_rim_and_opening',proof=box['evidence']))
                except ObservedMotionViolation:raise
                except (ValueError,ObservationRefused) as error:notes.append(dict(yaw_offset_deg=offset,reason=str(error)))
        while aims and len(visited)+yaw_attempts<max_views:
            self.check(ctx)
            point,distance=aims.pop(0)
            if any(np.linalg.norm(point-old[0])<.025 and abs(distance-old[1])<.025 for old in visited):
                continue
            visited.append((point.copy(),distance))
            try:
                self.look_at(point,ctx,distance_m=distance,min_camera_height_m=getattr(self,'destination_camera_min_height',.30),
                    transport_required=transport_required,height_reserve_m=getattr(self,'destination_height_reserve_m',0.))
                frame=self.scene(ctx)
            except ObservedMotionViolation:raise
            except ValueError as error:
                notes.append(dict(aim=point.tolist(),reason=str(error)));continue
            complete=frame.get('containers',[])
            notes.append(dict(aim=point.tolist(),frame_id=frame['frame_id'],complete_candidates=len(complete)))
            try:
                box=combine_registered_container(self.depth_records,now_simulation_time=float(self.env.data.time))
                self.destination=box
                self.surveys.append(dict(outcome='success',views=notes,destination=box))
                return dict(outcome='success',destination=box,
                    evidence=dict(source='registered_rgbd_rim_and_opening',proof=box['evidence']))
            except ObservationRefused:pass
            if not refreshed and self.before_hold is None:
                try:
                    hint=opening_refresh_hint(self.depth_records,now_simulation_time=float(self.env.data.time))
                    source=next(r for r in self.depth_records if r['frame']['frame_id']==hint['source_frame'])
                    if np.linalg.norm(np.asarray(self.env.base_pose())-source['base_pose'])>.001:
                        raise ObservationRefused('Исходная позиция шасси изменилась')
                    from .inspection_planning import check_observed_box_path
                    command=hint['command'];command[5]=self.env.command[5]
                    refreshed=True
                    check_observed_box_path(self.env,command,hint['occupied_box'],check=lambda:self.check(ctx))
                    self.move_angles(command,ctx)
                    fresh=self.scene(ctx)
                    notes.append(dict(revisit_source=hint['source_frame'],frame_id=fresh['frame_id']))
                    box=combine_registered_container(self.depth_records,now_simulation_time=float(self.env.data.time))
                    self.destination=box
                    self.surveys.append(dict(outcome='success',views=notes,destination=box))
                    return dict(outcome='success',destination=box,
                        evidence=dict(source='registered_rgbd_rim_and_opening',proof=box['evidence']))
                except (ObservationRefused,ValueError) as error:
                    if refreshed:notes.append(dict(revisit_refused=str(error)))
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
                point=np.asarray(hint['center'],float)
                if self.before_hold is not None:
                    camera=self.env.model.camera('wrist_depth').id
                    forward_z=-float(self.env.data.cam_xmat[camera].reshape(3,3)[2,2])
                    point[2]=float(self.env.data.cam_xpos[camera,2])+forward_z*.55
                aims.insert(0,(point,.55 if self.before_hold is not None else .45))
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
        from .observed_navigation import TranslationFeedback
        from .grasp_planning import plan_pick_stance
        self.env.phase='skill_approach'
        if self.target is None:return dict(outcome='unknown',reason='Нет наблюдаемой цели')
        self.update_grasp_axis()
        stance=plan_pick_stance(self.env,self.target,self.depth_records[-1],self.destination,
            check=lambda:self.check(ctx),preferred_fraction=self.parameters['pick_fraction'],
            lift_clearance=self.parameters['lift_clearance_m'],surface_offset=self.parameters['pick_surface_offset_m'],
            lateral_fraction=self.parameters['pick_lateral_fraction'],preferred_standoff=self.parameters['pick_standoff_m'],
            preferred_angle=self.parameters['pick_approach_angle_deg'],initial_lift_height=self.parameters['initial_lift_height_m'])
        self.pick_fraction=stance.fraction;self.pick_lift_height=float(stance.lift_xyz[2])
        self.planned_pick_xyz=stance.grasp_xyz.copy()
        destination=stance.base_xy
        travelled=0.;previous_pose=np.asarray(self.env.base_pose())[:2].copy()
        feedback=TranslationFeedback()
        for _ in range(240):
            pose=np.asarray(self.env.base_pose());delta=destination-pose[:2]
            travelled+=float(np.linalg.norm(pose[:2]-previous_pose));previous_pose=pose[:2].copy()
            if np.linalg.norm(delta)<.003:break
            if travelled>.22:return dict(outcome='failure',reason='Превышен предел локального подъезда',base_moved=True)
            ranges=self.env.lidar_scan()['ranges']
            # Stop on a near scan return rather than assuming a clear room.
            if any(r is not None and r<.12 for r in ranges):
                return dict(outcome='unknown',reason='Лидар не подтвердил свободный проход',base_moved=travelled>0)
            speed=feedback.update(delta,float(pose[2]),PERIOD)
            self.tick(self.env.command[:6],(*speed,0.),ctx)
        else:return dict(outcome='failure',reason='Подъезд не сошёлся',base_moved=travelled>0)
        for _ in range(12):self.tick(self.env.command[:6],ctx=ctx)
        if np.linalg.norm(destination-np.asarray(self.env.base_pose())[:2])>=.008:
            return dict(outcome='retry',reason='После остановки требуется уточнение подъезда',base_moved=travelled>0)
        return dict(outcome='success',base_moved=travelled>0,evidence=dict(local_travel_m=travelled,
            observed_pick_fraction=stance.fraction,chassis_clearance_m=stance.chassis_clearance_m,
            measured_pick_point_m=stance.grasp_xyz.tolist()))

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
        from .grasp_planning import observed_pick_patch
        self.target=new
        try:
            self.grasp_xyz=observed_pick_patch(self.depth_records[-1],new,self.env.base_pose(),
                fraction=getattr(self,'pick_fraction',.25),lateral_fraction=self.parameters['pick_lateral_fraction'],
                preferred_xyz=getattr(self,'planned_pick_xyz',None))
        except ObservationRefused:
            if not self.repair_enabled('reacquire_surface'):raise
            previous=np.asarray(self.target['center'],float)
            self.look_at(previous,ctx);fresh=self.estimate(ctx,'object')
            if np.linalg.norm(np.asarray(fresh['center'])-previous)>.10:
                return dict(outcome='unknown',reason='После нового обзора цель не совпала')
            self.target=new=fresh
            self.grasp_xyz=observed_pick_patch(self.depth_records[-1],new,self.env.base_pose(),
                fraction=getattr(self,'pick_fraction',.25),lateral_fraction=self.parameters['pick_lateral_fraction'])
            self.repair_events.append(dict(operator='reacquire_surface',new_frame_id=new['frame_id'],
                new_measured_point=self.grasp_xyz.tolist()))
        return dict(outcome='success',target=new,evidence=dict(frame_id=new['frame_id'],
            measured_pick_point_m=self.grasp_xyz.tolist()))

    def update_grasp_axis(self):
        if not self.repair_enabled('surface_axis_grasp'):return
        from .grasp_planning import observed_material_axis
        from .rgbd_observation import ObservationRefused
        try:axis=observed_material_axis(self.depth_records[-1],self.target)
        except (ObservationRefused,ValueError) as error:
            self.env.observed_grasp_axis=None
            self.repair_events.append(dict(operator='surface_axis_grasp',applied=False,reason=str(error)));return
        self.env.observed_grasp_axis=axis
        self.repair_events.append(dict(operator='surface_axis_grasp',applied=True,measured_axis=axis.tolist()))

    def pregrasp(self,ctx):
        from .grasp_planning import solve_pick_ik,plan_gripper_closure,observed_pick_patch
        from .rgbd_observation import ObservationRefused
        self.env.phase='skill_pregrasp'
        if self.grasp_xyz is None:return dict(outcome='unknown',reason='Нет свежей точки захвата')
        self.update_grasp_axis()
        # Base tracking and a fresh crop can change IK feasibility. Re-plan
        # from the newly measured cloud while the arm is still above it.
        points=[self.grasp_xyz.copy()]
        for fraction in (getattr(self,'pick_fraction',.25),.20,.30,.40,.50):
            try:points.append(observed_pick_patch(self.depth_records[-1],self.target,self.env.base_pose(),
                fraction=fraction,lateral_fraction=self.parameters['pick_lateral_fraction']))
            except (ValueError,ObservationRefused):continue
        errors=[];goal=None
        for point in points:
            self.check(ctx)
            xyz=point.copy();xyz[2]=max(.017,float(xyz[2])+self.parameters['pick_surface_offset_m'])
            try:
                path=plan_gripper_closure(self.env,xyz,check=lambda:self.check(ctx))
                candidate=solve_pick_ik(self.env,[*point[:2],max(.10,float(point[2])+.08)],30.,check=lambda:self.check(ctx))
                solve_pick_ik(self.env,[*point[:2],self.parameters['initial_lift_height_m']],170.,check=lambda:self.check(ctx))
            except ValueError as error:errors.append(str(error));continue
            self.grasp_xyz=point;self.closure_path=path;goal=candidate;break
        if goal is None:return dict(outcome='failure',reason='Свежая геометрия не допускает полного смыкания: '+str(errors[-1] if errors else 'нет участка'))
        self.move_angles(goal,ctx)
        return dict(outcome='success',evidence=dict(complete_closure_preflight=True,
            measured_pick_point_m=self.grasp_xyz.tolist(),closure_waypoints=len(self.closure_path)))

    def grasp(self,ctx):
        from .grasp_planning import solve_pick_ik,plan_gripper_closure
        self.env.phase='skill_grasp'
        # The measured visible surface supplies height; no cloth-node truth.
        # Geometric pad-centre offset above the visible cloth surface. It is
        # explicitly estimated, pending the same real gripper calibration as
        # the existing simulator contact geometry; it is not cloth truth.
        xyz=self.grasp_xyz.copy();xyz[2]=max(.017,float(xyz[2])+self.parameters['pick_surface_offset_m'])
        # Recheck before any descent, including partial retries/resumes.
        path=plan_gripper_closure(self.env,xyz,check=lambda:self.check(ctx))
        if self.repair_enabled('cartesian_feedback'):
            compensated=self.move_pick_feedback(xyz,30.,ctx)
            path=plan_gripper_closure(self.env,compensated,check=lambda:self.check(ctx))
        else:self.move_angles(path[0],ctx,duration=1.5)
        # Follow one continuous smooth closure, without stopping at every IK
        # waypoint. Preserve the same velocity/acceleration/jerk controller.
        duration=max(self.parameters['closure_duration_s'],float(np.max(np.sum(abs(np.diff(path,axis=0)),axis=0)/self.env.velocity_limit))*2.)
        count=math.ceil(duration/PERIOD)
        for n in range(count):
            u=(n+1)/count;blend=u*u*u*(10.-15.*u+6.*u*u)
            position=blend*(len(path)-1);i=min(int(position),len(path)-2)
            self.tick(path[i]+(path[i+1]-path[i])*(position-i),ctx=ctx)
        for _ in range(12):self.tick(path[-1],ctx=ctx)
        return dict(outcome='success',evidence=dict(command_only=True,holding_not_yet_confirmed=True))

    def lift(self,ctx):
        from .grasp_planning import solve_pick_ik
        self.env.phase='skill_lift'
        goal=solve_pick_ik(self.env,[*self.grasp_xyz[:2],self.parameters['initial_lift_height_m']],170.,check=lambda:self.check(ctx))
        self.move_angles(goal,ctx)
        # Verify this small lift BEFORE rotating the unconfirmed grasp into
        # the high transport posture. carry() performs the guarded second lift.
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
        candidates=frame.get('held_candidates')
        if candidates is None:candidates=frame.get('object_candidates',[])
        matches=[c for c in candidates if np.linalg.norm(np.asarray(c['center'])-jaw)<.08 and associated(c)]
        if len(matches)!=1:
            # Absence near the jaws is not proof of an empty grasp. Recovery
            # is authorised only by TWO fresh views of the matching material
            # still on the floor near its last measured location.
            if probe and self.target is not None:
                def on_floor(view):
                    return [c for c in view.get('object_candidates',[]) if associated(c)
                        and float(c['bounds_max'][2])<.04
                        and np.linalg.norm(np.asarray(c['center'])[:2]-np.asarray(self.target['center'])[:2])<.06
                        and np.linalg.norm(np.asarray(c['center'])-np.asarray(self.env.jaw_center()))>.065]
                ground=on_floor(frame)
                if len(ground)==1:
                    for _ in range(10):self.tick(self.env.command[:6],ctx=ctx)
                    second=self.scene(ctx);again=on_floor(second)
                    if (second['frame_id']!=frame['frame_id'] and
                        float(second['simulation_time'])-float(frame['simulation_time'])>=.3 and
                        len(again)==1 and np.linalg.norm(np.asarray(ground[0]['center'])-again[0]['center'])<.008):
                        self.empty_grasp_evidence=dict(source='two_frame_ground_reacquisition',
                            frames=[frame['frame_id'],second['frame_id']],confirmed=True,
                            simulation_time=float(second['simulation_time']))
                        return dict(outcome='failure',reason='Два кадра подтверждают: предмет остался на полу',
                            evidence=self.empty_grasp_evidence)
            return dict(outcome='unknown',reason='Камера не подтверждает удержание предмета у пальцев')
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
        probe_kwargs=dict(check=lambda:self.check(ctx),lowest_material_height_m=float(first['bounds_min'][2]))
        if self.repair_enabled('extended_hold_probe'):probe_kwargs['extended']=True
        if self.repair_enabled('kinematic_hold_probe'):probe_kwargs['kinematic']=True
        if any(self.repair_enabled(op) for op in ('extended_hold_probe','probe_feasible_view','kinematic_hold_probe')):
            probe_kwargs['obstacle']=self.destination
        probe_plan=plan_hold_probe(self.env,**probe_kwargs)
        self.move_angles(probe_plan.command,ctx)
        second=self.scene(ctx);new_jaw=np.asarray(self.env.jaw_center())
        observed=second.get('held_candidates')
        if observed is None:observed=second.get('object_candidates',[])
        candidates=[c for c in observed if associated(c)
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

    def recover_grasp(self,ctx):
        """Bounded sensor-authorised retry, never an unknown-hold release."""
        evidence=getattr(self,'empty_grasp_evidence',None)
        if (not evidence or evidence.get('confirmed') is not True or
            float(self.env.data.time)-evidence['simulation_time']>2.):
            return dict(outcome='unknown',reason='Нет свежего доказательства пустого захвата для повторной попытки')
        self.env.phase='skill_recover_grasp';self.empty_grasp_evidence=None
        self.before_hold=None;self.grasp_failures+=1
        # Raise/open without moving the base over fabric. Select another
        # observed surface percentile; approach() retains all swept-CAD gates.
        from .grasp_planning import solve_pick_ik
        goal=solve_pick_ik(self.env,[*self.env.jaw_center()[:2],.12],30.,check=lambda:self.check(ctx))
        self.move_angles(goal,ctx)
        self.inspect(ctx);self.target=self.estimate(ctx,'object')
        previous=self.parameters['pick_fraction']
        self.parameters['pick_fraction']=(.20 if previous>=.35 else .50)
        previous_lateral=self.parameters['pick_lateral_fraction']
        self.parameters['pick_lateral_fraction']=(.25 if previous_lateral>=.5 else .75)
        result=self.approach(ctx)
        self.grasp_recoveries.append(dict(attempt=self.grasp_failures,evidence=evidence,
            old_fraction=previous,new_fraction=self.parameters['pick_fraction'],
            old_lateral_fraction=previous_lateral,new_lateral_fraction=self.parameters['pick_lateral_fraction'],
            outcome=result['outcome']))
        return result

    def raise_verified_hold(self,ctx):
        """Prepare high transport only after low pickup has visual proof."""
        if getattr(self,'raised_verified_hold',False):return dict(outcome='success')
        if self.before_hold is None:return dict(outcome='unknown',reason='Высокий подъём требует подтверждённого захвата')
        from .manipulation_planning import hand_rotation
        from .inspection_planning import ObservedMotionViolation
        from scipy.spatial.transform import Rotation
        pose=np.asarray(self.env.base_pose());direction=np.array([math.cos(pose[2]),math.sin(pose[2])])
        xyz=[*(pose[:2]+.16*direction),getattr(self,'pick_lift_height',.24)]
        if self.repair_enabled('visibility_planned_lift'):
            from .repair_motion import plan_visible_lift
            relative=self.hold_rotation.T@(np.asarray(self.before_hold['center'])-self.hold_jaw)
            goal,count=plan_visible_lift(self.env,xyz,relative,check=lambda:self.check(ctx))
            self.repair_events.append(dict(operator='visibility_planned_lift',reserved_view_samples=count,
                predicted_visibility_is_not_hold_evidence=True))
        else:goal=self.env.solve_ik(xyz,self.env.command[:5],gripper=170.,downward=False)
        last=hand_rotation(self.env);ticks=0
        def guard():
            nonlocal last,ticks
            ticks+=1;rotation=hand_rotation(self.env)
            if Rotation.from_matrix(last.T@rotation).magnitude()>math.radians(3.):
                raise ObservedMotionViolation('Слишком большой шаг ориентации при высоком подъёме')
            last=rotation
            if ticks%4==0:
                result=self.verify_hold(ctx,probe=False)
                if result['outcome']!='success':raise ObservedMotionViolation('Высокий подъём: '+result['reason'])
        # Small angular increments permit feedback during intentional wrist
        # rotation, as in the already guarded destination inspection skill.
        duration=max(3.,float(np.sum(abs(goal[:5]-self.env.command[:5])))/.5*PERIOD)
        if duration>30.:return dict(outcome='unknown',reason='Нет конечного пути высокого подъёма')
        self.env.phase='skill_raise_verified_hold';self.move_angles(goal,ctx,duration=duration,motion_check=guard)
        self.raised_verified_hold=True
        return self.verify_hold(ctx,probe=False)

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
        # The destination was measured from independent views before pickup.
        # It is a bounded navigation hint, never fresh release evidence. Asking
        # an upward wrist camera to re-observe it while preserving a high hold
        # creates an impossible combination of arm constraints. Transport first;
        # support() still requires a NEW two-view opening measurement at arrival.
        box=self.destination
        if (box is None or not box.get('evidence') or
            float(self.env.data.time)-float(box.get('simulation_time',-1e6))>120.):
            return dict(outcome='unknown',reason='Нет измеренного положения корзины для локального маршрута')
        raised=self.raise_verified_hold(ctx)
        if raised['outcome']!='success':return raised
        held=self.verify_hold(ctx,probe=False)
        if held['outcome']!='success':return held
        view_moved=False
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
        destination=np.asarray(box['center'])[:2]-[float(half[0])+self.parameters['destination_standoff_m'],0.]
        moved=view_moved
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

    def reorient_for_destination(self,ctx,*,transport_required=False):
        """Intentional wrist reorientation, with visible grip checked throughout.

        This is a separate manipulation skill. Ordinary camera sweeps retain
        their original 3 degree guard. No hold reference is rebased during the
        move; the original material anchor must survive every fresh image.
        """
        from .inspection_planning import plan_camera_view,check_observed_box_path,ObservedMotionViolation
        from .manipulation_planning import hand_rotation
        from scipy.spatial.transform import Rotation
        held=self.verify_hold(ctx,probe=False)
        if held['outcome']!='success':return held
        box=self.destination;point=np.asarray(box['center'],float).copy()
        point[2]=float(box['bounds_max'][2])-.02
        # Optical distance is a search preference, not an equality required
        # by the task. Search reachable camera ranges around the observed rim;
        # every candidate retains the existing FK and obstacle-path gates.
        plan=None;errors=[]
        distances=sorted((.15,.20,.25,.30,.35,.40,.45,.55,.65,.75,.85),
            key=lambda d:abs(d-self.parameters['reorientation_distance_m']))
        for distance in distances:
            try:
                candidate=plan_camera_view(self.env,point,gripper=170.,distance_m=distance,
                    min_camera_height_m=self.parameters['reorientation_camera_height_m'],check=lambda:self.check(ctx),
                    minimum_jaw_height_m=float(box['bounds_max'][2])+.115 if transport_required else None,
                    maximum_jaw_reach_m=.192 if transport_required else None)
                check_observed_box_path(self.env,candidate.command,box,check=lambda:self.check(ctx))
                if self.repair_enabled('probe_feasible_view'):
                    from .manipulation_planning import probe_at_command
                    probe_at_command(self.env,candidate.command,lowest_material_height_m=None,
                        extended=self.repair_enabled('extended_hold_probe'),obstacle=box,check=lambda:self.check(ctx),
                        kinematic=self.repair_enabled('kinematic_hold_probe'))
                plan=candidate;break
            except ValueError as error:errors.append(str(error))
        if plan is None:return dict(outcome='unknown',reason='Нет пути изменения обзора: '+errors[-1])
        start=self.env.command[:6].copy();last=hand_rotation(self.env);checks=[];tick=0
        duration=max(3.,float(np.sum(abs(plan.command[:5]-start[:5])))/.5*PERIOD)
        if duration>30.:return dict(outcome='unknown',reason='Изменение обзора превышает конечный бюджет')
        def check_grip():
            nonlocal tick,last
            tick+=1
            if tick%2:return
            rotation=hand_rotation(self.env)
            change=float(Rotation.from_matrix(last.T@rotation).magnitude())
            if change>math.radians(3.):raise ObservedMotionViolation('Слишком большой шаг изменения ориентации захвата')
            observed=self.verify_hold(ctx,probe=False)
            if observed['outcome']!='success':raise ObservedMotionViolation('Во время изменения обзора: '+observed['reason'])
            checks.append(dict(frame_id=observed['evidence']['frame_id'],step_rotation_deg=math.degrees(change)))
            last=rotation;self.env.phase='skill_reorient_destination'
        self.env.phase='skill_reorient_destination'
        self.move_angles(plan.command,ctx,duration=duration,motion_check=check_grip)
        held=self.verify_hold(ctx)
        if held['outcome']!='success':return held
        held=self.verify_stationary_hold(ctx)
        return dict(held,evidence=dict(held.get('evidence',{}),reorientation_frames=checks,
            original_anchor_checked=True,gripper_closed=True))

    def support(self,ctx):
        self.env.phase='skill_support'
        previous=self.destination
        oriented=self.reorient_for_destination(ctx)
        if oriented['outcome']!='success':return oriented
        survey=self.survey_destination(ctx,max_views=4,transport_required=False)
        if survey['outcome']!='success':return survey
        box=self.destination
        if float(box['uncertainty_m'])>.02:return dict(outcome='unknown',reason='Граница корзины перед отпусканием недостаточно точна')
        from .multiview_container import opening_displacement
        if previous and opening_displacement(previous,box)>.06:
            return dict(outcome='unknown',reason='Корзина не совпала после подъезда')
        self.destination=box
        lo,hi=np.asarray(box['bounds_min']),np.asarray(box['bounds_max'])
        if min(hi[:2]-lo[:2])<.10:return dict(outcome='unknown',reason='Отверстие корзины слишком мало или не видно')
        from .manipulation_planning import plan_release_hover
        hover=plan_release_hover(self.env,box,fraction=self.parameters['release_fraction'],
            margin=self.parameters['release_margin_m'],check=lambda:self.check(ctx),
            via_command=self.transport_poses[-1]['command'] if self.transport_poses else None,
            roll_deg=self.parameters['release_roll_deg'],withdraw_height=self.parameters['withdraw_height_m'])
        self.release_plan=hover;self.release_xyz=hover.target
        # A deliberate change of wrist orientation is observed throughout;
        # it is never hidden inside an unguarded free-position IK movement.
        last=None;tick=0
        def guard():
            nonlocal last,tick
            tick+=1
            if tick%2:return
            from .manipulation_planning import hand_rotation
            from scipy.spatial.transform import Rotation
            rotation=hand_rotation(self.env)
            if last is not None and Rotation.from_matrix(last.T@rotation).magnitude()>math.radians(3.):
                raise ValueError('Слишком быстрый поворот при размещении')
            held=self.verify_hold(ctx,probe=False)
            if held['outcome']!='success':raise ValueError(held['reason'])
            last=rotation;self.env.phase='skill_support'
        for command in ([hover.via_command,hover.command] if hover.via_command is not None else [hover.command]):
            duration=max(3.,float(np.sum(abs(command[:5]-self.env.command[:5])))*PERIOD/.5)
            self.move_angles(command,ctx,duration=duration,motion_check=guard)
        return dict(outcome='success',evidence=dict(frame_id=box['frame_id'],rim_height_m=float(hi[2])))

    def release(self,ctx):
        self.env.phase='skill_release'
        if self.release_xyz is None:return dict(outcome='unknown',reason='Нет проверенной точки отпускания')
        held=self.verify_hold(ctx,probe=False)
        if held['outcome']!='success':return held
        self.env.phase='skill_release'
        self.before_release=self.scene(ctx)
        self.release_jaw=np.asarray(self.env.jaw_center()).copy()
        from .inspection_planning import check_observed_box_path
        opened=self.release_plan.open_command
        check_observed_box_path(self.env,opened,self.destination,check=lambda:self.check(ctx))
        self.move_angles(opened,ctx,duration=2.)
        self.release_commanded=True
        # Opening is only a command, not proof of detachment. A finite wrist
        # roll frees draped fabric before withdrawal; cameras and the separate
        # outcome evaluator must still confirm actual placement.
        for command in self.release_plan.roll_commands:
            check_observed_box_path(self.env,command,self.destination,check=lambda:self.check(ctx))
            self.move_angles(command,ctx,duration=1.5)
        return dict(outcome='success',evidence=dict(command_only=True))

    def withdraw(self,ctx):
        self.env.phase='skill_withdraw'
        if self.release_xyz is None:return dict(outcome='unknown',reason='Нет точки отвода')
        from .inspection_planning import check_observed_box_path
        # A high hover may already be at the top of the arm workspace. Search
        # an actual checked withdrawal, rather than demand another 7–12 cm up.
        goal=None
        if hasattr(self,'release_plan') and self.release_plan.withdraw_command is not None:
            try:
                check_observed_box_path(self.env,self.release_plan.withdraw_command,self.destination,check=lambda:self.check(ctx))
                goal=self.release_plan.withdraw_command
            except ValueError:pass
        for retreat,height in ((.035,self.parameters['withdraw_height_m']),(.05,.04),(.065,.02),(.065,0.)):
            if goal is not None:break
            point=self.release_xyz+[-retreat,0.,height]
            try:
                command=self.env.solve_ik(point,self.env.command[:5],gripper=30.,downward=False)
                check_observed_box_path(self.env,command,self.destination,check=lambda:self.check(ctx))
            except ValueError:continue
            goal=command;break
        if goal is None:return dict(outcome='unknown',reason='Нет достижимого отвода раскрытых пальцев с зазором над бортиком')
        self.move_angles(goal,ctx)
        for _ in range(40):self.tick(self.env.command[:6],ctx=ctx)
        return dict(outcome='success')

    def verify_place(self,ctx):
        from .placement_evidence import verify_released_target
        self.env.phase='skill_verify_place'
        if self.destination is None:return dict(outcome='unknown',reason='Нет наблюдаемой корзины')
        if not hasattr(self,'before_release') or self.before_hold is None:
            return dict(outcome='unknown',reason='Нет последовательности наблюдаемого удержания и отпускания')
        def verify():
            first=self.scene(ctx)
            for _ in range(10):self.tick(self.env.command[:6],ctx=ctx)
            return verify_released_target(first,self.scene(ctx),destination=self.destination,
                held_reference=self.before_hold,release_xyz=self.release_xyz,jaw_xyz=self.env.jaw_center(),
                before_release=self.before_release,release_jaw_xyz=self.release_jaw)
        result=verify()
        if result['outcome']=='success':return result
        from .inspection_planning import plan_camera_view,check_observed_box_path
        aim=np.asarray(self.destination['center'],float).copy()
        rim=float(self.destination['bounds_max'][2]);aim[2]=max(.015,rim*self.parameters['verification_aim_fraction'])
        for distance in sorted(set((.20,.30,.40,.50,self.parameters['verification_distance_m'])),
                               key=lambda d:abs(d-self.parameters['verification_distance_m'])):
            try:
                plan=plan_camera_view(self.env,aim,gripper=30.,distance_m=distance,
                    min_camera_height_m=max(.18,rim+.03),check=lambda:self.check(ctx))
                check_observed_box_path(self.env,plan.command,self.destination,check=lambda:self.check(ctx))
            except ValueError:continue
            self.move_angles(plan.command,ctx);self.env.phase='skill_verify_place'
            result=verify()
            if result['outcome']=='success':return result
        return result

    def stop_motion(self):
        """Release base commands even during cancellation, physics or I/O errors."""
        if callable(getattr(self.env,'stop_motion',None)):self.env.stop_motion()
        self.env.pending_command[6:]=0.;self.env.command[6:]=0.
        self.env.data.ctrl[self.env.wheel_ids]=0.

    def close(self):
        self.stop_motion()
        try:self.handle.close()
        finally:
            (self.folder/'destination-surveys.json').write_text(json.dumps(self.surveys,ensure_ascii=False,indent=2))
            (self.folder/'transport-poses.json').write_text(json.dumps(self.transport_poses,ensure_ascii=False,indent=2))
            (self.folder/'grasp-recoveries.json').write_text(json.dumps(self.grasp_recoveries,ensure_ascii=False,indent=2))
            (self.folder/'controller-repairs.json').write_text(json.dumps(dict(program=self.controller_program,
                events=self.repair_events,experience_sha256=self.experience.get('sha256')),ensure_ascii=False,indent=2))
            (self.folder/'depth-records.json').write_text(json.dumps(self.depth_manifest,ensure_ascii=False,indent=2))
            (self.folder/'observations.json').write_text(json.dumps(self.frames,ensure_ascii=False,indent=2))

    def operations(self):
        def declared(operation):
            def execute(ctx):
                before=self.base_motion_sequence
                result=operation(ctx)
                if self.base_motion_sequence!=before:return dict(result,base_moved=True)
                return result
            return execute
        return {name:declared(getattr(self,name)) for name in ('find','approach','reobserve','pregrasp','grasp',
            'lift','verify_hold','recover_grasp','carry','support','release','withdraw','verify_place')}
