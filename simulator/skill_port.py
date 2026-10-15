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

    def move_angles(self,goal,ctx,duration=1.5):
        start=self.env.command[:6].copy()
        count=max(round(duration/PERIOD),round(np.max(abs(goal-start)/self.env.velocity_limit)/PERIOD)+10)
        for n in range(count):
            u=(n+1)/count;blend=u*u*u*(10.-15.*u+6.*u*u)
            self.tick(start+(goal-start)*blend,ctx=ctx)
        for _ in range(12):self.tick(goal,ctx=ctx)
        return goal

    def look_at(self,point,ctx):
        """Aim the calibrated wrist camera at an already observed location."""
        import mujoco
        from scipy.optimize import least_squares
        from .explorer_model import servo_to_joint
        scratch=mujoco.MjData(self.env.model);scratch.qpos[:]=self.env.data.qpos
        camera=self.env.model.camera('wrist_depth').id
        start=self.env.command[:5].copy();point=np.asarray(point,float)
        def residual(angles):
            q=servo_to_joint([*angles,30.]);scratch.qpos[self.env.joint_q]=q
            for name,multiplier in (('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)):
                scratch.qpos[self.env.model.joint(name).qposadr[0]]=multiplier*q[5]
            mujoco.mj_forward(self.env.model,scratch)
            optical=scratch.cam_xmat[camera].reshape(3,3).T@(point-scratch.cam_xpos[camera])
            return np.r_[optical[:2]*3.,(-optical[2]-.24),np.maximum(.12-scratch.cam_xpos[camera,2],0.)*3.,(angles-start)*.00003]
        fits=[least_squares(residual,np.clip(seed,.001,[179.999]*4+[269.999]),
            bounds=([0.]*5,[180.]*4+[270.]),max_nfev=100) for seed in (start,[90.,90.,90.,90.,90.])]
        best=min(fits,key=lambda fit:np.linalg.norm(fit.fun[:3]))
        if np.linalg.norm(best.fun[:2])>.04:raise ValueError('Не найден обзор камеры с целью внутри кадра')
        self.move_angles(np.r_[best.x,30.],ctx)

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

    def find(self,ctx):
        self.env.phase='skill_find';self.inspect(ctx)
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
            if self.before_hold is None or not hasattr(self,'hold_jaw'):
                return dict(outcome='unknown',reason='Нет исходного подтверждения удержания')
            relative=np.asarray(first['center'])-jaw
            previous=np.asarray(self.before_hold['center'])-self.hold_jaw
            if np.linalg.norm(relative-previous)>.025:
                return dict(outcome='unknown',reason='Положение предмета относительно захвата изменилось')
            return dict(outcome='success',evidence=dict(confirmed=True,source='tracked_rgbd_geometry',frame_id=frame['frame_id']))
        self.move(jaw+[0.,0.,.02],170.,ctx,downward=False)
        second=self.scene(ctx);new_jaw=np.asarray(self.env.jaw_center())
        candidates=[c for c in second.get('object_candidates',[]) if associated(c)
            and np.linalg.norm(np.asarray(c['center'])-new_jaw)<.08]
        if len(candidates)!=1:return dict(outcome='unknown',reason='Повторный кадр не подтверждает удержание')
        moved=np.asarray(candidates[0]['center'])-first['center'];hand_moved=new_jaw-jaw
        if second['frame_id']==frame['frame_id'] or hand_moved[2]<.01 or np.linalg.norm(moved-hand_moved)>.015:
            return dict(outcome='unknown',reason='Предмет не подтверждён совместным движением с захватом')
        self.before_hold=candidates[0]
        self.hold_jaw=new_jaw
        return dict(outcome='success',evidence=dict(confirmed=True,source='two_frame_rgbd_motion_association',
            frames=[frame['frame_id'],second['frame_id']],semantic_identity='unverified_single_geometric_target'))

    def carry(self,ctx):
        self.env.phase='skill_find_destination'
        # Missing or partial container geometry cannot be replaced by the
        # simulated bin pose or an assumed pre-recorded destination.
        self.destination=self.estimate(ctx,'container')
        box=self.destination
        if float(box['uncertainty_m'])>.02:return dict(outcome='unknown',reason='Граница корзины измерена недостаточно точно')
        rim=float(box['bounds_max'][2]);pose=np.asarray(self.env.base_pose())
        direction=np.array([math.cos(pose[2]),math.sin(pose[2])])
        self.move([*(pose[:2]+.20*direction),max(.18,rim+.09)],170.,ctx,downward=False)
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
        self.env.phase='skill_support';box=self.estimate(ctx,'container')
        if float(box['uncertainty_m'])>.02:return dict(outcome='unknown',reason='Граница корзины перед отпусканием недостаточно точна')
        if self.destination and np.linalg.norm(np.asarray(box['center'])-self.destination['center'])>.06:
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
        self.move(self.release_xyz,30.,ctx,downward=False,duration=2.)
        return dict(outcome='success',evidence=dict(command_only=True))

    def withdraw(self,ctx):
        self.env.phase='skill_withdraw'
        if self.release_xyz is None:return dict(outcome='unknown',reason='Нет точки отвода')
        self.move(self.release_xyz+[-.035,0.,.07],30.,ctx,downward=False)
        for _ in range(40):self.tick(self.env.command[:6],ctx=ctx)
        return dict(outcome='success')

    def verify_place(self,ctx):
        self.env.phase='skill_verify_place';frame=self.scene(ctx)
        if self.destination is None:return dict(outcome='unknown',reason='Нет наблюдаемой корзины')
        lo,hi=np.asarray(self.destination['bounds_min']),np.asarray(self.destination['bounds_max'])
        matches=[]
        for item in frame.get('object_candidates',[]):
            evidence=item.get('evidence',{})
            if evidence.get('identity_association_verified') is not True or evidence.get('self_occlusion_not_resolved') is not False:continue
            lower,upper=np.asarray(item['bounds_min']),np.asarray(item['bounds_max'])
            margin=max(.015,float(item['uncertainty_m'])+float(self.destination['uncertainty_m']))
            evidence=item.get('evidence',{})
            if (evidence.get('release_temporal_association_verified') is True
                and evidence.get('support_surface_observed') is True
                and np.all(lower[:2]>lo[:2]+margin) and np.all(upper[:2]<hi[:2]-margin)
                and lower[2]>=0. and upper[2]<hi[2]-margin):matches.append(item)
        if len(matches)!=1:return dict(outcome='unknown',reason='Камера не подтверждает предмет внутри корзины после отпускания')
        return dict(outcome='success',evidence=dict(confirmed=True,source='rgbd_geometry',frame_id=frame['frame_id']))

    def stop_motion(self):
        """Release base commands even during cancellation, physics or I/O errors."""
        self.env.pending_command[6:]=0.;self.env.command[6:]=0.
        self.env.data.ctrl[self.env.wheel_ids]=0.

    def close(self):
        self.stop_motion()
        try:self.handle.close()
        finally:
            (self.folder/'observations.json').write_text(json.dumps(self.frames,ensure_ascii=False,indent=2))

    def operations(self):
        return {name:getattr(self,name) for name in ('find','approach','reobserve','pregrasp','grasp',
            'lift','verify_hold','carry','support','release','withdraw','verify_place')}
