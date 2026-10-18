"""Shared RGB-D/lidar room port for MuJoCo and ObservedRobotEnv.

Only the independent experiment runner may inspect physical object outcomes.
This module uses robot FK, odometry, lidar, RGB semantics and registered depth.
ACT alone supplies all grasp/place commands; geometric routines only acquire
camera views, approach poses and observed hold/placement evidence.
"""
from pathlib import Path
import math
import time

import numpy as np

from .grounded_skill_port import GroundedSkillPort
from .scan_navigation import ScanMap
from .observed_navigation import TranslationFeedback
from .semantic_vision import SemanticDetector,associate,detect_metric_crops,refresh_stationary_detections,floor_material_candidates
from .room_policy import RoomActor,contract_sha,runtime_files,issued_command
from .environment import PERIOD,LOW,HIGH


class Context:
    def __init__(self,check,revision=0):self.check,self.base_revision=check,revision


class RoomCleanupPort:
    search_layers=('floor','raised')
    def __init__(self,env,folder,model,profile,semantic_root,*,cancelled=lambda:False,progress=lambda _:None,device='cpu'):
        self.env,self.folder,self.profile=env,Path(folder),profile
        self.folder.mkdir(parents=True,exist_ok=True)
        self.cancelled,self.progress=cancelled,progress
        self.geometry=GroundedSkillPort(env,self.folder/'observations',cancelled,progress,semantic_root=semantic_root,query='sock')
        self.geometry.destination_height_reserve_m=.02
        self.detector=self.geometry.detector;self.actor=self.load_actor(model,profile,device=device)
        self.revision=0;self.seq=0;self.grid=ScanMap(extent=4.,radius=.35);self.view_index=0;self.held=False;self.hold_history=[]
        self.viewed_floor=set()
        self._scope=('simulation','room-scene','estimated-dynamics') if not hasattr(env,'transport') else tuple(env._scope)
        self.contract=contract_sha(profile,runtime_files())

    def load_actor(self,model,profile,*,device):return RoomActor(model,profile,device=device)

    def preflight(self):
        if hasattr(self.env,'transport'):
            # Hardware execution must be admitted by its explicit qualified
            # package owner; constructing a sensor port never grants motion.
            self.env.transport.preflight(self.contract)
        return dict(ready=True,blocked_by=[],policy_sha256=self.actor.sha,mission_contract_sha256=self.contract,
            simulation_qualified=False,physical_ready=False)

    def scope(self):return self._scope if not hasattr(self.env,'transport') else tuple(self.env._scope)
    def pose(self):return self.env.base_pose()
    def set_base_revision(self,value):self.revision=value
    def permit(self):
        if self.cancelled():raise InterruptedError('Room trial cancelled')
        if self.env.last_error:raise ValueError(self.env.last_error)
        if hasattr(self.env,'transport'):self.env.transport.permit()

    def begin(self,check):self.check=check;check()

    def _ctx(self,check=None):return Context(check or self.check,self.revision)

    def _scan(self):
        pose=np.asarray(self.pose())
        key=(float(self.env.data.time),*pose)
        if (getattr(self,'scan_key',None)==key and (not hasattr(self.env,'transport')
                or time.monotonic()-self.scan_at<.25)):return
        self.grid.radius=self.own_radius()
        scans=[]
        for scanner in (0,1):
            scan=self.env.lidar_scan(scanner,rays=360);site=self.env.data.site('laser'+str(scanner)+'_frame')
            angles=scan['angle_min']+np.arange(len(scan['ranges']))*scan['angle_increment']
            directions=np.c_[np.cos(angles),np.sin(angles),np.zeros(len(angles))]@site.xmat.reshape(3,3).T
            scans.append((site.xpos[:2].copy(),directions[:,:2],scan['ranges']))
        self.grid.add_current_scans(scans)
        cells=np.indices(self.grid.free.shape).transpose(1,2,0)*self.grid.resolution-self.grid.extent
        # Only the presently occupied chassis core is self-known. An extended
        # arm radius must not turn unseen space around the robot into free map.
        self.grid.free[np.linalg.norm(cells-pose[:2],axis=-1)<=.18]=True
        self.scan_key=key;self.scan_at=time.monotonic()

    def own_radius(self):
        from itertools import product
        signs=np.array(list(product((-1.,1.),repeat=3)));radius=.35;pose=np.asarray(self.pose())[:2]
        for i in range(self.env.model.ngeom):
            body=int(self.env.model.geom_bodyid[i])
            while body and self.env.model.body(body).name!='base_link':body=int(self.env.model.body_parentid[body])
            if not body:continue
            aabb=self.env.model.geom_aabb[i];local=aabb[:3]+signs*aabb[3:]
            world=local@self.env.data.geom_xmat[i].reshape(3,3).T+self.env.data.geom_xpos[i]
            radius=max(radius,float(np.linalg.norm(world[:,:2]-pose,axis=1).max())+.02)
        if self.held and self.geometry.before_hold is not None:
            jaw=np.asarray(self.env.jaw_center())[:2]
            item=self.geometry.before_hold
            half=np.linalg.norm(np.asarray(item['bounds_max'])-np.asarray(item['bounds_min']))/2
            radius=max(radius,float(np.linalg.norm(jaw-pose)+half+.025))
        return radius

    def plan(self,goal,held=False):
        goal={name:float(goal[name]) for name in ('x','y','yaw')}
        self._scan()
        if hasattr(self.env,'transport') and hasattr(self.env.transport,'plan'):
            nav=self.env.transport.plan(goal)
            if nav.get('reachable') is not True:return dict(reachable=False)
        try:
            path=self.grid.route(self.pose()[:2],[goal['x'],goal['y']])
            return dict(reachable=True,points=path.tolist(),cost=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()))
        except ValueError:return dict(reachable=False)

    def survey_candidates(self):
        out=[]
        if hasattr(self.env,'transport') and hasattr(self.env.transport,'frontiers'):
            out.extend(self.env.transport.frontiers())
        self._scan();pose=np.asarray(self.pose())
        # Observation sites come only from scanned cells and actual A* paths.
        # No hidden source/basket locations enter this exploration proposal.
        for distance in (0.,.45,.8,1.2,1.8):
            for index,angle in enumerate(np.linspace(-np.pi,np.pi,12,endpoint=False)):
                if distance==0. and index%3:continue
                xy=pose[:2]+distance*np.array([np.cos(angle),np.sin(angle)])
                goal=dict(x=float(xy[0]),y=float(xy[1]),yaw=float(angle))
                cell=self.grid.cell(xy)
                if self.grid.inside(cell) and self.grid.free[cell] and not self.grid.occupied[cell]:out.append(goal)
        for goal in out:
            gain=0;c,s=np.cos(goal['yaw']),np.sin(goal['yaw'])
            for forward in np.arange(.25,.81,.08):
                for lateral in np.arange(-.25,.26,.08):
                    cell=self.grid.cell(np.array([goal['x'],goal['y']])+np.array([[c,-s],[s,c]])@np.array([forward,lateral]))
                    if self.grid.inside(cell) and self.grid.free[cell] and not self.grid.occupied[cell] and cell not in self.viewed_floor:gain+=1
            goal['information_gain']=gain
        return out

    def _stop_base(self):
        for _ in range(12):self.geometry.tick(self.env.command[:6],ctx=self._ctx())

    def lidar_clearance(self,known_box=None):
        # Only a target whose full occupied volume has already passed the CAD
        # sweep may replace the generic near-return stop during final approach.
        # Every other near return, from either lidar, remains a stop condition.
        for scanner in (0,1):
            scan=self.env.lidar_scan(scanner,rays=360);site=self.env.data.site('laser'+str(scanner)+'_frame')
            for index,r in enumerate(scan['ranges']):
                if r is None or not np.isfinite(r) or r>=.12:continue
                if known_box is not None:
                    angle=scan['angle_min']+index*scan['angle_increment']
                    xyz=site.xpos+site.xmat.reshape(3,3)@np.array([r*np.cos(angle),r*np.sin(angle),0.])
                    margin=float(known_box['uncertainty_m'])+.01
                    if np.all(xyz>=np.asarray(known_box['bounds_min'])-margin) and np.all(xyz<=np.asarray(known_box['bounds_max'])+margin):continue
                raise ValueError('Lidar stopped room motion: unaccounted near obstacle')

    def prepare_navigation(self,*,check,held=False):
        self._stop_base()
        if held:
            if not self._transport_ready():raise ValueError('ACT has not reached a raised compact transport posture')
            if self.verify('hold',check=check)['outcome']!='success':raise ValueError('Hold unconfirmed before transport')
        else:
            goal=np.asarray(self.profile['initial_command'][:6])
            if self.geometry.destination is not None:
                from .inspection_planning import check_observed_box_path
                check_observed_box_path(self.env,goal,self.geometry.destination,check=check)
            self.geometry.move_angles(goal,self._ctx(check))

    def _transport_ready(self):
        jaw=np.asarray(self.env.jaw_center());base=np.asarray(self.pose())
        return (jaw[2]>=self.profile['transport_height_m'] and
            np.linalg.norm(jaw[:2]-base[:2])<=self.profile['transport_maximum_reach_m'])

    def navigate(self,goal,*,check,held=False):
        from src.room_cleanup import ViewUnavailable
        from .room_navigation import HeadingFeedback
        feedback=TranslationFeedback(maximum_speed=.04);heading=HeadingFeedback();ctx=self._ctx(check)
        path=[];anchor=np.asarray(self.pose());anchor_yaw=abs(math.atan2(math.sin(goal['yaw']-anchor[2]),math.cos(goal['yaw']-anchor[2])))
        for index in range(2400):
            check();pose=np.asarray(self.pose());delta=np.array([goal['x'],goal['y']])-pose[:2]
            yaw_error=(goal['yaw']-pose[2]+np.pi)%(2*np.pi)-np.pi
            # Brake inside the acceptance band, leaving room for measured
            # coast-down. Repeatedly braking at its outer edge can stall just
            # outside the final tolerance without ever completing arrival.
            if np.linalg.norm(delta)<.015 and abs(yaw_error)<.025:
                self._stop_base();settled=np.asarray(self.pose())
                if (np.linalg.norm(np.array([goal['x'],goal['y']])-settled[:2])<.025
                        and abs(math.atan2(math.sin(goal['yaw']-settled[2]),math.cos(goal['yaw']-settled[2])))<.04):
                    return dict(completed=True,pose=self.pose())
                continue
            if index%10==0:
                planned=self.plan(goal,held=held)
                if not planned.get('reachable'):
                    self._stop_base();raise ViewUnavailable('Navigation did not confirm a fresh full-footprint path')
                path=np.asarray(planned['points'])
                self.progress(dict(phase='navigate',pose=pose.tolist(),goal=goal,remaining_m=float(np.linalg.norm(delta)),yaw_error_rad=float(yaw_error)))
                if held and self.verify('hold',check=check)['outcome']!='success':raise ValueError('Hold lost during navigation')
            self.lidar_clearance()
            waypoint=next((p for p in path if np.linalg.norm(p-pose[:2])>.06),np.array([goal['x'],goal['y']]))
            speed=feedback.update(waypoint-pose[:2],pose[2],PERIOD)
            self.geometry.tick(self.env.command[:6],(*speed,heading.update(yaw_error,PERIOD)),ctx)
            if index and index%400==0:
                now=np.asarray(self.pose());turned=abs(math.atan2(math.sin(now[2]-anchor[2]),math.cos(now[2]-anchor[2])))
                if np.linalg.norm(now[:2]-anchor[:2])<.015 and turned<.015 and abs(yaw_error)>=anchor_yaw-.01:
                    self._stop_base();raise ViewUnavailable('Measured chassis made no route progress for 20 simulation seconds')
                anchor=now;anchor_yaw=abs(yaw_error)
        self._stop_base()
        raise ViewUnavailable('Finite navigation budget exhausted: distance %.3f m, heading %.3f rad'%(np.linalg.norm(delta),yaw_error))

    def search_view(self,check,layer='floor'):
        from src.room_cleanup import ViewUnavailable
        from .inspection_planning import CameraViewUnavailable,ObservedMotionViolation
        self._stop_base();pose=np.asarray(self.pose());self.view_index+=1;errors=[]
        for offset in ((self.view_index%3-1)*.6,0.,-.6,.6):
            raised=layer=='raised';angle=pose[2]+offset
            aim=np.r_[pose[:2]+(.65 if raised else .45)*np.array([np.cos(angle),np.sin(angle)]),.18 if raised else .05]
            try:self.geometry.look_at(aim,self._ctx(check),distance_m=.6 if raised else .4,
                min_camera_height_m=.35 if raised else .20,height_reserve_m=.02);return
            except ObservedMotionViolation:raise
            except CameraViewUnavailable as error:errors.append(str(error))
        raise ViewUnavailable('No feasible stopped search view: '+'; '.join(errors))

    def inspect_search_hint(self,hint,*,check):return self.look_at(hint,check=check,held=False)

    def look_at(self,target,*,check,held=False):
        from src.room_cleanup import ViewUnavailable
        from .inspection_planning import CameraViewUnavailable,ObservedMotionViolation
        self._stop_base()
        if target.get('kind')=='basket':
            rim=float(target['bounds_max'][2])
            if rim>.40:raise ViewUnavailable('Raised extent exceeds the supported basket height; no motion to that hint')
            self.geometry.destination_camera_min_height=max(.30,rim+.12)
            previous=self.geometry.destination
            self.geometry.destination=target
            try:
                result=self.geometry.survey_destination(self._ctx(check),transport_required=held)
                if result.get('outcome')!='success':raise ViewUnavailable(result.get('reason','Basket opening is unconfirmed'))
            finally:
                # Partial hypotheses may aim the camera, but must never leak
                # into the occupied-volume checks for a subsequent drive.
                if target.get('usable_for')=='camera_aim_only':self.geometry.destination=previous
            return
        try:self.geometry.look_at(target['center'],self._ctx(check),distance_m=.34,
                min_camera_height_m=.18,transport_required=held)
        except ObservedMotionViolation:raise
        except CameraViewUnavailable as error:raise ViewUnavailable(str(error)) from error

    def observe(self):
        self._stop_base();self.geometry.scene(self._ctx());record=self.geometry.depth_records[-1]
        frame=record['frame'];objects=[];hints=[]
        k=frame['intrinsics'];depth=record['depth'];rows,cols=np.indices(depth.shape)
        valid=np.isfinite(depth)&(depth>.03)&(depth<3.)
        rows,cols,z=rows[valid][::8],cols[valid][::8],depth[valid][::8]
        xyz=np.c_[(cols-k['cx'])/k['fx']*z,(rows-k['cy'])/k['fy']*z,z,np.ones(len(z))]
        world=(np.asarray(frame['optical_to_world'])@xyz.T).T
        for xy in world[(world[:,2]>-.01)&(world[:,2]<.06),:2]:
            cell=self.grid.cell(xy)
            if self.grid.inside(cell):self.viewed_floor.add(cell)
        from .multiview_container import combine_registered_container,refresh_registered_container
        from .rgbd_observation import ObservationRefused
        try:baskets=[combine_registered_container(self.geometry.depth_records,now_simulation_time=float(self.env.data.time))]
        except ObservationRefused:baskets=frame['container_hypotheses']
        low_objects=floor_material_candidates(frame,baskets)
        detections=self.detector.detect(record['rgb'],'sock. laundry basket',frame['frame_id'])
        try:associate([d for d in detections if 'sock' in d.get('label','').lower()],low_objects,frame)
        except ValueError:
            detections+=detect_metric_crops(self.detector,record['rgb'],'sock',frame,low_objects)
        # Model latency never relabels an old exposure as fresh. Capture new
        # RGB-D and carry forward only detections whose actual pixels/depth and
        # camera pose remain independently consistent with the model input.
        reference=record;self._stop_base();self.geometry.scene(self._ctx());record=self.geometry.depth_records[-1];frame=record['frame']
        detections=refresh_stationary_detections(reference,record,detections)
        try:
            if len(baskets)==1 and baskets[0].get('evidence',{}).get('shape')=='raised_walls_with_visible_opening':
                baskets=[refresh_registered_container(self.geometry.depth_records,baskets[0],record)]
            else:baskets=frame['container_hypotheses']
        except ObservationRefused:baskets=frame['container_hypotheses']
        low_objects=floor_material_candidates(frame,baskets)
        for kind,query,candidates in (('sock','sock',low_objects),('basket','laundry basket',baskets)):
            labelled=[d for d in detections if query in d.get('label','').lower()]
            try:target=associate(labelled,candidates,frame)
            except ValueError:continue
            target=dict(target,kind=kind,frame_id=frame['frame_id'],geometry_source='registered_depth',semantic_source='rgb_model')
            if kind=='basket' and (target.get('evidence',{}).get('shape')!='raised_walls_with_visible_opening'
                    or target.get('uncertainty_m',1.)>.02):
                hints.append(dict(target,confidence=target['semantic_evidence']['detection']['confidence'],usable_for='camera_aim_only'))
            else:objects.append(target)
        self.seq+=1
        from .learning import write_json
        write_json(self.folder/'latest-perception.json',dict(frame_id=frame['frame_id'],sequence=self.seq,
            low_candidates=len(frame['low_objects']),basket_candidates=len(baskets),
            model_detections=detections,objects=objects,hints=hints))
        return dict(sequence=self.seq,captured_at=frame['captured_monotonic'],base_revision=self.revision,scope=self.scope(),objects=objects,hints=hints)

    def local_admission(self,kind,target):
        pose=np.asarray(self.pose());point=np.asarray(target['center']);delta=point[:2]-pose[:2]
        c,s=np.cos(pose[2]),np.sin(pose[2]);local=np.array([[c,s],[-s,c]])@delta
        radius=(np.asarray(target['bounds_max'])-np.asarray(target['bounds_min']))[:2].max()/2
        if kind=='sock':admitted=abs(local[0]-.20)<.025 and abs(local[1])<.025
        else:
            from .inspection_planning import check_observed_box_path,CameraViewUnavailable
            admitted=radius+.134<=local[0]<=radius+.29 and abs(local[1])<.025
            if admitted:
                try:check_observed_box_path(self.env,self.env.command[:6],target,check=self.check)
                except CameraViewUnavailable:admitted=False
        return dict(inside_envelope=bool(admitted))

    def local_align(self,kind,target,*,check,held=False):
        from src.room_cleanup import ViewUnavailable
        from scipy.ndimage import binary_dilation
        from .inspection_planning import check_observed_box_path,CameraViewUnavailable
        pose=np.asarray(self.pose());lo,hi=np.asarray(target['bounds_min']),np.asarray(target['bounds_max'])
        if kind=='basket' and target.get('evidence',{}).get('shape')!='raised_walls_with_visible_opening':
            raise ViewUnavailable('Local basket approach requires a complete observed rim and opening')
        radius=(hi-lo)[:2].max()/2
        stands=[.20] if kind=='sock' else [radius+reserve for reserve in (.134,.17,.20,.225,.25,.275)]
        goal=None;failures=[]
        for stand in stands:
            proposal=np.asarray(target['center'])[:2]-stand*np.array([np.cos(pose[2]),np.sin(pose[2])])
            try:check_observed_box_path(self.env,self.env.command[:6],target,check=check,base_target=proposal)
            except CameraViewUnavailable as error:failures.append(str(error));continue
            goal=proposal;break
        if goal is None:raise ViewUnavailable('No CAD-clear local stance in the supported arm envelope: '+'; '.join(failures))
        self._scan()
        # Substitute measured 3-D occupied geometry ONLY for this observed
        # target's 2-D projection. Other obstacles and unknown space retain the
        # conservative global footprint; the target box sweep above is strict.
        cells=np.indices(self.grid.free.shape).transpose(1,2,0)*self.grid.resolution-self.grid.extent
        projected=np.all(cells>=lo[:2]-.02,axis=-1)&np.all(cells<=hi[:2]+.02,axis=-1)
        occupied=self.grid.occupied&~projected;unknown=(~self.grid.free)&~projected
        n=int(np.ceil(self.grid.radius/self.grid.resolution));x,y=np.mgrid[-n:n+1,-n:n+1]
        blocked=binary_dilation(occupied|unknown,structure=x*x+y*y<=n*n,border_value=1)
        for fraction in np.linspace(0,1,max(2,int(np.linalg.norm(goal-pose[:2])/.005))):
            cell=self.grid.cell(pose[:2]+fraction*(goal-pose[:2]))
            if not self.grid.inside(cell) or blocked[cell]:raise ViewUnavailable('Local alignment crosses another obstacle or unseen space')
        feedback=TranslationFeedback(maximum_speed=.03);ctx=self._ctx(check)
        for step in range(800):
            check();now=np.asarray(self.pose());delta=goal-now[:2]
            if np.linalg.norm(delta)<.008:self._stop_base();return dict(completed=True,observed_3d_alignment=True)
            self.lidar_clearance(known_box=target)
            self.geometry.tick(self.env.command[:6],(*feedback.update(delta,now[2],PERIOD),0.),ctx)
            if held and step%10==0 and self.verify('hold',check=check)['outcome']!='success':raise ValueError('Hold unconfirmed during local alignment')
        raise ValueError('Finite local alignment did not converge')

    def prepare_policy(self,phase,*,check,held=False):
        self._stop_base()
        if phase=='pick':self.geometry.move_angles(np.asarray(self.profile['initial_command'][:6]),self._ctx(check))

    def learned_manipulation(self,phase,target,*,check):
        ctx=self._ctx(check)
        if phase=='pick':
            self.geometry.target=target;self.hold_history=[]
        else:
            self.geometry.destination=target;self.geometry.before_release=self.geometry.scene(ctx)
            self.geometry.release_jaw=np.asarray(self.env.jaw_center()).copy()
            self.geometry.release_xyz=self.geometry.release_jaw.copy()
            self.geometry.release_commanded=False
        self.actor.begin(phase,target,self.env.command);base=np.asarray(self.pose())
        residual=np.zeros(6)
        for step in range(1600):
            check()
            if hasattr(self.env,'transport'):self.geometry.scene(ctx)
            rgb=self.env.render_rgb('wrist')
            if rgb.shape!=(240,320,3):
                import cv2
                rgb=cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA)
            command=issued_command(self.env.command,np.clip(self.actor(self.env.command,rgb),LOW,HIGH),residual)
            if phase=='place' and not self.geometry.release_commanded and command[5]<self.env.command[5]-.5:
                from .room_policy import admit_release
                admit_release(target,self.env.jaw_center())
                self.geometry.before_release=self.geometry.scene(ctx)
                self.geometry.release_jaw=np.asarray(self.env.jaw_center()).copy()
                self.geometry.release_xyz=self.geometry.release_jaw.copy()
                self.geometry.release_commanded=True
            if hasattr(self.env,'transport'):
                self.env.transport.action_kind='policy'
                try:self.env.step(command)
                finally:self.env.transport.action_kind='view'
            else:self.env.step(command,applied=True)
            if step%10==0:
                self.progress(dict(phase='policy_'+phase,steps=step,command=command.tolist()))
                self._stop_base()
                if phase=='place' and not self.geometry.release_commanded:
                    if self.geometry.verify_hold(ctx,probe=False).get('outcome')!='success':
                        raise ValueError('Observed hold lost before neural release')
                previously_held=self.held
                proof=self._proof(phase,ctx)
                if phase=='pick' and previously_held and proof.get('outcome')!='success':
                    raise ValueError('Previously verified hold was lost during ACT lift')
                if proof.get('outcome')=='success' and (phase!='pick' or self._transport_ready()):
                    self.held=phase=='pick';self._stop_base()
                    return dict(completed=True,teacher_fallback=False,policy_sha256=self.actor.sha,
                        base_moved=bool(np.linalg.norm(np.asarray(self.pose())-base)>.003),steps=step+1)
        return dict(completed=False,teacher_fallback=False,policy_sha256=self.actor.sha)

    def _proof(self,phase,ctx):
        if phase=='pick':
            if self.held:return self.geometry.verify_hold(ctx,probe=False)
            from .manipulation_planning import hand_rotation
            frame=self.geometry.scene(ctx);jaw=np.asarray(self.env.jaw_center());rotation=hand_rotation(self.env)
            matches=[c for c in frame.get('held_candidates',[]) if np.linalg.norm(np.asarray(c['center'])-jaw)<.08
                and c['bounds_min'][2]>.045 and c.get('uncertainty_m',1.)<=.02]
            if len(matches)!=1:self.hold_history=[];return dict(outcome='unknown')
            candidate=matches[0];relative=rotation.T@(np.asarray(candidate['center'])-jaw)
            self.hold_history.append(dict(time=frame['simulation_time'],id=frame['frame_id'],
                jaw=jaw.copy(),relative=relative,candidate=candidate))
            self.hold_history=self.hold_history[-16:]
            first=self.hold_history[0]
            stable=(frame['simulation_time']-first['time']>=1. and np.linalg.norm(jaw-first['jaw'])>=.015
                and max(np.linalg.norm(r['relative']-relative) for r in self.hold_history)<.015)
            if not stable:return dict(outcome='unknown')
            self.geometry.before_hold=candidate;self.geometry.hold_jaw=jaw.copy();self.geometry.hold_rotation=rotation.copy()
            self.held=True
            return dict(outcome='success',evidence=dict(source='observed_ACT_lift_motion',frames=[r['id'] for r in self.hold_history]))
        from .placement_evidence import verify_released_target
        if self.geometry.before_hold is None or not self.geometry.release_commanded:return dict(outcome='unknown')
        first=self.geometry.scene(ctx)
        for _ in range(10):self.geometry.tick(self.env.command[:6],ctx=ctx)
        return verify_released_target(first,self.geometry.scene(ctx),destination=self.geometry.destination,
            held_reference=self.geometry.before_hold,release_xyz=self.geometry.release_xyz,jaw_xyz=self.env.jaw_center(),
            before_release=self.geometry.before_release,release_jaw_xyz=self.geometry.release_jaw)

    def verify(self,kind,*,check):
        self._stop_base();proof=self._proof('pick' if kind=='hold' else 'place',self._ctx(check))
        if kind=='hold' and proof.get('outcome')=='success':
            for _ in range(10):self.geometry.tick(self.env.command[:6],ctx=self._ctx(check))
            proof=self._proof('pick',self._ctx(check))
        return dict(outcome=proof.get('outcome','unknown'),evidence=dict(source='rgbd_temporal',
            confirmed=proof.get('outcome')=='success',scope=self.scope(),base_revision=self.revision,
            captured_at=time.monotonic(),frame_ids=[r['frame']['frame_id'] for r in self.geometry.depth_records[-2:]],detail=proof))

    def stop(self):self.geometry.stop_motion()
    def close(self):self.geometry.close()
