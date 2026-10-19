"""Semantic gating around the observed geometric executor, without hidden labels."""
from .skill_port import SimulatorSkillPort
from .semantic_vision import SemanticDetector,associate


class GroundedSkillPort(SimulatorSkillPort):
    def __init__(self,*args,semantic_root,query,**kwargs):
        super().__init__(*args,**kwargs);self.query=query
        self.detector=SemanticDetector(semantic_root,self.folder/'recognition',self.cancelled)
        self.semantic_frames=[]

    def estimate(self,ctx,kind):
        if kind!='object':return super().estimate(ctx,kind)
        frame=self.scene(ctx)
        detections=self.detector.detect(self.env.render_rgb('wrist'),self.query,frame['frame_id'])
        try:item=associate(detections,frame['low_objects'],frame)
        except ValueError as error:
            # An ambiguous crop needs a new sensor viewpoint. It must not
            # bypass the existing bounded re-observation/reframing protocol.
            from .rgbd_observation import ObservationRefused
            raise ObservationRefused(str(error)) from error
        self.semantic_frames.append(frame['frame_id'])
        return dict(item,frame_id=frame['frame_id'],simulation_time=frame['simulation_time'])

    def survey_destination(self,ctx,*,max_views=8,transport_required=True):
        if self.query!='wrapped cookie':return super().survey_destination(ctx,max_views=max_views,transport_required=transport_required)
        import numpy as np
        from .surface_destination import observed_trays,combine_trays
        from .rgbd_observation import ObservationRefused
        from .inspection_planning import ObservedMotionViolation
        self.env.phase='skill_find_destination';pose=np.asarray(self.env.base_pose());direction=np.array([np.cos(pose[2]),np.sin(pose[2])]);side=np.array([-direction[1],direction[0]])
        views=[]
        if self.destination is not None:aims=[np.asarray(self.destination['center'])]*4
        else:aims=[np.r_[pose[:2]+forward*direction+lateral*side,.025] for forward,lateral in ((.35,.25),(.35,-.25),(.45,.25),(.45,-.25))]
        for index,aim in enumerate(aims[:max_views]):
            try:
                self.look_at(aim,ctx,distance_m=.34+index*.035,min_camera_height_m=.16,transport_required=transport_required)
                self.scene(ctx);record=self.depth_records[-1];found=observed_trays(record)
                if len(found)!=1:continue
                views.append(dict(candidate=found[0],frame_id=record['frame']['frame_id'],
                    camera_position=np.asarray(record['frame']['optical_to_world'])[:3,3].tolist()))
                try:self.destination=combine_trays(views)
                except ObservationRefused:continue
                return dict(outcome='success',evidence=dict(source='two_view_observed_tray',destination=self.destination))
            except ObservedMotionViolation:raise
            except (ValueError,ObservationRefused):continue
        return dict(outcome='unknown',reason='Не получены два независимых обзора опоры подноса')

    def navigate_observed(self,goal,ctx,*,held=False):
        import math
        import numpy as np
        from .scan_navigation import ScanMap
        from .observed_navigation import TranslationFeedback
        from .environment import PERIOD
        grid=ScanMap();feedback=TranslationFeedback(maximum_speed=.04);distance=0.;previous=np.asarray(self.env.base_pose())[:2]
        for step in range(1400):
            pose=np.asarray(self.env.base_pose());distance+=float(np.linalg.norm(pose[:2]-previous));previous=pose[:2].copy()
            if np.linalg.norm(np.asarray(goal)-pose[:2])<.035:return
            if distance>3.:raise ValueError('Превышен бюджет наблюдаемого маршрута')
            if step%10==0:
                for scanner in (0,1):
                    scan=self.env.lidar_scan(scanner,rays=360);site=self.env.data.site('laser'+str(scanner)+'_frame')
                    angles=scan['angle_min']+np.arange(len(scan['ranges']))*scan['angle_increment']
                    rays=np.c_[np.cos(angles),np.sin(angles),np.zeros(len(angles))]@site.xmat.reshape(3,3).T
                    grid.add(site.xpos[:2],rays[:,:2],scan['ranges'])
                # The robot's present footprint is occupied by its known body;
                # only that footprint, not unseen space ahead, is seeded free.
                cells=np.indices(grid.free.shape).transpose(1,2,0)*grid.resolution-grid.extent
                grid.free[np.linalg.norm(cells-pose[:2],axis=-1)<=grid.radius]=True
                for item in (self.target,self.destination):
                    if item is not None:grid.block_box(item['bounds_min'][:2],item['bounds_max'][:2])
                route=grid.route(pose[:2],goal)
            if any(r is not None and r<.12 for r in self.env.lidar_scan()['ranges']):raise ValueError('Лидар остановил маршрут')
            waypoint=next((point for point in route if np.linalg.norm(point-pose[:2])>.06),np.asarray(goal))
            speed=feedback.update(waypoint-pose[:2],float(pose[2]),PERIOD)
            self.tick(self.env.command[:6],(*speed,0.),ctx)
            if held and step%10==0:
                result=self.verify_hold(ctx,probe=False)
                if result['outcome']!='success':raise ValueError(result['reason'])
        raise ValueError('Маршрут не завершён за конечный бюджет')

    def find(self,ctx):
        if self.query!='wrapped cookie':return self._local_find(ctx)
        import numpy as np
        self.delivery_start_xy=np.asarray(self.env.base_pose())[:2].copy()
        self.inspect(ctx)
        survey=self.survey_destination(ctx)
        if survey['outcome']!='success':return survey
        pose=np.asarray(self.env.base_pose());direction=np.array([np.cos(pose[2]),np.sin(pose[2])])
        # Search sector is relative to the robot, not a hidden source position.
        from .inspection_planning import ObservedMotionViolation
        for distance in (.9,1.1,.7):
            try:
                self.look_at(np.r_[pose[:2]+distance*direction,.03],ctx,distance_m=.75,min_camera_height_m=.20)
                self.target=self.estimate(ctx,'object')
                return dict(outcome='success',target=self.target,evidence=dict(source='RGB_semantics_and_registered_depth',identity_proven=False))
            except ObservedMotionViolation:raise
            except ValueError:continue
        return dict(outcome='unknown',reason='Запрошенный предмет не найден в наблюдаемом секторе')

    def _local_find(self,ctx):
        result=super().find(ctx)
        if result['outcome']=='success':result['evidence']=dict(source='RGB_semantics_and_registered_depth',semantic_frames=list(self.semantic_frames),identity_proven=False)
        return result

    def approach(self,ctx):
        import numpy as np
        if self.target is not None:
            pose=np.asarray(self.env.base_pose());delta=np.asarray(self.target['center'])[:2]-pose[:2]
            if np.linalg.norm(delta)>.45:
                direction=delta/np.linalg.norm(delta);goal=np.asarray(self.target['center'])[:2]-.38*direction
                self.navigate_observed(goal,ctx)
                self.look_at(self.target['center'],ctx);self.target=self.estimate(ctx,'object')
        return super().approach(ctx)


    def carry(self,ctx):
        if self.query=='wrapped cookie' and hasattr(self,'delivery_start_xy'):
            if self.destination is None:return dict(outcome='unknown',reason='Нет измеренного назначения переноса')
            raised=self.raise_verified_hold(ctx)
            if raised['outcome']!='success':return raised
            import numpy as np
            from .manipulation_planning import plan_transport_pose,transport_pose_satisfied,hand_rotation
            if np.linalg.norm(np.asarray(self.env.base_pose())[:2]-self.delivery_start_xy)>.4:
                transport=plan_transport_pose(self.env,max(.18,float(self.destination['bounds_max'][2])+.09),check=lambda:self.check(ctx))
                reference=transport.reference_rotation.T@np.array([0.,0.,1.])
                def guard():
                    gravity=hand_rotation(self.env).T@np.array([0.,0.,1.])
                    if np.arccos(np.clip(reference@gravity,-1.,1.))>np.deg2rad(3.):raise ValueError('Наклон захвата изменился перед маршрутом')
                if not transport.retained:self.move_angles(transport.command,ctx,motion_check=guard)
                if not transport_pose_satisfied(self.env,transport.minimum_height_m,transport.maximum_reach_m):
                    return dict(outcome='unknown',reason='Нет компактной транспортной позы для маршрута')
                held=self.verify_stationary_hold(ctx)
                if held['outcome']!='success':return held
                self.navigate_observed(self.delivery_start_xy,ctx,held=True)
        return super().carry(ctx)

    def verify_place(self,ctx):
        if self.query!='wrapped cookie':return super().verify_place(ctx)
        from .surface_destination import verify_on_tray
        if not hasattr(self,'before_release') or self.before_hold is None:return dict(outcome='unknown',reason='Нет подтверждённого удержания до отпускания')
        first=self.scene(ctx)
        for _ in range(10):self.tick(self.env.command[:6],ctx=ctx)
        return verify_on_tray(first,self.scene(ctx),destination=self.destination,held_reference=self.before_hold,
            before_release=self.before_release,release_jaw=self.release_jaw,jaw_xyz=self.env.jaw_center())
