"""Measured camera transforms and conservative feature/depth object continuity.

A class label is only the initial hypothesis. Losing feature correspondence is
unknown, never a new object silently inheriting the old object's identity.
"""
from collections import deque
import json
import math
import threading
import time
import uuid
import cv2
import numpy as np
from surface_foreground import locate

BRACKET_WAIT_SECONDS=.10
MAX_PENDING_FRAMES=6


class JointSamplePending(ValueError):
    """An exposure has a left sample but still awaits its measured right sample."""


class JointHistory:
    def __init__(self):
        self.lock=threading.Lock();self.boot=None;self.samples=[deque(maxlen=512) for _ in range(6)];self.invalid=[False]*6
    def add(self, sample):
        with self.lock:
            if sample['boot_id']!=self.boot:
                self.boot=sample['boot_id'];self.samples=[deque(maxlen=512) for _ in range(6)];self.invalid=[False]*6
            index=sample['joint']-1
            if not 0<=index<6:raise ValueError('Invalid servo ID')
            if sample.get('position_valid') is True and sample.get('error')==0 and sample.get('device_error')==0:
                stamp=sample['acquired_monotonic_ns']/1e9;angle=sample['physical_deg']
                if not math.isfinite(stamp) or not math.isfinite(angle):
                    self.invalid[index]=True;raise ValueError('Invalid joint sample geometry')
                if self.samples[index] and stamp<=self.samples[index][-1][0]:
                    self.invalid[index]=True;raise ValueError('Joint sample time did not advance')
                self.invalid[index]=False;self.samples[index].append((stamp,angle))
            else:self.samples[index].clear();self.invalid[index]=True
    def at(self, when, boot):
        with self.lock:
            if self.boot is None:raise JointSamplePending('Waiting for first measured joint samples')
            if boot!=self.boot:raise ValueError('Camera pose belongs to another controller boot')
            if not math.isfinite(when):raise ValueError('Invalid exposure time')
            result=[]
            pending=False
            for index,samples in enumerate(self.samples):
                if self.invalid[index]:raise ValueError('Joint feedback reported an actual error')
                pairs=[(a,b) for a,b in zip(samples,list(samples)[1:]) if a[0]<=when<=b[0] and 0<b[0]-a[0]<=.075]
                if pairs:
                    a,b=pairs[-1];u=(when-a[0])/(b[0]-a[0]);result.append(a[1]+u*(b[1]-a[1]))
                elif not samples or samples[-1][0]<=when:
                    pending=True
                else:raise ValueError('No measured joint samples bracketing camera exposure')
            if pending:raise JointSamplePending('Waiting for measured joint sample after camera exposure')
            return result


class CommandHistory:
    """Exposure-time command estimate; never interpolate an unsent future goal."""
    def __init__(self):
        self.lock=threading.Lock();self.samples=deque(maxlen=512);self.epoch=None
    def add(self,state):
        with self.lock:
            arm=state.get('arm') or {};ref=state.get('manual_reference') or {}
            epoch=((state.get('identity') or {}).get('boot'),ref.get('session'),ref.get('reference_generation'))
            if epoch!=self.epoch or not arm.get('reference_valid') or not arm.get('estimated'):
                self.samples.clear();self.epoch=epoch
            if arm.get('reference_valid') and arm.get('estimated'):
                values=arm.get('servo_deg');stamp=state.get('monotonic_ns',0)/1e9
                if not values or len(values)!=6 or not all(math.isfinite(v) for v in values):
                    raise ValueError('Incomplete command estimate')
                if not self.samples or stamp>self.samples[-1][0]:self.samples.append((stamp,list(values)))
    def at(self,when,boot):
        with self.lock:
            if self.epoch is None or self.epoch[0]!=boot:raise ValueError('Command reference boot changed')
            rows=[row for row in self.samples if 0<=when-row[0]<=.15]
            if not rows:raise JointSamplePending('No command state at camera exposure')
            return list(rows[-1][1])


def reacquire_candidate(reference, current, bbox):
    """New object hypothesis in a verified static view, NOT a continued track.

    A slow semantic model names an old image. Require matching appearance and
    fresh separated depth, then allocate a new ID in FeatureObject(current,...).
    Labels remain explicitly unverified; outcome verification starts afterward.
    """
    gap=float(current['stamp'])-float(reference['stamp'])
    if not 0<=gap<120 or str(reference['frame'])!=str(current['frame']):
        raise ValueError('Semantic observation too old or camera frame changed')
    if not np.array_equal(reference['k'],current['k']) or not np.array_equal(reference['d'],current['d']):
        raise ValueError('Camera calibration changed during recognition')
    old,new=locate(reference,bbox),locate(current,bbox)
    if old['outcome']!='candidate' or new['outcome']!='candidate':
        raise ValueError('No fresh separated object at the candidate location')
    points=[np.array([v['position'][k] for k in ('x','y','z')]) for v in (old,new)]
    if np.linalg.norm(points[1]-points[0])>.01:raise ValueError('Candidate moved during recognition')
    x1,y1,x2,y2=np.rint(bbox).astype(int)
    a=cv2.cvtColor(reference['rgb'],cv2.COLOR_BGR2GRAY)[y1:y2,x1:x2].astype(float).ravel()
    b=cv2.cvtColor(current['rgb'],cv2.COLOR_BGR2GRAY)[y1:y2,x1:x2].astype(float).ravel()
    if len(a)!=len(b) or min(np.std(a),np.std(b))<8:raise ValueError('Appearance cannot be associated')
    correlation=float(np.corrcoef(a,b)[0,1])
    if not math.isfinite(correlation) or correlation<.92:raise ValueError('Candidate appearance changed during recognition')
    return dict(bbox=list(bbox),image_stamp=float(current['stamp']),
                previous_image_stamp=float(reference['stamp']),appearance_correlation=correlation,
                semantic_identity_verified=False,continuous_from_previous_image=False)


class FeatureObject:
    def __init__(self, sample, bbox):
        geometry=locate(sample,bbox)
        if geometry['outcome']!='candidate':raise ValueError('Object depth not separated: '+geometry.get('reason','unknown'))
        rgb=sample['rgb'];h,w=rgb.shape[:2];x1,y1,x2,y2=np.rint(bbox).astype(int)
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Invalid detection box')
        # Select texture on the separated foreground only, not the entire box.
        v,u=np.mgrid[:h,:w];z=sample['depth'];k=sample['k']
        rays=cv2.undistortPoints(np.stack([u,v],axis=-1).astype(float).reshape(-1,1,2),k,sample['d']).reshape(h,w,2)
        xyz=np.concatenate([rays*z[:,:,None],z[:,:,None]],axis=-1)
        plane=np.asarray(geometry['evidence']['plane_camera']);height=xyz@plane[:3]+plane[3]
        mask=((u>=x1)&(u<x2)&(v>=y1)&(v<y2)&np.isfinite(z)&(z>.15)&(z<2)&
              (height>geometry['evidence']['min_separation_m'])&(height<.25)).astype(np.uint8)*255
        self.gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        self.points=cv2.goodFeaturesToTrack(self.gray,100,.02,3,mask=mask)
        if self.points is None or len(self.points)<10:raise ValueError('Object has insufficient visible tracking texture')
        spread=np.ptp(self.points[:,0],axis=0)
        self.initial_scale=float(max(1.,math.sqrt(max(1.,spread[0]*spread[1]))))
        self.initial=len(self.points);self.stamp=float(sample['stamp']);self.object_id=uuid.uuid4().hex
        self.frame=str(sample['frame']);self.ended=False
    def update(self, sample):
        if self.ended:raise ValueError('Object association already lost')
        stamp=float(sample['stamp'])
        if not self.stamp<stamp<=self.stamp+.4 or str(sample['frame'])!=self.frame:
            self.ended=True;raise ValueError('Object tracking frame gap or frame change')
        gray=cv2.cvtColor(sample['rgb'],cv2.COLOR_BGR2GRAY)
        nxt,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None)
        if nxt is None:self.ended=True;raise ValueError('Object occluded')
        back,rev,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,nxt,None)
        if back is None:self.ended=True;raise ValueError('No reciprocal object association')
        h,w=gray.shape;uv=nxt[:,0]
        good=((ok[:,0]>0)&(rev[:,0]>0)&(np.linalg.norm(back[:,0]-self.points[:,0],axis=1)<.7)&
              (uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1))
        points=nxt[good];fraction=len(points)/self.initial
        if len(points)<8 or fraction<.7:
            self.ended=True;raise ValueError('Object identity lost through occlusion or ambiguous optical flow')
        uv=points[:,0];pixels=np.rint(uv).astype(int);z=sample['depth'][pixels[:,1],pixels[:,0]]
        valid=np.isfinite(z)&(z>.15)&(z<2.)
        if valid.mean()<.85:
            self.ended=True;raise ValueError('Object depth became unavailable')
        uv=uv[valid];z=z[valid]
        rays=cv2.undistortPoints(uv.astype(float).reshape(-1,1,2),sample['k'],sample['d'])[:,0]
        xyz=np.column_stack([rays*z[:,None],z]);center=np.median(xyz,axis=0)
        if np.percentile(np.linalg.norm(xyz-center,axis=1),90)>.15:
            self.ended=True;raise ValueError('Tracked depth no longer forms a small object')
        spread=np.ptp(points[:,0],axis=0);scale=float(math.sqrt(max(1.,spread[0]*spread[1]))/self.initial_scale)
        self.gray,self.points,self.stamp=gray,points,stamp
        return dict(object_id=self.object_id,point_camera=center,association_fraction=fraction,object_scale=scale)


class MeasuredVision:
    def __init__(self, root, node, arm, model, maps):
        from std_msgs.msg import String
        self.root,self.arm,self.model,self.maps=root,arm,model,maps
        self.factory_mode=getattr(arm,'factory_timed',False) is True
        command_mode=self.factory_mode or getattr(arm,'profile',{}).get('manual_reference_version')==1
        self.command_mode=command_mode
        self.history=CommandHistory() if command_mode else JointHistory()
        self.lock=threading.RLock();self.track=None;self.frames=deque(maxlen=400)
        self.error=None;self.last_stamp=0.;self.settings=None;self.generation=0
        self.pending=deque();self.enqueued_stamp=0.;self.waiting_for_joints=None
        def sample(message):
            try:self.history.add(json.loads(message.data))
            except (KeyError,ValueError,TypeError):pass
        self.subscription=node.create_subscription(String,'/explorer/controller_state' if command_mode else '/explorer/servo_sample',sample,100)
        threading.Thread(target=self.worker,daemon=True).start()
    def snapshot(self):
        with np.load(self.root/'data/rgbd-snapshot.npz',allow_pickle=False) as raw:sample=dict(raw)
        if not 0<=time.time()-float(sample['stamp'])<.4:raise ValueError('RGB-D frame is stale')
        return sample
    def geometry(self, sample, settings):
        if getattr(self,'factory_mode',False):
            state=self.arm.reference();angles=state.get('servo_deg')
            settled=float(state['at'])+float(state.get('runtime_ms',0))/1000+.12
            if float(sample['stamp'])<settled:raise JointSamplePending('Camera exposure predates settled factory command estimate')
            if not angles or len(angles)!=6 or state.get('phase')!='command_elapsed_observation_required':
                raise ValueError('Factory arm command estimate is not stable')
        else:
            state=self.arm._state();boot=(state.get('identity') or {}).get('boot')
            exposure=float(sample['stamp'])-(state['at']-state['monotonic_ns']/1e9)
            angles=self.history.at(exposure,boot)
        model=self.model()
        with model.lock:
            model.set_state(angles[:5],settings['gripper_linkage_rad'])
            mount=np.array(model.state.get_global_link_transform('arm4'))
            tool=np.array(model.state.get_global_link_transform('Gripping'))
        return mount@np.asarray(settings['camera_to_mount']),tool[:3,3],angles
    def start(self, sample, bbox, settings):
        with self.lock:
            self.generation+=1;self.settings=settings;self.track=FeatureObject(sample,bbox)
            self.frames.clear();self.error=None;self.last_stamp=float(sample['stamp'])
            self.pending.clear();self.enqueued_stamp=self.last_stamp;self.waiting_for_joints=None
        return self.track.object_id
    def start_tracker(self, tracker, settings):
        with self.lock:
            self.generation+=1;self.settings=settings;self.track=tracker
            self.frames.clear();self.error=None;self.last_stamp=tracker.stamp
            self.pending.clear();self.enqueued_stamp=self.last_stamp;self.waiting_for_joints=None
    def stop(self):
        with self.lock:self.generation+=1;self.track=None;self.pending.clear();self.waiting_for_joints=None

    def process_snapshot(self, sample, now=None):
        """Retry the same exposure for at most 100 ms; never extrapolate a pose."""
        clock=time.monotonic if now is None else lambda:now
        now=clock()
        with self.lock:
            track,settings,generation=self.track,self.settings,self.generation
            if track is None or self.error is not None:return 0
            stamp=float(sample['stamp'])
            if stamp>self.enqueued_stamp:
                if len(self.pending)>=MAX_PENDING_FRAMES:
                    self.error='Measured camera pose queue overflow';self.pending.clear()
                    raise ValueError(self.error)
                self.pending.append((sample,now+BRACKET_WAIT_SECONDS));self.enqueued_stamp=stamp
        processed=0
        for _ in range(MAX_PENDING_FRAMES):
            with self.lock:
                if track is not self.track or generation!=self.generation or not self.pending:return processed
                queued,deadline=self.pending[0]
            try:
                if clock()>=deadline:raise ValueError('Measured joint bracket deadline expired')
                transform,tcp,angles=self.geometry(queued,settings)
                if clock()>=deadline:raise ValueError('Measured joint bracket deadline expired')
                with self.lock:
                    if track is not self.track or generation!=self.generation:return processed
                    observation=track.update(queued)
                    obj=(transform@np.r_[observation['point_camera'],1])[:3]
                    plane=np.asarray(settings['floor_plane_base']);pose=self.maps.pose()
                    value=dict(at=float(queued['stamp']),object_id=observation['object_id'],confidence=observation['association_fraction'],
                        confidence_kind='feature_retention_not_semantic_probability',
                        association_fraction=observation['association_fraction'],identity_association_verified=True,
                        depth_validated=True,camera_pose_measured=not getattr(self,'command_mode',False),
                        camera_pose_validated_for_execution=bool(settings.get('handeye_execution_authorized')),
                        camera_pose_validation_record=settings.get('handeye_physical_validation_record'),
                        camera_pose_source='command_estimate' if getattr(self,'command_mode',False) else 'servo_measurement',frame='base_footprint',
                        object_xyz=obj.tolist(),tcp_xyz=tcp.tolist(),base_xyyaw=[pose[k] for k in ('x','y','yaw')],
                        object_scale=observation.get('object_scale'),object_tcp_distance_m=float(np.linalg.norm(obj-tcp)),
                        floor_clearance_m=float(obj@plane[:3]+plane[3]),
                        gripper_open_measured=not getattr(self,'command_mode',False) and abs(angles[5]-settings['open_deg'])<.5,
                        gripper_open_estimated=getattr(self,'command_mode',False) and abs(angles[5]-settings['open_deg'])<.5)
                    self.frames.append(value);self.last_stamp=float(queued['stamp']);self.pending.popleft()
                    self.waiting_for_joints=None;processed+=1
            except JointSamplePending as exc:
                with self.lock:
                    if track is self.track and generation==self.generation:self.waiting_for_joints=str(exc)
                return processed
            except (OSError,ValueError,KeyError,TypeError,cv2.error) as exc:
                with self.lock:
                    if track is self.track and generation==self.generation:
                        self.error=str(exc);self.pending.clear();self.waiting_for_joints=None
                raise
        return processed

    def worker(self):
        while True:
            try:
                with self.lock:track=self.track;settings=self.settings;generation=self.generation
                if track is not None:
                    self.process_snapshot(self.snapshot())
            except (OSError,ValueError,KeyError,TypeError,cv2.error) as exc:
                with self.lock:
                    if track is self.track and generation==self.generation and track is not None:self.error=str(exc)
            time.sleep(.03)
    def latest(self):
        with self.lock:
            if self.error:raise ValueError('Наблюдение предмета потеряно: '+self.error)
            if not self.frames or not 0<=time.time()-self.frames[-1]['at']<.5:
                raise ValueError('Нет свежего положения отслеживаемого предмета')
            return dict(self.frames[-1])
    def observe(self, permit, duration=1.2):
        started=time.time();end=time.monotonic()+duration+2
        while time.monotonic()<end:
            permit()
            with self.lock:
                frames=[dict(f) for f in self.frames if f['at']>=started]
            if len(frames)>=3 and frames[-1]['at']-frames[0]['at']>=duration:
                self.latest();return frames
            time.sleep(.05)
        raise ValueError('Недостаточно непрерывных наблюдений предмета')
