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
    old,new=locate(reference,bbox,include_mask=True),locate(current,bbox,include_mask=True)
    if old['outcome']!='candidate' or new['outcome']!='candidate':
        raise ValueError('No fresh separated object at the candidate location')
    points=[np.array([v['position'][k] for k in ('x','y','z')]) for v in (old,new)]
    if np.linalg.norm(points[1]-points[0])>.01:raise ValueError('Candidate moved during recognition')
    appearance = static_appearance(reference, current, old, new)
    return dict(bbox=list(bbox),image_stamp=float(current["stamp"]),
                previous_image_stamp=float(reference["stamp"]),
                position_camera=new["position"],depth_evidence=new["evidence"],
                semantic_identity_verified=False,continuous_from_previous_image=False,**appearance)


def static_appearance(reference, current, old, new):
    """Fixed-view depth silhouette + appearance, without invented optical flow."""
    a, b = old["foreground_mask"], new["foreground_mask"]
    if a.shape != b.shape or reference["rgb"].shape != current["rgb"].shape:
        raise ValueError("Static observation image geometry changed")
    union = a | b
    overlap = a & b
    fraction = float(overlap.sum() / max(1, union.sum()))
    if fraction < .90 or overlap.sum() < 20:
        raise ValueError("Static foreground silhouette changed")
    pixels_old = reference["rgb"][overlap].astype(float)
    pixels_new = current["rgb"][overlap].astype(float)
    color_delta = np.abs(pixels_new - pixels_old)
    if float(np.percentile(color_delta, 90)) > 18 or float(np.mean(color_delta)) > 8:
        raise ValueError("Static foreground appearance changed")
    # A nearby object replacing uniform cloth can be visually indistinguishable.
    # This evidence supports only fixed-view continuity, never semantic identity,
    # physical contact, or attachment to the gripper.
    correlation = None
    if min(float(pixels_old.std()), float(pixels_new.std())) >= 8:
        correlation = float(np.corrcoef(pixels_old.ravel(), pixels_new.ravel())[0, 1])
        if not math.isfinite(correlation) or correlation < .92:
            raise ValueError("Static foreground texture changed")
    return {"appearance_correlation": correlation, "appearance_mean_delta": float(color_delta.mean()),
        "foreground_iou": fraction, "association_kind": "stationary_depth_and_appearance"}


class StationaryDepthObject:
    """Texture-poor object continuity valid only in one stationary RGB-D view."""
    stationary_only = True
    association_kind = "stationary_depth_and_appearance"

    def __init__(self, sample, bbox, stationary_context):
        self.context = stationary_context
        context = stationary_context(sample)
        self.anchor = {"epoch": tuple(context["epoch"]), "servo_deg": list(context["servo_deg"]),
            "odom_xyyaw": list(context["odom_xyyaw"])}
        self.bbox = list(bbox)
        self.reference = sample
        self.geometry = locate(sample, bbox, include_mask=True)
        if self.geometry["outcome"] != "candidate":
            raise ValueError("No resolved stationary foreground")
        self.object_id = uuid.uuid4().hex
        self.stamp = float(sample["stamp"])
        self.frame = str(sample["frame"])
        self.ended = False

    def update(self, sample):
        try:
            if self.ended:
                raise ValueError("Static object association already lost")
            stamp = float(sample["stamp"])
            if not self.stamp < stamp <= self.stamp + .4 or str(sample["frame"]) != self.frame:
                raise ValueError("Static object frame gap or camera change")
            if not np.array_equal(sample["k"], self.reference["k"]) or not np.array_equal(sample["d"], self.reference["d"]):
                raise ValueError("Static camera calibration changed")
            context = self.context(sample)
            if context["epoch"] != self.anchor["epoch"]:
                raise ValueError("Static object reference epoch changed")
            if np.max(np.abs(np.asarray(context["servo_deg"])-self.anchor["servo_deg"])) > .15:
                raise ValueError("Static object camera arm moved")
            pose = np.asarray(context["odom_xyyaw"])
            origin = np.asarray(self.anchor["odom_xyyaw"])
            yaw = abs(math.atan2(math.sin(pose[2]-origin[2]), math.cos(pose[2]-origin[2])))
            if np.linalg.norm(pose[:2]-origin[:2]) > .002 or yaw > .005:
                raise ValueError("Static object camera base moved")
            geometry = locate(sample, self.bbox, include_mask=True)
            if geometry["outcome"] != "candidate":
                raise ValueError("Static foreground depth lost")
            evidence = static_appearance(self.reference, sample, self.geometry, geometry)
            centre = np.asarray([geometry["position"][key] for key in ("x", "y", "z")])
            anchor = np.asarray([self.geometry["position"][key] for key in ("x", "y", "z")])
            if np.linalg.norm(centre-anchor) > .008:
                raise ValueError("Static foreground depth position changed")
            mask = geometry["foreground_mask"]
            pixels = np.argwhere(mask)
            depth = sample["depth"][mask]
            rays = cv2.undistortPoints(pixels[:, ::-1].astype(float).reshape(-1, 1, 2), sample["k"], sample["d"])[:, 0]
            xyz = np.column_stack([rays*depth[:, None], depth])
            low, high = np.percentile(xyz, [5, 95], axis=0)
            self.stamp = stamp
            return {"object_id": self.object_id, "point_camera": centre,
                "association_fraction": evidence["foreground_iou"], "association_kind": self.association_kind,
                "object_extent_camera_m": np.maximum(high-low, .001),
                "object_position_uncertainty_m": max(.002, geometry["position"]["depth_spread_m"]/2),
                "object_scale": float(math.sqrt(mask.sum()/self.geometry["foreground_mask"].sum())),
                "grasp_point_camera": geometry["grasp_surface"]["point_camera"],
                "grasp_pixel": geometry["grasp_surface"]["pixel"],
                "grasp_selection": "observed_stationary_foreground_interior", "association_evidence": evidence,
                "semantic_identity_verified": False, "contact_verified": False, "held_verified": False}
        except (ValueError, KeyError, TypeError, cv2.error):
            self.ended = True
            raise


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
        self.intrinsic = np.asarray(sample["k"]).copy()
        self.distortion = np.asarray(sample["d"]).copy()
    def update(self, sample):
        if self.ended:raise ValueError('Object association already lost')
        stamp=float(sample['stamp'])
        if not self.stamp<stamp<=self.stamp+.4 or str(sample['frame'])!=self.frame:
            self.ended=True;raise ValueError('Object tracking frame gap or frame change')
        if (sample["rgb"].shape[:2] != self.gray.shape or not np.array_equal(sample["k"], self.intrinsic)
                or not np.array_equal(sample["d"], self.distortion)):
            self.ended = True
            raise ValueError("Object tracker camera calibration or image geometry changed")
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
        # Keep the centroid for identity/motion estimates, but never send a
        # concave object's empty centroid to the gripper. Use observed depth
        # at a currently associated foreground feature for the contact target.
        grasp_index = int(np.argmin(np.linalg.norm(xyz-center, axis=1)))
        if np.percentile(np.linalg.norm(xyz-center,axis=1),90)>.15:
            self.ended=True;raise ValueError('Tracked depth no longer forms a small object')
        # Robust visual geometry is part of the evidence contract.  It is not a
        # calibrated object model: the percentiles and median absolute
        # deviation state exactly what this RGB-D observation supports.
        low,high=np.percentile(xyz,[5,95],axis=0)
        extent=np.maximum(high-low,.001)
        radial=np.linalg.norm(xyz-center,axis=1)
        uncertainty=max(.001,float(np.median(np.abs(radial-np.median(radial)))))
        spread=np.ptp(points[:,0],axis=0);scale=float(math.sqrt(max(1.,spread[0]*spread[1]))/self.initial_scale)
        self.gray,self.points,self.stamp=gray,points,stamp
        return dict(object_id=self.object_id,point_camera=center,association_fraction=fraction,object_scale=scale,
                    object_extent_camera_m=extent,object_position_uncertainty_m=uncertainty,
                    grasp_point_camera=xyz[grasp_index].tolist(),grasp_pixel=uv[grasp_index].tolist(),
                    grasp_selection="observed_associated_foreground_feature")


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
        stamp = float(sample["stamp"])
        depth_stamp = float(sample.get("depth_stamp", float("nan")))
        if not math.isfinite(depth_stamp) or abs(stamp-depth_stamp) > .04:
            raise ValueError("RGB-D color/depth exposures are not synchronized")
        if sample["depth"].shape != sample["rgb"].shape[:2]:
            raise ValueError("RGB-D color/depth image geometry is not registered")
        return sample
    def geometry(self, sample, settings):
        if getattr(self,'factory_mode',False):
            from factory_arm_observation import settled_reference,CommandPosePending
            try:state=settled_reference(self.root,self.arm.boot,float(sample["stamp"]))
            except CommandPosePending as exc:raise JointSamplePending(str(exc)) from exc
            angles=state["servo_deg"]
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

    def wait_snapshot(self, permit, after_stamp=None, timeout_s=3.):
        """Bounded non-actuating wait after on-demand inference or a view change."""
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s)
                or not .05 <= timeout_s <= 5):
            raise ValueError("Invalid RGB-D waiting budget")
        if after_stamp is not None and (type(after_stamp) not in (int, float)
                or not math.isfinite(after_stamp)):
            raise ValueError("Invalid required RGB-D exposure time")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            permit()
            sample = None
            try:
                sample = self.snapshot()
            except ValueError as exc:
                if str(exc) != "RGB-D frame is stale":
                    raise
            if sample is not None and (after_stamp is None or float(sample["stamp"]) > after_stamp):
                permit()
                return sample
            time.sleep(.03)
        raise ValueError("No fresh RGB-D exposure within the bounded waiting budget")

    def preview_visibility(self, point, servo_deg, settings):
        """Nominal view at an IK endpoint; never measured or executed feedback."""
        sample = self.snapshot()
        model = self.model()
        with model.lock:
            model.set_state(servo_deg[:5], settings["gripper_linkage_rad"])
            transform = np.asarray(model.state.get_global_link_transform("arm4")) @ np.asarray(
                settings["camera_to_mount"])
        camera_point = (np.linalg.inv(transform) @ np.r_[point, 1.])[:3]
        height, width = sample["rgb"].shape[:2]
        depth = float(camera_point[2])
        pixel = None
        if depth > 0:
            projected, _ = cv2.projectPoints(camera_point.reshape(1, 1, 3), np.zeros(3),
                np.zeros(3), sample["k"], sample["d"])
            pixel = projected[0, 0].tolist()
        in_image = pixel is not None and 1 <= pixel[0] < width - 1 and 1 <= pixel[1] < height - 1
        # Match the existing FeatureObject policy, not an invented sensor range.
        depth_trackable = .15 < depth < 2.
        return {"observable": bool(in_image and depth_trackable), "point_camera_m": camera_point.tolist(),
            "pixel": pixel, "image_size": [width, height], "in_image": bool(in_image),
            "within_tracking_depth_policy": bool(depth_trackable), "tracking_depth_policy_m": [.15, 2.],
            "source": "nominal_ik_camera_projection", "measured": False, "executed": False}
    def start(self, sample, bbox, settings):
        tracker = self.make_tracker(sample, bbox)
        with self.lock:
            self.generation+=1;self.settings=settings;self.track=tracker
            self.frames.clear();self.error=None;self.last_stamp=float(sample['stamp'])
            self.pending.clear();self.enqueued_stamp=self.last_stamp;self.waiting_for_joints=None
        return self.track.object_id
    def stationary_context(self, sample):
        from factory_arm_observation import settled_reference
        stamp = float(sample["stamp"])
        if not 0 <= time.time()-stamp < .4:
            raise ValueError("Stationary tracker requires a fresh RGB-D exposure")
        arm = settled_reference(self.root, self.arm.boot, stamp)
        state = json.loads((self.root/"data/status.json").read_text())
        if state.get("boot_id") != self.arm.boot or not 0 <= time.time()-state.get("at", 0) < .5:
            raise ValueError("Stationary tracker requires fresh matching controller state")
        if not 0 <= state.get("sensor_age", {}).get("odom", math.inf) < .25:
            raise ValueError("Stationary tracker requires fresh measured base odometry")
        for key in ("velocity", "odom_velocity"):
            velocity = np.asarray(state.get(key), dtype=float)
            if velocity.shape != (3,) or not np.isfinite(velocity).all() or np.max(np.abs(velocity)) > .002:
                raise ValueError("Stationary tracker requires stopped base")
        pose = state.get("raw_pose", {})
        values = [pose.get(key) for key in ("x", "y", "yaw")]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
            raise ValueError("Stationary tracker base pose unavailable")
        return {"epoch": (self.arm.boot, arm["at"], arm.get("trajectory_sha256")),
            "servo_deg": arm["servo_deg"], "odom_xyyaw": values}

    def make_tracker(self, sample, bbox):
        """Prefer moving optical correspondence; cloth fallback is static only."""
        try:
            return FeatureObject(sample, bbox)
        except ValueError as exc:
            if str(exc) != "Object has insufficient visible tracking texture" or not self.factory_mode:
                raise
            return StationaryDepthObject(sample, bbox, self.stationary_context)
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
            if getattr(self,"factory_mode",False):
                from factory_arm_observation import settled_reference,CommandPosePending
                try:settled_reference(self.root,self.arm.boot,stamp)
                except CommandPosePending as exc:
                    # Follow identity in image space while the arm moves. No
                    # base-frame coordinate or metric contact claim is produced.
                    if stamp>track.stamp:track.update(sample)
                    self.pending.clear();self.enqueued_stamp=stamp
                    self.waiting_for_joints=str(exc)
                    return 0
            if stamp>self.enqueued_stamp:
                if len(self.pending)>=MAX_PENDING_FRAMES:
                    self.error='Measured camera pose queue overflow';self.pending.clear()
                    raise ValueError(self.error)
                self.pending.append((sample,None if getattr(self,"factory_mode",False) else now+BRACKET_WAIT_SECONDS));self.enqueued_stamp=stamp
        processed=0
        for _ in range(MAX_PENDING_FRAMES):
            with self.lock:
                if track is not self.track or generation!=self.generation or not self.pending:return processed
                queued,deadline=self.pending[0]
            try:
                if deadline is not None and clock()>=deadline:raise ValueError("Measured joint bracket deadline expired")
                transform,tcp,angles=self.geometry(queued,settings)
                if deadline is not None and clock()>=deadline:raise ValueError("Measured joint bracket deadline expired")
                with self.lock:
                    if track is not self.track or generation!=self.generation:return processed
                    observation=track.update(queued)
                    obj=(transform@np.r_[observation['point_camera'],1])[:3]
                    # A rigid transform rotates an extent; absolute rotation is
                    # the conservative axis-aligned base-frame bounding box.
                    extent_camera=np.asarray(observation.get('object_extent_camera_m',[0,0,0]),dtype=float)
                    extent_base=np.abs(transform[:3,:3])@extent_camera
                    plane=np.asarray(settings['floor_plane_base']);pose=self.maps.pose()
                    command_mode=getattr(self,'command_mode',False)
                    gripper_open=abs(angles[5]-settings['open_deg'])<.5
                    value=dict(at=float(queued['stamp']),object_id=observation['object_id'],confidence=observation['association_fraction'],
                        confidence_kind=("stationary_foreground_overlap_not_semantic_probability"
                            if observation.get("association_kind") == "stationary_depth_and_appearance"
                            else "feature_retention_not_semantic_probability"),
                        association_kind=observation.get("association_kind", "reciprocal_feature_flow"),
                        stationary_only=bool(getattr(track, "stationary_only", False)),
                        association_fraction=observation['association_fraction'],identity_association_verified=True,
                        depth_validated=True,camera_pose_measured=not getattr(self,'command_mode',False),
                        camera_pose_validated_for_execution=bool(settings.get('handeye_execution_authorized')),
                        camera_pose_validation_record=settings.get('handeye_physical_validation_record'),
                        camera_pose_source='command_estimate' if getattr(self,'command_mode',False) else 'servo_measurement',frame='base_footprint',
                        object_xyz=obj.tolist(),tcp_xyz=tcp.tolist(),base_xyyaw=[pose[k] for k in ('x','y','yaw')],
                        object_scale=observation.get('object_scale'),object_tcp_distance_m=float(np.linalg.norm(obj-tcp)),
                        object_extent_xyz_m=extent_base.tolist(),
                        object_position_uncertainty_m=float(observation.get('object_position_uncertainty_m',0.)),
                        floor_clearance_m=float(obj@plane[:3]+plane[3]),
                        gripper_open_measured=gripper_open if not command_mode else None,
                        gripper_open_commanded=gripper_open if command_mode else None,
                        gripper_open_estimated=gripper_open if command_mode else None,
                        gripper_aperture_measured_deg=float(angles[5]) if not command_mode else None,
                        gripper_aperture_command_deg=float(angles[5]) if command_mode else None)
                    if "grasp_point_camera" in observation:
                        grasp = (transform@np.r_[observation["grasp_point_camera"], 1])[:3]
                        value.update(grasp_xyz=grasp.tolist(),grasp_pixel=observation.get("grasp_pixel"),
                            grasp_selection=observation.get("grasp_selection"),
                            grasp_contact_verified=False)
                    if "association_evidence" in observation:
                        value["association_evidence"] = observation["association_evidence"]
                    self.frames.append(value);self.last_stamp=float(queued['stamp']);self.pending.popleft()
                    self.waiting_for_joints=None;processed+=1
            except JointSamplePending as exc:
                with self.lock:
                    if track is self.track and generation==self.generation:
                        self.waiting_for_joints=str(exc)
                        if getattr(self,"factory_mode",False):
                            if float(queued["stamp"])>track.stamp:track.update(queued)
                            self.pending.clear()
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
    def tracking_status(self, require_motion=False):
        with self.lock:
            if self.error:raise ValueError("Object association lost: "+self.error)
            if self.track is None or self.track.ended or not 0<=time.time()-self.track.stamp<.4:
                raise ValueError("No fresh image-space object association")
            if require_motion and getattr(self.track, "stationary_only", False):
                raise ValueError("Texture-poor target supports stationary inspection only; moving correspondence unavailable")
            return dict(object_id=self.track.object_id,image_stamp=self.track.stamp,
                identity_association_verified=True,metric_pose_pending=self.waiting_for_joints is not None,
                stationary_only=bool(getattr(self.track, "stationary_only", False)),
                motion_tracking_available=not bool(getattr(self.track, "stationary_only", False)),
                measured_joint_positions=False)
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
