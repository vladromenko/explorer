"""Outcome evidence rules; lost objects and uncalibrated geometry stay unknown."""
import math
import numpy as np

def validated_camera_pose(frame):
    return frame.get('camera_pose_measured') is True or (
        frame.get('camera_pose_source')=='command_estimate' and
        frame.get('camera_pose_validated_for_execution') is True and
        bool(frame.get('camera_pose_validation_record')))


def verify_lift(before, after, calibration_validated=False):
    evidence=dict(verifier='rgbd_lift_v1',calibration_validated=calibration_validated,
                  object_identity_verified=False)
    def unknown(reason):return dict(outcome='unknown',evidence=dict(evidence,reason=reason))
    if not calibration_validated:return unknown('RGB-D, floor and hand-eye calibration unvalidated')
    if len(before)<3 or len(after)<3:return unknown('Three observations required on both sides')
    frames=before+after
    try:
        ids={f['object_id'] for f in frames}
        numeric=np.array([[f['at'],f['confidence'],f['floor_clearance_m'],
                           *f['object_xyz'],*f['tcp_xyz'],*f['base_xyyaw']] for f in frames],dtype=float)
        if numeric.shape!=(len(frames),12) or not np.all(np.isfinite(numeric)):
            return unknown('Missing or invalid geometry')
        if len(ids)!=1 or min(f['confidence'] for f in frames)<.85:
            return unknown('Object identity not stable')
        if any(not f.get('depth_validated',False) for f in frames):return unknown('Invalid depth association')
        if any(f.get('identity_association_verified') is not True or f.get('frame')!='base_footprint' or
               not validated_camera_pose(f) for f in frames):
            return unknown('Object association and validated camera-motion compensation required')
        stamps=[f['at'] for f in frames]
        if any(b<=a for a,b in zip(stamps,stamps[1:])) or stamps[-1]-stamps[0]>30:
            return unknown('Invalid observation timing')
        if after[-1]['at']-after[0]['at']<1:return unknown('Hold shorter than one second')
        base=np.array([f['base_xyyaw'] for f in frames])
        if np.max(np.abs(base-base[0]))>.005:return unknown('Base moved during outcome check')
        pre=np.median([f['object_xyz'] for f in before],axis=0)
        pre_tcp=np.median([f['tcp_xyz'] for f in before],axis=0)
        objects=np.array([f['object_xyz'] for f in after]);tools=np.array([f['tcp_xyz'] for f in after])
        if min(tools[:,2]-pre_tcp[2])<.03:return unknown('No confirmed tool lift')
        evidence['object_identity_verified']=True
        rise=objects[:,2]-pre[2]
        held=(min(rise)>.025 and min(f['floor_clearance_m'] for f in after)>.008 and
              max(np.linalg.norm(objects-tools,axis=1))<.12 and
              np.max(np.ptp(objects-tools,axis=0))<.015)
        stayed=(max(np.linalg.norm(objects-pre,axis=1))<.015 and max(rise)<.01)
        evidence.update(object_rise_m=rise.tolist(),floor_clearance_m=[f['floor_clearance_m'] for f in after],
                        hold_duration_s=after[-1]['at']-after[0]['at'])
        return dict(outcome='success' if held else 'failure' if stayed else 'unknown',evidence=evidence)
    except (KeyError,TypeError,ValueError):return unknown('Incomplete visual evidence')


def verify_place(before, after, zone, calibration_validated=False):
    evidence=dict(verifier='rgbd_place_v1', calibration_validated=calibration_validated)
    def unknown(reason):return dict(outcome='unknown',evidence=dict(evidence,reason=reason))
    if not calibration_validated or len(before)<3 or len(after)<3:
        return unknown('Calibrated geometry and three observations on both sides required')
    try:
        frames=before+after
        if any(f.get('identity_association_verified') is not True or not validated_camera_pose(f) or
               f.get('depth_validated') is not True or f.get('frame')!='base_footprint' for f in frames):
            return unknown('Unverified tracking or camera geometry')
        if len({f['object_id'] for f in frames})!=1:return unknown('Object identity changed')
        stamps=[f['at'] for f in frames]
        if not all(math.isfinite(t) for t in stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):
            return unknown('Invalid observation timing')
        if after[-1]['at']-after[0]['at']<1 or stamps[-1]-stamps[0]>30:
            return unknown('No recent stable placement interval')
        obj=np.array([f['object_xyz'] for f in after]);tcp=np.array([f['tcp_xyz'] for f in after])
        base=np.array([f['base_xyyaw'] for f in frames]);center=np.asarray(zone['center_xyz'])
        numeric=np.r_[obj.ravel(),tcp.ravel(),base.ravel(),center.ravel(),zone['radius_m']]
        if obj.shape!=(len(after),3) or tcp.shape!=obj.shape or center.shape!=(3,) or not np.isfinite(numeric).all():
            return unknown('Invalid geometry')
        if np.max(np.abs(base-base[0]))>.005:return unknown('Base moved during verification')
        if any(f.get('gripper_open_measured') is not True for f in after):return unknown('Opening not measured')
        stationary=np.max(np.ptp(obj,axis=0))<.012
        supported=np.max(np.abs(obj[:,2]-center[2]))<zone['support_tolerance_m']
        inside=np.max(np.linalg.norm(obj[:,:2]-center[:2],axis=1))<zone['radius_m']
        detached=np.min(np.linalg.norm(tcp-obj,axis=1))>.06
        evidence.update(stationary=bool(stationary),supported=bool(supported),inside=bool(inside),detached=bool(detached))
        # An object still moving with the withdrawing hand cannot count as placed.
        return dict(outcome='success' if stationary and supported and inside and detached else 'unknown',evidence=evidence)
    except (KeyError,TypeError,ValueError):return unknown('Incomplete placement evidence')
