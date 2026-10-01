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
    """Verify a placement from object support and tool/object separation.

    Command-mode arm coordinates are useful camera-pose provenance but are not
    servo measurements.  A successful place therefore depends on independent
    RGB-D evidence: the same tracked object remains supported in the requested
    footprint while the tool withdraws and separation increases.
    """
    evidence=dict(verifier='rgbd_place_v2', calibration_validated=calibration_validated)
    def unknown(reason):return dict(outcome='unknown',evidence=dict(evidence,reason=reason))
    if not calibration_validated or len(before)<3 or len(after)<3:
        return unknown('Calibrated geometry and three observations on both sides required')
    try:
        frames=before+after
        if any(f.get('identity_association_verified') is not True or not validated_camera_pose(f) or
               f.get('depth_validated') is not True or f.get('frame')!='base_footprint' for f in frames):
            return unknown('Unverified tracking or camera geometry')
        if len({f['object_id'] for f in frames})!=1:return unknown('Object identity changed')
        if min(float(f.get('confidence',0)) for f in frames)<.85:return unknown('Object association confidence too low')
        stamps=[f['at'] for f in frames]
        if not all(math.isfinite(t) for t in stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):
            return unknown('Invalid observation timing')
        if after[-1]['at']-after[0]['at']<1 or stamps[-1]-stamps[0]>30:
            return unknown('No recent stable placement interval')
        before_obj=np.array([f['object_xyz'] for f in before]);before_tcp=np.array([f['tcp_xyz'] for f in before])
        obj=np.array([f['object_xyz'] for f in after]);tcp=np.array([f['tcp_xyz'] for f in after])
        base=np.array([f['base_xyyaw'] for f in frames]);center=np.asarray(zone['center_xyz'])
        extents=np.array([f['object_extent_xyz_m'] for f in after])
        uncertainty=np.array([f.get('object_position_uncertainty_m',float('nan')) for f in after])
        numeric=np.r_[before_obj.ravel(),before_tcp.ravel(),obj.ravel(),tcp.ravel(),base.ravel(),
                      center.ravel(),extents.ravel(),uncertainty,zone['radius_m'],zone['support_tolerance_m']]
        if (before_obj.shape!=(len(before),3) or before_tcp.shape!=before_obj.shape or
                obj.shape!=(len(after),3) or tcp.shape!=obj.shape or extents.shape!=obj.shape or
                center.shape!=(3,) or not np.isfinite(numeric).all()):
            return unknown('Invalid geometry')
        if np.max(np.abs(base-base[0]))>.005:return unknown('Base moved during verification')
        tolerance=float(zone['support_tolerance_m'])+float(np.max(uncertainty))
        bottom=obj[:,2]-extents[:,2]/2
        support_error=bottom-center[2]
        stationary=np.max(np.ptp(obj,axis=0))<.012+float(np.max(uncertainty))
        supported=np.max(np.abs(support_error))<=tolerance
        footprint_radius=np.linalg.norm(extents[:,:2],axis=1)/2+uncertainty
        zone_distance=np.linalg.norm(obj[:,:2]-center[:2],axis=1)+footprint_radius
        inside=np.max(zone_distance)<=float(zone['radius_m'])
        pre_object=np.median(before_obj,axis=0);post_object=np.median(obj,axis=0)
        pre_tool=np.median(before_tcp,axis=0);post_tool=np.median(tcp,axis=0)
        tool_withdrawal=float(np.linalg.norm(post_tool-pre_tool))
        object_displacement=float(np.linalg.norm(post_object-pre_object))
        pre_separation=float(np.linalg.norm(pre_tool-pre_object))
        post_separation=float(np.linalg.norm(post_tool-post_object))
        separation_gain=post_separation-pre_separation
        object_persistent=object_displacement<=.02+float(np.max(uncertainty))
        detached=(tool_withdrawal>=.04 and separation_gain>=.03 and post_separation>=.06 and object_persistent)
        follows_tool=(tool_withdrawal>=.04 and object_displacement>=max(.025,.55*tool_withdrawal) and
                      separation_gain<.02)
        opening_measured=all(f.get('gripper_open_measured') is True for f in after)
        opening_commanded=all(f.get('gripper_open_commanded') is True or
                              f.get('gripper_open_estimated') is True for f in after)
        evidence.update(stationary=bool(stationary),supported=bool(supported),inside=bool(inside),
                        detached=bool(detached),object_persistent=bool(object_persistent),
                        follows_tool=bool(follows_tool),tool_withdrawal_m=tool_withdrawal,
                        object_displacement_m=object_displacement,separation_gain_m=separation_gain,
                        post_separation_m=post_separation,support_error_m=support_error.tolist(),
                        zone_extent_distance_m=zone_distance.tolist(),gripper_open_measured=opening_measured,
                        gripper_open_commanded=opening_commanded,
                        gripper_open_provenance='measured' if opening_measured else
                                                'command_estimate' if opening_commanded else 'unavailable')
        if stationary and supported and inside and detached:
            return dict(outcome='success',evidence=evidence)
        if follows_tool:
            return dict(outcome='failure',evidence=dict(evidence,reason='Object followed the withdrawing tool'))
        if not inside:
            return dict(outcome='failure',evidence=dict(evidence,reason='Object is outside the target zone'))
        if np.min(support_error)<-tolerance:
            return dict(outcome='failure',evidence=dict(evidence,reason='Object fell below the support surface'))
        return dict(outcome='unknown',evidence=dict(evidence,reason='Visual evidence does not prove stable separation and support'))
    except (KeyError,TypeError,ValueError):return unknown('Incomplete placement evidence')
