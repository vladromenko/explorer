"""Bounded scene-preparation proposals and independent visual verification."""
import math
import numpy as np


ALLOWED_KINDS=('light_rigid','soft_cloth','paper')


def propose(object_geometry, gripper, obstacles, permission):
    if permission.get('target_contact') is not True:raise ValueError('Target-contact permission is absent')
    if object_geometry.get('kind') not in ALLOWED_KINDS:raise ValueError('Object kind is not allowed for scene preparation')
    if object_geometry.get('mass_g',1000)>200:raise ValueError('Only light objects are allowed')
    center=np.asarray(object_geometry.get('center_xyz'),float);extent=np.asarray(object_geometry.get('extent_xyz'),float)
    if center.shape!=(3,) or extent.shape!=(3,) or not np.isfinite(np.r_[center,extent]).all():raise ValueError('Invalid target geometry')
    if np.any(extent<=0) or np.max(extent)>.35:raise ValueError('Target geometry is outside the preparation envelope')
    candidates=[]
    for direction in ((1,0),(-1,0),(0,1),(0,-1)):
        delta=np.asarray([direction[0],direction[1],0.],float)*min(.06,max(.025,float(max(extent[:2]))))
        goal=center+delta;clear=True
        for obstacle in obstacles:
            obstacle_center=np.asarray(obstacle['center_xyz'],float);obstacle_extent=np.asarray(obstacle['extent_xyz'],float)
            if np.all(np.abs(goal-obstacle_center)<(extent+obstacle_extent)/2+.015):clear=False
        if clear:
            candidates.append(dict(kind='bounded_push',start_xyz=center.tolist(),goal_xyz=goal.tolist(),
                                   direction=list(direction),maximum_travel_m=float(np.linalg.norm(delta)),
                                   permitted_contact_object=object_geometry['id'],collision_policy='target_only'))
    if object_geometry.get('kind')=='soft_cloth' and object_geometry.get('accessible_edge') is True:
        candidates.append(dict(kind='bounded_edge_pull',start_xyz=center.tolist(),goal_xyz=(center+[0,-.04,0]).tolist(),
                               maximum_travel_m=.04,permitted_contact_object=object_geometry['id'],
                               collision_policy='target_only',requires_executor='verified_soft_edge_pull'))
    return candidates


def verify(before, after, proposal):
    evidence=dict(verifier='scene_change_v1',target=proposal.get('permitted_contact_object'))
    if len(before)<3 or len(after)<3:return dict(state='unknown',reason='Three observations required',evidence=evidence)
    try:
        if len({row['object_id'] for row in before+after})!=1:return dict(state='unknown',reason='Target identity changed',evidence=evidence)
        if any(row.get('identity_association_verified') is not True or row.get('depth_validated') is not True for row in before+after):
            return dict(state='unknown',reason='Tracking or depth is not verified',evidence=evidence)
        a=np.asarray([row['object_xyz'] for row in before],float);b=np.asarray([row['object_xyz'] for row in after],float)
        collateral=max(float(row.get('collateral_displacement_m',0)) for row in after)
        delta=np.median(b,axis=0)-np.median(a,axis=0);requested=np.asarray(proposal['goal_xyz'])-np.asarray(proposal['start_xyz'])
        progress=float(delta@requested/max(np.linalg.norm(requested),1e-9));moved=float(np.linalg.norm(delta))
        stable=np.max(np.ptp(b,axis=0))<.015
        evidence.update(displacement_m=delta.tolist(),progress_m=progress,stable=bool(stable),collateral_displacement_m=collateral)
        if collateral>.02:return dict(state='failure',reason='Non-target object moved too far',evidence=evidence)
        if stable and progress>=.015 and moved<=float(proposal['maximum_travel_m'])+.025:
            return dict(state='success',reason='Target moved in the intended direction and settled',evidence=evidence)
        if stable and moved<.008:return dict(state='failure',reason='Target did not move',evidence=evidence)
        return dict(state='unknown',reason='Scene did not settle into a verified state',evidence=evidence)
    except (KeyError,TypeError,ValueError):return dict(state='unknown',reason='Incomplete preparation evidence',evidence=evidence)
