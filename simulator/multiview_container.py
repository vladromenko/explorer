"""Registered, fully observed rim and opening evidence across camera views.

Two complete rims and two independent actual opening measurements are required.
This supplements the existing single-view detector; partial geometry alone
never becomes a destination, and this does not establish a collision-free path.
"""
import numpy as np
from itertools import product
from scipy.spatial import cKDTree,ConvexHull,QhullError

from .rgbd_observation import ObservationRefused,deproject_depth
from .observed_free_space import DepthFreeSpace


def opening_displacement(first,second):
    """Compare the observed opening, not the midpoint of a cropped wall.

    The lower visible wall changes with camera height. It cannot move the
    identity anchor of the same fixed rim up or down by half that crop.
    """
    def anchor(box):
        lo,hi=np.asarray(box['bounds_min'],float),np.asarray(box['bounds_max'],float)
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite(np.r_[lo,hi]).all() or np.any(hi<=lo):
            raise ObservationRefused('Некорректная измеренная граница корзины')
        return np.r_[(lo[:2]+hi[:2])/2,hi[2]]
    return float(np.linalg.norm(anchor(first)-anchor(second)))


def _position(record):
    return np.asarray(record['frame']['optical_to_world'],float)[:3,3]


def _distinct(first,second):
    return (first['frame']['frame_id']!=second['frame']['frame_id'] and
            np.linalg.norm(_position(first)-_position(second))>=.02)


def _registered_rim(record,lo,hi,rim_points):
    frame=record['frame'];depth=np.asarray(record['depth'],float);robot=np.asarray(record['robot_depth'],float)
    if robot.shape!=depth.shape or not np.isfinite(robot).all():return None
    points=deproject_depth(depth,frame['intrinsics'],frame['optical_to_world'])
    valid=(np.isfinite(points).all(-1)&(depth>=.035)&(depth<=2.5)&
           ~(np.isfinite(robot)&(robot>0)&(np.abs(depth-robot)<=.008)))
    center=(lo+hi)/2;half=(hi-lo)/2;relative=np.abs(points[:,:,:2]-center[:2])/half[:2]
    selected=valid&(np.abs(points[:,:,2]-hi[2])<=.025)&(relative.max(-1)>=.65)&(relative.max(-1)<=1.2)
    anchors=points[selected]
    if len(anchors)<20:return None
    distances,_=cKDTree(rim_points).query(anchors,k=1);anchors=anchors[distances<=.025]
    if len(anchors)<20:return None
    angles=np.arctan2(anchors[:,1]-center[1],anchors[:,0]-center[0])
    sectors=np.unique(np.floor((angles+np.pi)/(np.pi/2)).astype(int).clip(0,3))
    if len(sectors)<2:return None
    return dict(registered_rim_pixels=len(anchors),registered_rim_sectors=len(sectors))


def _opening(record,lo,hi,rim_points,uncertainty):
    frame=record['frame'];depth=np.asarray(record['depth'],float)
    robot=np.asarray(record['robot_depth'],float)
    if robot.shape!=depth.shape or not np.isfinite(robot).all() or np.any(robot<=0):return None
    points=deproject_depth(depth,frame['intrinsics'],frame['optical_to_world'])
    valid=(np.isfinite(points).all(-1)&(depth>=.035)&(depth<=2.5)&
           ~(np.isfinite(robot)&(robot>0)&(np.abs(depth-robot)<=.008)))
    center=(lo+hi)/2;half=(hi-lo)/2
    # Measure distance from the actual registered rim polygon, rather than
    # assuming that the centre of a bounding rectangle must be visible. The
    # same uncertainty + 12 mm inset is used by placement verification.
    try:hull=ConvexHull(rim_points[:,:2])
    except QhullError:return None
    signed=(np.einsum('hwj,kj->hwk',points[:,:,:2],hull.equations[:,:2])+
            hull.equations[:,2])
    margin=max(.02,uncertainty+.012)
    inside=signed.max(-1)<-margin
    if np.count_nonzero(valid&inside&(points[:,:,2]>=hi[2]-.035))>=8:
        return dict(obstructed=True)
    opening=valid&inside&(points[:,:,2]<hi[2]-.035)&(points[:,:,2]>=-.003)
    count=int(opening.sum())
    if count<8:return None
    patch=points[opening]
    if max(np.ptp(patch[:,:2],axis=0))<.01:return None
    # A cropped close-up needs measured rim correspondences, rather than an
    # unanchored patch of floor elsewhere in the scene.
    relative=np.abs(points[:,:,:2]-center[:2])/half[:2]
    top=valid&(np.abs(points[:,:,2]-hi[2])<=.025)&(relative.max(-1)>=.65)&(relative.max(-1)<=1.2)
    anchors=points[top]
    if len(anchors)<20:return None
    distances,_=cKDTree(rim_points).query(anchors,k=1)
    anchors=anchors[distances<=.025]
    if len(anchors)<20:return None
    angles=np.arctan2(anchors[:,1]-center[1],anchors[:,0]-center[0])
    sectors=np.unique(np.floor((angles+np.pi)/(np.pi/2)).astype(int).clip(0,3))
    if len(sectors)<2:return None
    return dict(frame_id=frame['frame_id'],opening_pixels=count,
                registered_rim_pixels=len(anchors),registered_rim_sectors=len(sectors),
                required_interior_margin_m=margin,
                measured_minimum_interior_margin_m=float(-signed.max(-1)[opening].max()))


def _ray_opening(available,lo,hi,rim_points,uncertainty,now,require_fresh):
    """A measured empty volume below the rim is opening evidence, not a floor assumption."""
    try:hull=ConvexHull(rim_points[:,:2])
    except QhullError:return None
    sensors=[]
    for record in available:
        registration=_registered_rim(record,lo,hi,rim_points)
        if registration is None:continue
        depth=np.array(record['depth'],float,copy=True);robot=np.asarray(record['robot_depth'],float)
        if robot.shape!=depth.shape:continue
        depth[~np.isfinite(robot)|(robot<=0)|(np.abs(depth-robot)<=.008)]=np.nan
        sensors.append((record,DepthFreeSpace(depth,record['frame']['intrinsics'],
            np.asarray(record['frame']['optical_to_world']),record['frame']['frame_id'],record['frame']['simulation_time']),registration))
    margin=max(.02,uncertainty+.012)
    offsets=np.array(list(product((-.002,.002),repeat=3)))
    centers=[np.array([x,y,hi[2]-.0355]) for x in np.arange(lo[0],hi[0],.006)
             for y in np.arange(lo[1],hi[1],.006)]
    # Entire 4 mm cells, not isolated corners or guessed interior coordinates,
    # must be witnessed by at least eight real depth rays in each view.
    for latest,sensor,registration in reversed(sensors):
        if require_fresh and now-float(latest['frame']['simulation_time'])>.25:continue
        for center in centers:
            box=center+offsets
            if np.any(np.einsum('ij,kj->ik',box[:,:2],hull.equations[:,:2])+hull.equations[:,2]>=-margin):continue
            proof=sensor.box_witness(box)
            if not proof or proof['covered_depth_pixels']<8:continue
            proof.update(registration)
            for earlier,other,previous_registration in reversed(sensors):
                if not _distinct(earlier,latest):continue
                previous=other.box_witness(box)
                if not previous or previous['covered_depth_pixels']<8:continue
                previous.update(previous_registration)
                return latest,earlier,dict(source='two_view_measured_free_volume',
                    center_world_m=center.tolist(),box_corners_world_m=box.tolist(),
                    center_depth_below_rim_m=.0355,
                    views=[previous,proof],unseen_floor_assumed=False)
    return None


def _combine(records,*,now_simulation_time,max_map_age_s=30.,require_fresh=True):
    """Require complete observed boundary plus independent interior pixels.

    The scene is treated as static only within this bounded diagnostic window.
    The latest opening view must still satisfy the existing 250 ms freshness
    gate. All frames must be real depth captures with resolved robot geometry.
    """
    if not np.isfinite(now_simulation_time):raise ObservationRefused('Invalid observation time')
    available=[]
    for record in records:
        frame=record['frame'];age=now_simulation_time-float(frame['simulation_time'])
        if (0.<=age<=max_map_age_s and frame.get('frame_id') and
            frame.get('robot_self_filter',{}).get('available') is True):
            available.append(record)
    # Repeated stopped captures at the same camera pose add no independent
    # baseline. Keep the newest at each such pose so old opening pixels cannot
    # contradict a newly occupied basket. All raw records remain preserved;
    # every genuinely distinct viewpoint still participates in the same proof.
    unique=[]
    for record in reversed(available):
        matrix=np.asarray(record['frame']['optical_to_world'])
        if any(record['frame']['intrinsics']==old['frame']['intrinsics']
                and np.linalg.norm(_position(record)-_position(old))<.005
                and np.linalg.norm(matrix[:3,:3]-np.asarray(old['frame']['optical_to_world'])[:3,:3])<.01
                and np.shape(record['depth'])==np.shape(old['depth'])
                and np.allclose(record['depth'],old['depth'],atol=.002,rtol=0.,equal_nan=True)
                and np.allclose(record['robot_depth'],old['robot_depth'],atol=.002,rtol=0.,equal_nan=True)
                and bool(record['frame'].get('observed_rims'))==bool(old['frame'].get('observed_rims'))
                for old in unique):continue
        unique.append(record)
    available=list(reversed(unique))
    rims=[]
    for record in available:
        candidates=record['frame'].get('observed_rims',[])
        if len(candidates)!=1:continue
        rim=candidates[0];evidence=rim['evidence']
        if (rim.get('robot_mask_resolved') is True and evidence.get('rim_sectors')==4
            and evidence.get('touches_image_border') is False
            and evidence.get('shape')=='complete_rim_without_opening_proof'):
            rims.append((record,rim))
    for second,current in reversed(rims):
        for first,previous in reversed(rims):
            if not _distinct(first,second):continue
            lower=np.asarray([previous['bounds_min'],current['bounds_min']],float)
            upper=np.asarray([previous['bounds_max'],current['bounds_max']],float)
            if (lower.shape!=(2,3) or upper.shape!=(2,3) or
                not np.isfinite(np.r_[lower,upper]).all()):continue
            disagreement=float(max(np.abs(lower[0]-lower[1]).max(),np.abs(upper[0]-upper[1]).max()))
            if disagreement>.025:continue
            if np.linalg.norm(np.asarray(previous['center'])-current['center'])>.025:continue
            lo,hi=lower.mean(0),upper.mean(0)
            if min(hi[:2]-lo[:2])<.09 or min(hi-lo)<=0.:continue
            points=np.asarray(previous['evidence']['rim_points']+current['evidence']['rim_points'],float)
            if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():continue
            openings=[];contradiction=False
            uncertainty=max(.012,.006+disagreement/2)
            if uncertainty>.02:continue
            for record in available:
                proof=_opening(record,lo,hi,points,uncertainty)
                if proof and proof.get('obstructed'):
                    if record is first or record is second:contradiction=True
                    continue
                if proof:openings.append((record,proof))
            if contradiction:continue
            for latest,proof in reversed(openings):
                if require_fresh and now_simulation_time-float(latest['frame']['simulation_time'])>.25:continue
                for earlier,old in reversed(openings):
                    if not _distinct(earlier,latest):continue
                    uncertainty=max(.012,.006+disagreement/2)
                    if uncertainty>.02:continue
                    return dict(center=((lo+hi)/2).tolist(),bounds_min=lo.tolist(),bounds_max=hi.tolist(),
                        uncertainty_m=uncertainty,confidence=.75,robot_mask_resolved=True,
                        semantic_identity='unverified',frame_id=proof['frame_id'],
                        simulation_time=latest['frame']['simulation_time'],evidence=dict(
                            shape='raised_walls_with_visible_opening',observation_method='registered_rim_and_opening_views',
                            complete_rim_frames=[first['frame']['frame_id'],second['frame']['frame_id']],
                            opening_views=[old,proof],rim_sectors=4,rim_height_m=float(hi[2]),
                            bound_disagreement_m=disagreement,
                            rim_camera_baseline_m=float(np.linalg.norm(_position(first)-_position(second))),
                            opening_camera_baseline_m=float(np.linalg.norm(_position(earlier)-_position(latest))),
                            bounds_are_visible_estimate=True,motion_clearance_not_established=True,
                            real_calibration_validated=False))
            ray=_ray_opening(available,lo,hi,points,uncertainty,now_simulation_time,require_fresh)
            if ray is not None:
                latest,earlier,proof=ray
                return dict(center=((lo+hi)/2).tolist(),bounds_min=lo.tolist(),bounds_max=hi.tolist(),
                    uncertainty_m=uncertainty,confidence=.75,robot_mask_resolved=True,
                    semantic_identity='unverified',frame_id=latest['frame']['frame_id'],
                    simulation_time=latest['frame']['simulation_time'],evidence=dict(
                        shape='raised_walls_with_visible_opening',observation_method='registered_rim_and_measured_empty_volume',
                        complete_rim_frames=[first['frame']['frame_id'],second['frame']['frame_id']],
                        opening_proof=proof,rim_sectors=4,rim_height_m=float(hi[2]),bound_disagreement_m=disagreement,
                        rim_camera_baseline_m=float(np.linalg.norm(_position(first)-_position(second))),
                        opening_camera_baseline_m=float(np.linalg.norm(_position(earlier)-_position(latest))),
                        bounds_are_visible_estimate=True,motion_clearance_not_established=True,
                        real_calibration_validated=False))
    raise ObservationRefused('Нет полного наблюдаемого бортика и двух согласованных обзоров внутренней части')


def combine_registered_container(records,*,now_simulation_time):
    return _combine(records,now_simulation_time=now_simulation_time)


def refresh_registered_container(records,candidate,latest):
    """Recheck a known two-view opening on NEW pixels without pair search.

    Does not update a timestamp on cached evidence. Requires the same observed
    rim correspondences, current interior/free-volume witnesses, and an older
    independent opening view. Original rim/uncertainty requirements remain.
    """
    evidence=candidate['evidence'];now=latest['frame']['simulation_time']
    indexed={r['frame']['frame_id']:r for r in records if 0<=now-r['frame']['simulation_time']<=30.}
    try:rim_records=[indexed[key] for key in evidence['complete_rim_frames']]
    except KeyError:raise ObservationRefused('Original observed rim views expired') from None
    points=[]
    for record in rim_records:
        rims=record['frame'].get('observed_rims',[])
        if len(rims)!=1:raise ObservationRefused('Original complete rim is missing')
        points.extend(rims[0]['evidence']['rim_points'])
    points=np.asarray(points,float);lo,hi=np.asarray(candidate['bounds_min']),np.asarray(candidate['bounds_max'])
    new=_opening(latest,lo,hi,points,candidate['uncertainty_m'])
    if new and new.get('obstructed'):raise ObservationRefused('Fresh opening is obstructed')
    old_views=evidence.get('opening_views') or evidence.get('opening_proof',{}).get('views',[])
    independent=next((v for v in old_views if v['frame_id'] in indexed and _distinct(indexed[v['frame_id']],latest)),None)
    if independent is None:raise ObservationRefused('Fresh opening lacks an independent recorded baseline')
    refreshed=dict(evidence,refresh_source='new_registered_opening_pixels')
    if new is not None and 'opening_views' in evidence:refreshed['opening_views']=[independent,new]
    elif evidence.get('opening_proof'):
        registration=_registered_rim(latest,lo,hi,points)
        sensor=DepthFreeSpace(latest['depth'],latest['frame']['intrinsics'],np.asarray(latest['frame']['optical_to_world']),
            latest['frame']['frame_id'],now)
        witness=sensor.box_witness(evidence['opening_proof']['box_corners_world_m'])
        if not registration or not witness or witness['covered_depth_pixels']<8:raise ObservationRefused('Fresh measured opening volume is unconfirmed')
        witness.update(registration)
        refreshed['opening_proof']=dict(evidence['opening_proof'],views=[independent,witness])
    else:raise ObservationRefused('Fresh opening is not visible')
    refreshed['opening_camera_baseline_m']=float(np.linalg.norm(_position(indexed[independent['frame_id']])-_position(latest)))
    return dict(candidate,frame_id=latest['frame']['frame_id'],simulation_time=now,evidence=refreshed)


def opening_refresh_hint(records,*,now_simulation_time):
    """Return an old own-robot camera command only; never an actionable destination."""
    candidate=_combine(records,now_simulation_time=now_simulation_time,require_fresh=False)
    source=next(r for r in records if r['frame']['frame_id']==candidate['frame_id'])
    command=np.asarray(source.get('command'),float)
    if command.shape!=(6,) or not np.isfinite(command).all():
        raise ObservationRefused('Нет сохранённой собственной обзорной позы')
    return dict(command=command.copy(),source_frame=candidate['frame_id'],actionable_destination=False,
        occupied_box={k:candidate[k] for k in ('bounds_min','bounds_max','uncertainty_m')})
