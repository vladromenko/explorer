"""Observed shallow-tray evidence, separate from tall-basket acceptance."""
import numpy as np
from .rgbd_observation import deproject_depth,_clusters,_candidate,ObservationRefused


def observed_trays(record):
    frame=record['frame'];depth=np.asarray(record['depth']);robot=np.asarray(record['robot_depth'])
    points=deproject_depth(depth,frame['intrinsics'],frame['optical_to_world'])
    valid=np.isfinite(points).all(-1)&(depth>=.035)&(depth<=2.5)
    valid&=~(np.isfinite(robot)&(robot>0)&(abs(depth-robot)<=.008))
    raised=valid&(points[:,:,2]>=.016)&(points[:,:,2]<=.07)
    results=[]
    for cloud,pixels in _clusters(points,raised,min_pixels=40,cell_m=.015):
        lo,hi=np.quantile(cloud,[.02,.98],axis=0);span=hi-lo
        if not (.12<=span[0]<=.4 and .12<=span[1]<=.4):continue
        rim=cloud[cloud[:,2]>=hi[2]-.006];xy=(lo[:2]+hi[:2])/2
        if len(rim)<30:continue
        angles=np.arctan2(rim[:,1]-xy[1],rim[:,0]-xy[0]);sectors=np.unique(np.floor((angles+np.pi)/(np.pi/2)).clip(0,3))
        if len(sectors)!=4:continue
        h,w=depth.shape
        if np.any((pixels[:,0]<2)|(pixels[:,0]>=h-2)|(pixels[:,1]<2)|(pixels[:,1]>=w-2)):continue
        inner=valid&np.all(points[:,:,:2]>lo[:2]+.025,axis=-1)&np.all(points[:,:,:2]<hi[:2]-.025,axis=-1)
        floor=points[inner&(points[:,:,2]>=.003)&(points[:,:,2]<=hi[2]-.012)]
        if len(floor)<30:continue
        z=float(np.median(floor[:,2]))
        if np.quantile(abs(floor[:,2]-z),.9)>.004:continue
        if min(np.ptp(floor[:,:2],axis=0))<.06:continue
        item=_candidate((lo+hi)/2,lo,hi,.012,.75,dict(shape='observed_shallow_tray',
            measured_support_height_m=z,visible_support_pixels=len(floor),rim_sectors=4,semantic_identity_proven=False))
        results.append(dict(item,frame_id=frame['frame_id'],simulation_time=frame['simulation_time']))
    return results


def combine_trays(views):
    for b in reversed(views):
        for a in reversed(views):
            if a['frame_id']==b['frame_id'] or np.linalg.norm(np.asarray(a['camera_position'])-b['camera_position'])<.02:continue
            first,second=a['candidate'],b['candidate']
            if first.get('frame_id')!=a['frame_id'] or second.get('frame_id')!=b['frame_id']:continue
            age=second['simulation_time']-first['simulation_time']
            if not 0.<age<=30.:continue
            if max(np.max(abs(np.asarray(first[key])-second[key])) for key in ('bounds_min','bounds_max'))>.025:continue
            if abs(first['evidence']['measured_support_height_m']-second['evidence']['measured_support_height_m'])>.006:continue
            return dict(second,evidence=dict(second['evidence'],multiview_consistent=True,
                source_frames=[a['frame_id'],b['frame_id']],real_calibration_validated=False))
    raise ObservationRefused('Нужны два независимых обзора бортика и поверхности подноса')


def verify_on_tray(first,second,*,destination,held_reference,before_release,release_jaw,jaw_xyz):
    from .placement_evidence import _color_matches
    unknown=lambda text:dict(outcome='unknown',reason=text)
    try:
        if (first['frame_id']==second['frame_id'] or first['simulation_time']<=before_release['simulation_time'] or
            second['simulation_time']-first['simulation_time']<.3):return unknown('Нет двух свежих кадров размещения')
        lo,hi=np.asarray(destination['bounds_min']),np.asarray(destination['bounds_max']);z=destination['evidence']['measured_support_height_m']
        if not destination['evidence'].get('multiview_consistent') or destination['uncertainty_m']>.02:return unknown('Опора не подтверждена')
        def matches(frame):
            output=[]
            for c in frame.get('object_candidates',[]):
                lower,upper,center=np.asarray(c['bounds_min']),np.asarray(c['bounds_max']),np.asarray(c['center'])
                e=c.get('evidence',{})
                if (e.get('self_occlusion_not_resolved') is False and e.get('visible_pixels',0)>=100 and
                    c.get('uncertainty_m',1.)<=.02 and _color_matches(c,held_reference) and
                    np.all(lower[:2]>lo[:2]+.025) and np.all(upper[:2]<hi[:2]-.025) and
                    abs(lower[2]-z)<=.012 and np.linalg.norm(center-jaw_xyz)>.065):output.append(c)
            return output
        a,b=matches(first),matches(second)
        if len(a)!=1 or len(b)!=1:return unknown('Размещённый предмет не виден однозначно')
        c=np.asarray(b[0]['center']);prior=[p for p in before_release.get('object_candidates',[]) if _color_matches(p,held_reference)]
        held=[p for p in prior if np.linalg.norm(np.asarray(p['center'])-release_jaw)<.08]
        if len(held)!=1 or held[0]['center'][2]-c[2]<.025:return unknown('Отделение от захвата не подтверждено')
        if any(np.linalg.norm(np.asarray(p['center'])-c)<.04 for p in prior):return unknown('Похожий предмет уже находился на опоре')
        if np.linalg.norm(np.asarray(a[0]['center'])-c)>.008:return unknown('Предмет ещё движется')
        low,high=np.asarray(b[0]['bounds_min']),np.asarray(b[0]['bounds_max'])
        free=np.asarray(before_release.get('surface_points',[]));support=np.asarray(second.get('surface_points',[]))
        if free.ndim!=2 or support.ndim!=2:return unknown('Поверхность опоры не наблюдается')
        footprint=np.all(free[:,:2]>low[:2],axis=1)&np.all(free[:,:2]<high[:2],axis=1)&(abs(free[:,2]-z)<.004)
        outside=np.any((support[:,:2]<low[:2]-.003)|(support[:,:2]>high[:2]+.003),axis=1)
        near=np.linalg.norm(support[:,:2]-c[:2],axis=1)<.09
        inside=np.all(support[:,:2]>lo[:2]+.025,axis=1)&np.all(support[:,:2]<hi[:2]-.025,axis=1)
        if np.count_nonzero(footprint)<6 or np.count_nonzero(outside&near&inside&(abs(support[:,2]-z)<.004))<6:return unknown('До и после размещения не видна опора')
        return dict(outcome='success',evidence=dict(confirmed=True,source='two_frame_observed_supported_tray',
            frames=[first['frame_id'],second['frame_id']],support_height_m=z,identity_proven=False))
    except (ValueError,TypeError,KeyError,IndexError):return unknown('Неполные наблюдения опоры')
