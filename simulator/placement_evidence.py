"""Conservative post-release evidence from registered wrist RGBD observations.

This is geometric association with a previously tracked single target, not
semantic recognition. No simulator object poses, labels or contacts enter it.
Occlusion, ambiguity or an unseen support surface leave the result unknown.
"""
import numpy as np


def _color_matches(candidate, reference):
    try:
        a=np.asarray(candidate['appearance']['rgb_median'],float)
        b=np.asarray(reference['appearance']['rgb_median'],float)
        return (a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all()
                and a.sum()>.05 and b.sum()>.05 and
                np.linalg.norm(a/a.sum()-b/b.sum())<.08)
    except (KeyError,TypeError,ValueError):return False


def verify_released_target(first,second,*,destination,held_reference,release_xyz,jaw_xyz,
                           before_release,release_jaw_xyz):
    """Require an associated stationary target inside a measured opening.

    Call only after a visually verified hold, release and withdrawal. Two fresh
    frames must bracket at least 0.3 simulation seconds. A surrounding measured
    surface must lie immediately below the material; still-in-gripper, rim,
    outside, static unrelated geometry and duplicate candidates are rejected.
    """
    unknown=lambda reason:dict(outcome='unknown',reason=reason)
    try:
        times=np.asarray([before_release['simulation_time'],first['simulation_time'],second['simulation_time']],float)
        if (not np.isfinite(times).all() or first['frame_id']==second['frame_id'] or
            before_release['frame_id'] in (first['frame_id'],second['frame_id']) or
            times[1]<=times[0] or times[2]-times[1]<.3):
            return unknown('Нет двух свежих кадров после отпускания')
        lo,hi=np.asarray(destination['bounds_min']),np.asarray(destination['bounds_max'])
        uncertainty=float(destination['uncertainty_m'])
        if not np.isfinite(uncertainty) or uncertainty<0. or uncertainty>.02:
            return unknown('Граница корзины измерена недостаточно точно')
        margin=max(.02,uncertainty+.012)
        release=np.asarray(release_xyz,float);jaw=np.asarray(jaw_xyz,float)
        release_jaw=np.asarray(release_jaw_xyz,float)
        if not np.isfinite(np.r_[lo,hi,release,jaw,release_jaw]).all():
            return unknown('Некорректная геометрия проверки размещения')
        def matches(frame):
            result=[]
            for candidate in frame.get('object_candidates',[]):
                evidence=candidate.get('evidence',{})
                lower,upper=np.asarray(candidate['bounds_min']),np.asarray(candidate['bounds_max'])
                center=np.asarray(candidate['center'])
                if (evidence.get('self_occlusion_not_resolved') is not False or
                    evidence.get('visible_pixels',0)<100 or candidate.get('uncertainty_m',1.)>.02 or
                    not np.isfinite(np.r_[lower,upper,center]).all() or
                    not _color_matches(candidate,held_reference)):
                    continue
                if (np.all(lower[:2]>lo[:2]+margin) and np.all(upper[:2]<hi[:2]-margin)
                    and lower[2]>=-.003 and upper[2]<hi[2]-margin
                    and np.linalg.norm(center[:2]-release[:2])<.08
                    and np.linalg.norm(center-jaw)>.065):result.append(candidate)
            return result
        a,b=matches(first),matches(second)
        if len(a)!=1 or len(b)!=1:
            return unknown('Предмет внутри корзины не виден однозначно на двух кадрах')
        prior=[c for c in before_release.get('object_candidates',[]) if _color_matches(c,held_reference)
            and c.get('evidence',{}).get('self_occlusion_not_resolved') is False
            and c.get('evidence',{}).get('visible_pixels',0)>=100]
        held=[c for c in prior if np.linalg.norm(np.asarray(c['center'])-release_jaw)<.08]
        if len(held)!=1:
            return unknown('Нет наблюдаемого предмета у пальцев непосредственно перед отпусканием')
        centers=np.asarray([a[0]['center'],b[0]['center']])
        if float(held[0]['center'][2])-centers[1,2]<.025:
            return unknown('Не подтверждено отделение предмета от захвата вниз')
        if any(np.linalg.norm(np.asarray(c['center'])-centers[1])<.04 for c in prior):
            return unknown('На месте размещения уже был похожий предмет')
        previous_surface=np.asarray(before_release.get('surface_points',[]),float)
        lower,upper=np.asarray(b[0]['bounds_min']),np.asarray(b[0]['bounds_max'])
        if previous_surface.ndim!=2 or previous_surface.shape[1]!=3:
            return unknown('Место размещения до отпускания не наблюдалось')
        footprint=(np.all(previous_surface[:,:2]>lower[:2],axis=1)&
                   np.all(previous_surface[:,:2]<upper[:2],axis=1)&
                   (previous_surface[:,2]<=lower[2]-.003)&
                   (previous_surface[:,2]>=lower[2]-.025))
        if np.count_nonzero(footprint)<6:
            return unknown('Не подтверждено свободное место под предметом до отпускания')
        if np.linalg.norm(centers[1]-centers[0])>.008:
            return unknown('Предмет после отпускания ещё перемещается')
        support=np.asarray(second.get('surface_points',[]),float)
        if support.ndim!=2 or support.shape[1]!=3:
            return unknown('Опора под предметом не наблюдается')
        # Measured samples outside the material silhouette, near it and below.
        # Cloth samples themselves cannot be used to prove their own support.
        outside=np.any((support[:,:2]<lower[:2]-.003)|(support[:,:2]>upper[:2]+.003),axis=1)
        near=np.linalg.norm(support[:,:2]-centers[1,:2],axis=1)<.09
        below=(support[:,2]<=lower[2]+.003)&(support[:,2]>=lower[2]-.025)
        inside=np.all(support[:,:2]>lo[:2]+margin,axis=1)&np.all(support[:,:2]<hi[:2]-margin,axis=1)
        if np.count_nonzero(outside&near&below&inside)<6:
            return unknown('Не видна поверхность опоры рядом с отпущенным предметом')
        return dict(outcome='success',evidence=dict(confirmed=True,
            source='two_frame_registered_rgbd_after_release',frames=[first['frame_id'],second['frame_id']],
            stable_inside_observed_opening=True,support_surface_observed=True,
            semantic_identity='unverified_single_geometric_target'))
    except (KeyError,TypeError,ValueError,IndexError):
        return unknown('Неполные наблюдения для проверки размещения')
