import copy

import numpy as np
import pytest

from simulator.multiview_container import combine_registered_container
from simulator.rgbd_observation import extract_candidates,ObservationRefused


def test_cropped_lower_wall_does_not_move_the_observed_opening():
    from simulator.multiview_container import opening_displacement
    first=dict(bounds_min=[.32,.20,.08],bounds_max=[.57,.45,.30])
    cropped=dict(bounds_min=[.32,.20,.22],bounds_max=[.57,.45,.30])
    assert opening_displacement(first,cropped)==pytest.approx(0.)
    for shift in ([.07,0.,0.],[0.,0.,.07]):
        moved={k:(np.asarray(v)+shift).tolist() for k,v in first.items()}
        assert opening_displacement(first,moved)>.06


def capture(identifier,bx,shift,stamp,*,hole=True):
    k=dict(fx=100.,fy=100.,cx=63.5,cy=63.5)
    pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[bx,0.,1.]
    depth=np.ones((128,128))
    depth[40:88,40+shift:44+shift]=.7;depth[40:88,84+shift:88+shift]=.7
    depth[40:44,40+shift:88+shift]=.7;depth[84:88,40+shift:88+shift]=.7
    depth[42:44,40+shift:88+shift]=.8
    if not hole:depth[48:80,48+shift:80+shift]=np.nan
    robot=np.full(depth.shape,8.)
    features=extract_candidates(depth,k,pose,robot_depth=robot)
    frame=dict(**features,frame_id=identifier,simulation_time=stamp,
        intrinsics=k,optical_to_world=pose.tolist(),robot_self_filter={'available':True})
    return dict(frame=frame,depth=depth,robot_depth=robot)


def records():
    a=capture('rim-a',0.,0,1.,hole=False)
    b=capture('rim-b',.028,-4,2.,hole=False)
    c=capture('opening-c',0.,0,3.)
    d=capture('opening-d',.028,-4,4.)
    # Opening close-ups need not supply a complete rim. They must register
    # their visible rim against the independently measured complete boundary.
    c['frame']['observed_rims']=[];d['frame']['observed_rims']=[]
    return [a,b,c,d]


def test_partial_views_need_full_boundary_two_openings_and_registration():
    result=combine_registered_container(records(),now_simulation_time=4.)
    assert result['uncertainty_m']<=.02
    assert result['semantic_identity']=='unverified'
    e=result['evidence']
    assert e['complete_rim_frames']==['rim-a','rim-b'] or e['complete_rim_frames']==['rim-b','rim-a']
    assert len(e['opening_views'])==2 and all(v['opening_pixels']>=8 for v in e['opening_views'])
    assert e['motion_clearance_not_established']
    assert all(v['measured_minimum_interior_margin_m']>=v['required_interior_margin_m']
               for v in e['opening_views'])


@pytest.mark.parametrize('kind',['only_rims','one_opening','same_origin','stale','missing_self_mask','moved_rim'])
def test_missing_or_contradictory_evidence_never_invents_an_opening(kind):
    views=records();now=4.
    if kind=='only_rims':views=views[:2]
    elif kind=='one_opening':views=views[:3]
    elif kind=='same_origin':views[3]=copy.deepcopy(views[2]);views[3]['frame']['frame_id']='duplicate-view'
    elif kind=='stale':now=6.
    elif kind=='missing_self_mask':views[3]['robot_depth'][:]=np.nan
    elif kind=='moved_rim':views[1]['frame']['observed_rims'][0]['bounds_max'][0]+=.10
    with pytest.raises(ObservationRefused):combine_registered_container(views,now_simulation_time=now)


def test_a_previously_observed_solid_lid_cannot_be_reinterpreted_as_an_empty_rim():
    views=records()
    for record in views[:2]:record['depth'][50:78,50:78]=.7
    with pytest.raises(ObservationRefused):combine_registered_container(views,now_simulation_time=4.)


def test_visible_robot_surfaces_cannot_supply_interior_depth_proof():
    views=records()
    for record in views[2:]:record['robot_depth']=record['depth'].copy()
    with pytest.raises(ObservationRefused):combine_registered_container(views,now_simulation_time=4.)


def test_slow_pair_search_cannot_refresh_timestamp_without_new_opening_pixels():
    from simulator.multiview_container import refresh_registered_container
    views=records();candidate=combine_registered_container(views,now_simulation_time=4.)
    latest=capture('new-exposure',.028,-4,4.1);latest['frame']['observed_rims']=[]
    fresh=refresh_registered_container(views+[latest],candidate,latest)
    assert fresh['frame_id']=='new-exposure'
    assert fresh['evidence']['opening_views'][-1]['frame_id']=='new-exposure'
    assert fresh['evidence']['opening_camera_baseline_m']>=.02
    obstructed=copy.deepcopy(latest);obstructed['depth'][50:78,46:74]=.7
    with pytest.raises(ObservationRefused):refresh_registered_container(views+[obstructed],candidate,obstructed)
