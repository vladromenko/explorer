import copy

import pytest

from simulator.placement_evidence import verify_released_target


def scene():
    target=dict(center=[.3,0.,.02],bounds_min=[.28,-.02,.009],bounds_max=[.32,.02,.03],
        uncertainty_m=.012,appearance={'rgb_median':[.6,.3,.6]},
        evidence={'self_occlusion_not_resolved':False,'visible_pixels':400})
    first=dict(frame_id='one',simulation_time=1.,object_candidates=[copy.deepcopy(target)])
    second=dict(frame_id='two',simulation_time=1.5,object_candidates=[copy.deepcopy(target)],
        surface_points=[[.3+x,y,.002] for x in [-.035,.035] for y in [-.03,0.,.03]])
    args=dict(destination=dict(bounds_min=[.175,-.125,0.],bounds_max=[.425,.125,.3],uncertainty_m=.012),
        held_reference=target,release_xyz=[.3,0.,.335],jaw_xyz=[.3,0.,.4],
        release_jaw_xyz=[.3,0.,.335],before_release=dict(frame_id='before',simulation_time=.5,
            object_candidates=[dict(copy.deepcopy(target),center=[.3,0.,.31])],
            surface_points=[[.3+x,y,.002] for x in [-.01,.01] for y in [-.01,0.,.01]]))
    return first,second,args


def test_observed_stable_material_inside_bin_with_visible_support():
    first,second,args=scene()
    result=verify_released_target(first,second,**args)
    assert result['outcome']=='success' and result['evidence']['confirmed'] is True


@pytest.mark.parametrize('problem',['same_frame','short_interval','no_support','self_as_support',
    'still_held','outside','rim','different_color','moving','ambiguous','occluded','bad_numbers',
    'prior_object','no_release_frame','no_drop','bad_time','unseen_release_place'])
def test_unconfirmed_placement_never_counts_as_success(problem):
    first,second,args=scene();target=second['object_candidates'][0]
    if problem=='same_frame':second['frame_id']='one'
    if problem=='short_interval':second['simulation_time']=1.1
    if problem=='no_support':second['surface_points']=[]
    if problem=='self_as_support':second['surface_points']=[[.3,0.,.01]]*20
    if problem=='still_held':args['jaw_xyz']=[.3,0.,.03]
    if problem=='outside':target['bounds_max'][0]=.44
    if problem=='rim':target['bounds_max'][2]=.3
    if problem=='different_color':target['appearance']['rgb_median']=[.01,.01,.9]
    if problem=='moving':target['center'][0]+=.02
    if problem=='ambiguous':second['object_candidates'].append(copy.deepcopy(target))
    if problem=='occluded':target['evidence']['self_occlusion_not_resolved']=True
    if problem=='bad_numbers':target['center'][0]=float('nan')
    if problem=='bad_time':first['simulation_time']=float('nan')
    if problem=='prior_object':args['before_release']['object_candidates'].append(copy.deepcopy(target))
    if problem=='no_release_frame':args['before_release']['object_candidates']=[]
    if problem=='no_drop':args['before_release']['object_candidates'][0]['center'][2]=.03
    if problem=='unseen_release_place':args['before_release']['surface_points']=[]
    assert verify_released_target(first,second,**args)['outcome']=='unknown'
