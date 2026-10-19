"""Protocol fixtures are not simulated or hardware acceptance evidence."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.room_cleanup import RoomCleanup,CleanupLimits,approach_candidates
from src.room_cleanup_rpc import RobotRPC,save
from simulator.room_policy import issued_command,place_clock
from simulator.room_transfer import qualification
from simulator.room_validation import room_scene


class Clock:
    def __init__(self):self.now=1.
    def __call__(self):self.now+=.001;return self.now


class Port:
    def __init__(self):
        self.clock=Clock();self.rev=0;self.seq=0;self.stop_count=0;self.moves=[];self.phases=[]
        self.identity=('boot','map','calib');self.proof=True;self.complete=True
        self.positions={'sock':[1.,-.6,.02],'basket':[-1.,.8,.30]};self.observations=[]
    def scope(self):return self.identity
    def permit(self):pass
    def pose(self):return (0.,0.,0.)
    def preflight(self):return dict(ready=True,policy_sha256='weight',mission_contract_sha256='new-mission')
    def begin(self,check):check()
    def search_view(self,check):check()
    def plan(self,goal,**kw):return dict(reachable=True,cost=1.)
    def survey_candidates(self):return []
    def set_base_revision(self,v):self.rev=v
    def navigate(self,goal,check,held):check();self.moves.append((goal,held));return dict(completed=self.complete)
    def observe(self):
        self.seq+=1;self.observations.append(self.rev);objects=[]
        for kind,xyz in self.positions.items():
            p=np.array(xyz);size=np.array([.125,.125,.15] if kind=='basket' else [.07,.025,.01])
            objects.append(dict(kind=kind,center=xyz,bounds_min=(p-size).tolist(),bounds_max=(p+size).tolist(),
                confidence=.8,geometry_source='registered_depth',semantic_source='rgb_model',frame_id=str(self.seq)))
        return dict(sequence=self.seq,captured_at=self.clock(),scope=self.scope(),base_revision=self.rev,objects=objects)
    def look_at(self,*args,check,**kw):check()
    def local_admission(self,*args):return dict(inside_envelope=True)
    def prepare_policy(self,*args,check,**kw):check()
    def learned_manipulation(self,phase,target,check):
        check();self.phases.append(phase);return dict(completed=True,teacher_fallback=False,policy_sha256='weight')
    def verify(self,kind,check):
        check();return dict(outcome='success' if self.proof else 'unknown',evidence=dict(source='rgbd_temporal',
            confirmed=self.proof,scope=self.scope(),base_revision=self.rev,frame_ids=[str(self.seq),str(self.seq+1)],captured_at=self.clock()))
    def stop(self):self.stop_count+=1


def run(port):return RoomCleanup(port,lambda _:None,clock=port.clock).run()


def test_policy_handoff_reacquires_two_current_frames_after_a_missing_target():
    p=Port();original=p.observe;calls=0
    def observe():
        nonlocal calls
        calls+=1;value=original()
        if calls==2:
            value['objects']=[item for item in value['objects'] if item['kind']!='sock']
            p.clock.now+=3.  # the first confirmation expires during this call
            value['captured_at']=p.clock()
        return value
    p.observe=observe;coordinator=RoomCleanup(p,lambda _:None,clock=p.clock)
    coordinator.scope=p.scope();coordinator.deadline=p.clock()+100
    target=coordinator.manipulation_observation('sock')
    assert calls==4 and target['frame_id']=='4'
    assert p.clock()-target['captured_at']<coordinator.limits.target_max_age_s
    assert coordinator.seen['sock']['frame_id']=='4' and not p.phases


def test_missing_handoff_target_cannot_reuse_a_single_old_confirmation():
    from src.room_cleanup import ViewUnavailable
    p=Port();original=p.observe;calls=0
    def observe():
        nonlocal calls
        calls+=1;value=original()
        if calls>1:value['objects']=[item for item in value['objects'] if item['kind']!='sock']
        return value
    p.observe=observe;coordinator=RoomCleanup(p,lambda _:None,clock=p.clock)
    coordinator.scope=p.scope();coordinator.deadline=p.clock()+100
    with pytest.raises(ViewUnavailable,match='consecutive'):coordinator.manipulation_observation('sock')
    assert calls==4 and 'sock' not in coordinator.targets and 'sock' not in coordinator.seen
    assert not p.phases


@pytest.mark.parametrize('positions',[
    {'sock':[1.,-.6,.02],'basket':[-1.,.8,.30]},
    {'sock':[-1.2,.7,.02],'basket':[1.,-.9,.12]}])
def test_whole_sensor_driven_sequence_reobserves_after_each_drive(positions):
    p=Port();p.positions=positions;r=run(p)
    assert r['state']=='succeeded' and r['placed_verified'] and p.phases==['pick','place']
    assert len(p.moves)==2 and p.moves[1][1] is True
    assert p.observations.count(1)>=2 and p.observations.count(2)>=2 and p.stop_count==1
    assert not r['physical_ready']


@pytest.mark.parametrize('failure',['accepted_only','held_unknown','stale','reused','epoch','teacher','ambiguous','stale_proof'])
def test_unconfirmed_or_reused_evidence_cannot_finish(failure):
    p=Port()
    if failure=='accepted_only':p.complete=False
    if failure=='held_unknown':p.proof=False
    if failure in ('stale','reused','epoch','ambiguous'):
        original=p.observe
        def observe():
            v=original()
            if failure=='stale':v['captured_at']=0.
            if failure=='reused':v['sequence']=1
            if failure=='epoch':p.identity=('boot','new-map','calib')
            if failure=='ambiguous':v['objects'].append(copy.deepcopy(v['objects'][0]))
            return v
        p.observe=observe
    if failure=='teacher':p.learned_manipulation=lambda *a,**k:dict(completed=True,teacher_fallback=True,policy_sha256='weight')
    if failure=='stale_proof':
        original=p.verify
        def verify(*a,**k):v=original(*a,**k);v['evidence']['captured_at']=0.;return v
        p.verify=verify
    r=run(p);assert r['state']=='failed' and not r['placed_verified'] and p.stop_count==1


def test_moved_target_is_relocalized_before_policy():
    p=Port();original=p.navigate
    def navigate(*a,**k):
        v=original(*a,**k)
        if len(p.moves)==1:p.positions['sock']=[1.4,-.6,.02]
        return v
    p.navigate=navigate;r=run(p)
    assert r['state']=='succeeded' and len(p.moves)==3
    assert any(e['stage']=='target_moved' for e in r['events'])


def test_error_always_stops_and_never_places_after_lost_hold():
    p=Port();p.verify=lambda *a,**k:(_ for _ in ()).throw(ValueError('lost'))
    r=run(p);assert r['state']=='failed' and p.phases==['pick'] and p.stop_count==1


def test_infeasible_basket_view_selects_another_checked_approach():
    from src.room_cleanup import ViewUnavailable
    p=Port();original=p.look_at;failed=[]
    def look_at(target,**kwargs):
        if target['kind']=='basket' and not failed:
            failed.append(True);raise ViewUnavailable('Rim cannot be viewed from this approach')
        return original(target,**kwargs)
    p.look_at=look_at;r=run(p)
    assert r['state']=='succeeded' and len(p.moves)==3
    assert p.moves[1][0]!=p.moves[2][0] and p.moves[2][1] is True


def test_live_semantic_requests_stay_inside_service_storage_and_preserve_audit(tmp_path,monkeypatch):
    from simulator.semantic_vision import SemanticDetector
    import simulator.semantic_cache as cache
    root=tmp_path/'models';audit=tmp_path/'mission/recognition'
    def infer(storage,folder,**kwargs):
        assert folder.is_relative_to(root) and (folder/'image.png').exists()
        request=json.loads((folder/'request.json').read_text())
        assert request['frame_id']=='frame' and request['query']=='sock'
        (folder/'result.json').write_text(json.dumps(dict(frame_id='frame',detections=[])))
    monkeypatch.setattr(cache,'infer',infer)
    assert SemanticDetector(root,audit).detect(np.zeros((10,10,3),np.uint8),'sock','frame')==[]
    assert len(list(audit.glob('*/result.json')))==1


def test_metric_crop_grounding_uses_pixels_and_maps_model_box_to_original_exposure():
    from simulator.semantic_vision import detect_metric_crops,associate,projected_bounds
    frame=dict(frame_id='original',optical_to_world=np.eye(4).tolist(),intrinsics=dict(fx=100.,fy=100.,cx=50.,cy=50.))
    candidate=dict(center=[0,0,1],bounds_min=[-.1,-.1,1],bounds_max=[.1,.1,1.01],confidence=.75,uncertainty_m=.006)
    calls=[]
    class Detector:
        def detect(self,rgb,query,frame_id):
            calls.append((rgb.shape,query,frame_id))
            return [dict(bbox=[8,8,rgb.shape[1]-8,rgb.shape[0]-8],confidence=.48,label='sock')]
    image=np.zeros((100,100,3),np.uint8)
    rows=detect_metric_crops(Detector(),image,'sock',frame,[candidate])
    assert calls[0][1]=='sock' and calls[0][2].startswith('original:metric-crop:') and calls[0][0][0]<100
    assert rows[0]['source_frame_id']=='original' and rows[0]['label']=='sock'
    assert associate(rows,[candidate],frame)['semantic_evidence']['detection']['confidence']==.48
    candidate['confidence']=.3
    assert detect_metric_crops(Detector(),image,'sock',frame,[candidate])==[]


def test_slow_semantics_need_a_new_matching_exposure_not_a_relabelled_timestamp():
    from simulator.semantic_vision import refresh_stationary_detections
    def frame(identifier):return dict(frame=dict(frame_id=identifier,intrinsics=dict(fx=100.),optical_to_world=np.eye(4).tolist()),
        rgb=np.full((30,30,3),100,np.uint8),depth=np.full((30,30),.5))
    reference,current=frame('model-input'),frame('fresh-camera')
    detections=[dict(bbox=[3,3,27,27],label='sock',confidence=.48)]
    result=refresh_stationary_detections(reference,current,detections)
    assert result[0]['model_frame_id']=='model-input' and result[0]['tracked_to_frame_id']=='fresh-camera'
    assert not refresh_stationary_detections(reference,reference,detections)
    current['depth'][3:27,3:27]=.6
    assert not refresh_stationary_detections(reference,current,detections)
    current=frame('fresh');current['rgb'][3:27,3:27]=0
    assert not refresh_stationary_detections(reference,current,detections)
    current=frame('fresh');current['frame']['optical_to_world'][0][3]=.01
    assert not refresh_stationary_detections(reference,current,detections)


def test_room_command_has_identical_bounded_sequential_actuation():
    reference=[90,90,90,90,90,30,0,0,0]
    action=issued_command(reference,[94,90,90,90,90,30,.1,.1,.3])
    assert action.tolist()==[92,90,90,90,90,30,0,0,0]
    with pytest.raises(ValueError):issued_command(reference,[float('nan')]*9)


def test_small_learned_deltas_are_not_lost_to_integer_degree_host_commands():
    reference=np.array([90.]*5+[30.]+[0.]*3);residual=np.zeros(6)
    for _ in range(4):
        predicted=reference.copy();predicted[0]+=.25
        reference=issued_command(reference,predicted,residual)
        assert np.max(abs(residual))<=.5 and np.all(reference[6:]==0)
    assert reference[0]==91.


@pytest.mark.parametrize('queued,fresh_delta,accepted',[(True,3.,True),(True,21.,False),(False,3.,False)])
def test_cached_outlier_is_recomputed_once_but_fresh_guard_is_not_relaxed(queued,fresh_delta,accepted):
    from simulator.room_policy import RoomActor
    reference=np.array([90.]*5+[30.]+[0.]*3);bad=reference.copy();bad[2]+=25.
    fresh=reference.copy();fresh[2]+=fresh_delta
    calls=[]
    class Actor:
        last_action_elapsed_s=12.25;last_action_was_queued=queued
        def __call__(self,*_):return bad
        def replan(self,*_):calls.append('fresh');return fresh
    actor=RoomActor.__new__(RoomActor);actor.actor=Actor();actor.replans=0
    residual=np.zeros(6)
    if accepted:
        command=actor.command(reference,np.zeros((240,320,3),np.uint8),residual)
        assert command[2]==92. and np.max(abs(command[:6]-reference[:6]))<=2.
    else:
        with pytest.raises(ValueError,match='Недопустимый шаг сустава 3'):
            actor.command(reference,np.zeros((240,320,3),np.uint8),residual)
        assert np.all(residual==0.)
    assert calls==(['fresh'] if queued else [])
    assert actor.command_evidence['prediction'][2]==115.


def test_cached_replan_preserves_the_original_policy_clock(monkeypatch):
    from simulator.learning import LoadedPolicy
    actor=LoadedPolicy.__new__(LoadedPolicy);actor.last_action_was_queued=True
    actor.last_action_elapsed_s=12.25;actor.elapsed_s=12.30
    cleared=[];actor.policy=SimpleNamespace(reset=lambda:cleared.append(True));observed=[]
    def infer(self,*_):observed.append(self.elapsed_s);self.elapsed_s+=.05;return [0.]*9
    monkeypatch.setattr(LoadedPolicy,'__call__',infer)
    actor.replan(None,None)
    assert observed==[12.25] and actor.elapsed_s==pytest.approx(12.30) and cleared==[True]


def test_placement_clock_comes_from_matching_training_geometry_not_saved_waypoints():
    profile=dict(placement_proposals=[dict(height_m=.3,half_size_m=[.125,.125],elapsed_s=42.,observed_command=[90]*9)])
    target=dict(bounds_min=[1.,1.,0.],bounds_max=[1.25,1.25,.3])
    assert place_clock(profile,target,[90]*9)==42.
    target['bounds_max'][2]=.6
    with pytest.raises(ValueError):place_clock(profile,target,[90]*9)


def test_rpc_rejects_cross_session_and_nonfinite_actions_without_actuation(tmp_path):
    called=[];transport=SimpleNamespace(permit=lambda:None,stop=lambda:called.append('stop'),apply=lambda *a,**k:pytest.fail('unsafe apply'))
    rpc=RobotRPC(transport,tmp_path,'mine')
    save(tmp_path/'000001.request.json',dict(schema=1,session='other',operation='apply',arguments={}))
    assert rpc.serve_next();assert 'mismatch' in json.loads((tmp_path/'000001.response.json').read_text())['error']
    # Raw malformed JSON must also stop and produce an owned error response.
    (tmp_path/'000002.request.json').write_text('{')
    assert rpc.serve_next() and called==['stop','stop']


def room_groups():
    groups=[]
    stages=['search','approach_sock','policy_pick','verify_hold','approach_basket','policy_place','verify_place']
    for base in (1600000010,1600000020):
        rows=[dict(seed=base+i,state='succeeded',placed_verified=True,success=True,physical_outcome=dict(success=True),
            room_layout=True,collision_sampling='each_physics_substep',
            immutable_source_verified=True,source_revision='a'*40,
            collision_free=True,actor_truth_inputs=False,teacher_fallback=False,physical_ready=False,simulation_only=True,
            mission_contract_sha256='contract',model_sha256='weight',domain_variation=room_scene(base+i),
            events=[dict(stage=s) for s in stages]) for i in range(10)]
        groups.append(dict(episodes=10,successes=10,results=rows))
    return groups


def test_room_qualification_requires_both_whole_mission_groups():
    groups=room_groups();profile=dict(model_checkpoint_sha256={'model.safetensors':'weight'})
    bundle=dict(checkpoint_sha256=profile['model_checkpoint_sha256'])
    assert len(qualification(groups,profile,bundle)[0])==20
    for field,value in [('collision_free',False),('actor_truth_inputs',True),('teacher_fallback',True),('placed_verified',False),('immutable_source_verified',False)]:
        bad=copy.deepcopy(groups);bad[1]['results'][0][field]=value
        with pytest.raises(ValueError):qualification(bad,profile,bundle)
    groups[1]['results'][0]['seed']=groups[0]['results'][0]['seed']
    with pytest.raises(ValueError,match='overlap'):qualification(groups,profile,bundle)


def test_room_recipe_really_separates_search_and_destination_and_keeps_user_bin():
    for seed in range(1600000000,1600000030):
        scene=room_scene(seed);a,b=np.array(scene['sock_position']),np.array(scene['bin_position'])
        assert np.linalg.norm(a[:2]-b[:2])>=1. and np.linalg.norm(a[:2])>.7
        assert abs(a[0]-.5)>.35 and abs(b[0]-.5)>.35
        if seed%10==4:assert scene['parameters']['bin']==dict(height_m=.30,shape='round',inner_half_size_m=[.125,.125])
