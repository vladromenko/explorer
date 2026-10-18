"""Sensor-driven room cleanup over a shared navigation/perception/policy port.

The coordinator never receives simulator body positions or contacts. Navigation
and semantic perception are separate from the learned manipulation policy; a
local ACT report cannot qualify this new composition or its clock handoff.
"""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class CleanupLimits:
    search_stops: int = 24
    realignments: int = 4
    target_max_age_s: float = 2.
    task_timeout_s: float = 1200.
    source_standoff_m: float = .20
    basket_reserve_m: float = .134


class ViewUnavailable(ValueError):
    """A planning/sensing failure with no executed safety-guard violation."""


def point(value, dimensions=3):
    if not isinstance(value,(list,tuple)) or len(value)!=dimensions:
        raise ValueError('Missing measured point')
    result=tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result):raise ValueError('Nonfinite measured point')
    return result


def approach_candidates(target,pose,*,source=True,limits=CleanupLimits()):
    """Pose hypotheses around an OBSERVED object; the port must plan each path.

    They are not free-space assertions or predetermined object locations.
    """
    center=point(target['center']);pose=point(pose)
    lo,hi=point(target['bounds_min']),point(target['bounds_max'])
    radius=max(hi[0]-lo[0],hi[1]-lo[1])/2
    # Global 2-D navigation stops outside the conservative arm footprint.
    # A separate observed 3-D local alignment admits the learned workspace.
    stand=.55 if source else radius+.55
    if not .1<=stand<=.9:raise ValueError('Object outside the supported local policy envelope')
    facing=math.atan2(center[1]-pose[1],center[0]-pose[0])
    return [dict(x=center[0]-stand*math.cos(facing+offset),
                 y=center[1]-stand*math.sin(facing+offset),yaw=facing+offset)
            for offset in (0.,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3,math.pi/2,-math.pi/2,math.pi)]


class RoomCleanup:
    def __init__(self,port,record,*,cancelled=lambda:False,clock=time.monotonic,limits=CleanupLimits()):
        self.port,self.record,self.cancelled,self.clock,self.limits=port,record,cancelled,clock,limits
        self.sequence=-1;self.base_revision=0;self.targets={};self.seen={};self.events=[]
        self.held=False;self.scope=None;self.last_motion_at=-float('inf')
        self.hints={};self.hint_seen={}

    def check(self):
        if self.cancelled():raise InterruptedError('Room cleanup cancelled')
        if self.clock()>self.deadline:raise TimeoutError('Room cleanup budget exhausted')
        self.port.permit()
        if self.scope is not None and self.port.scope()!=self.scope:
            raise ValueError('Boot, map or calibration changed during cleanup')

    def event(self,stage,result=None):
        self.events.append(dict(stage=stage,at=self.clock(),result=result))
        self.record(dict(state='running',stage=stage,held_verified=self.held,events=self.events))

    def observe(self):
        for attempt in range(3):
            self.check();requested=self.clock();value=self.port.observe();self.check()
            stamp=value.get('captured_at');sequence=value.get('sequence')
            if (type(sequence) is not int or sequence<=self.sequence or value.get('base_revision')!=self.base_revision
                    or value.get('scope')!=self.scope):
                raise ValueError('Camera observation is reused or belongs to a previous base pose')
            if (type(stamp) in (int,float) and requested<=stamp<=self.clock()
                    and self.clock()-stamp<=self.limits.target_max_age_s and stamp>self.last_motion_at):break
            self.sequence=sequence
            # A model's cold start may finish after the exposure expired.
            # Acquire new pixels; never relabel the old exposure as fresh.
            self.event('expired_exposure',dict(attempt=attempt+1,captured_at=stamp,requested_at=requested,finished_at=self.clock()))
        else:raise ViewUnavailable('Three fresh camera acquisitions expired before semantic inference completed')
        self.sequence=sequence
        hints=[h for h in value.get('hints',[]) if h.get('kind')=='basket']
        if len(hints)==1:
            hint=hints[0]
            if (hint.get('usable_for')=='camera_aim_only' and hint.get('geometry_source')=='registered_depth'
                    and hint.get('semantic_source')=='rgb_model' and hint.get('confidence',0)>=.35):
                previous=self.hint_seen.get('basket');center=point(hint['center'])
                if previous and previous['frame_id']!=hint['frame_id'] and math.dist(point(previous['center']),center)<.08:
                    self.hints['basket']=dict(hint,captured_at=stamp)
                self.hint_seen['basket']=hint
        # A second same-class object is ambiguous. Do not overwrite one target
        # with whichever candidate happened to be enumerated last.
        kinds=[item.get('kind') for item in value.get('objects',[])]
        for kind in ('sock','basket'):
            if kinds.count(kind)>1:
                self.seen.pop(kind,None);self.targets.pop(kind,None)
                raise ViewUnavailable('Ambiguous visible '+kind+' candidates')
        for item in value.get('objects',[]):
            kind=item.get('kind')
            if kind not in ('sock','basket'):continue
            if (item.get('geometry_source')!='registered_depth' or item.get('semantic_source')!='rgb_model'
                    or item.get('confidence',0)<.35 or not item.get('frame_id')):continue
            center=point(item['center']);point(item['bounds_min']);point(item['bounds_max'])
            previous=self.seen.get(kind)
            # A second independent exposure must agree in world space. A stale
            # remembered target is a search hint, never manipulation evidence.
            consistent=previous and previous['frame_id']!=item['frame_id'] and math.dist(previous['center'],center)<.06
            observed=dict(item,captured_at=stamp,base_revision=self.base_revision)
            if consistent:self.targets[kind]=observed
            elif kind in self.targets:self.targets.pop(kind)
            self.seen[kind]=observed
        return value

    def navigate(self,goal,stage,*,held=False):
        self.check()
        if held:self.verify('hold')
        self.event(stage+'_moving',dict(goal=goal,held=held))
        try:result=self.port.navigate(goal,check=self.check,held=held)
        except ViewUnavailable:
            # The stopped port may have driven part of a route. No exposure
            # from before that failed attempt remains manipulation evidence.
            self.base_revision+=1;self.last_motion_at=self.clock()
            self.port.set_base_revision(self.base_revision);self.seen.clear()
            raise
        if result.get('completed') is not True:raise ValueError('Navigation accepted without a completed arrival')
        self.check();self.base_revision+=1;self.last_motion_at=self.clock()
        self.port.set_base_revision(self.base_revision)
        self.seen.clear();self.event(stage,result)

    def view(self,operation,*args,**kwargs):
        before=point(self.port.pose())
        try:return getattr(self.port,operation)(*args,check=self.check,**kwargs)
        finally:
            after=point(self.port.pose())
            if math.dist(before[:2],after[:2])>.003 or abs(math.atan2(math.sin(after[2]-before[2]),math.cos(after[2]-before[2])))>.01:
                self.base_revision+=1;self.last_motion_at=self.clock();self.port.set_base_revision(self.base_revision);self.seen.clear()
                self.event('observation_reposition',dict(operation=operation))

    def search(self):
        visited=set();inspected=set()
        for index in range(self.limits.search_stops):
            self.check();self.event('search_view',dict(stop=index+1))
            for layer in getattr(self.port,'search_layers',(None,)):
                try:
                    if layer is None:self.view('search_view')
                    else:self.view('search_view',layer=layer)
                    self.observe();self.observe()
                except ViewUnavailable as error:self.event('search_view_unavailable',dict(reason=str(error)))
                hint=self.hints.get('basket')
                if hint and 'basket' not in self.targets and hasattr(self.port,'inspect_search_hint'):
                    key=(*[round(v/.2) for v in hint['center']],*[round(v/.5) for v in self.port.pose()[:2]])
                    if key not in inspected:
                        inspected.add(key)
                        try:
                            self.view('inspect_search_hint',hint);self.observe();self.observe()
                        except ViewUnavailable as error:self.event('search_hint_unconfirmed',dict(reason=str(error)))
                if all(k in self.targets for k in ('sock','basket')):break
            self.event('search',dict(stop=index+1,found=sorted(self.targets)))
            if all(k in self.targets for k in ('sock','basket')):return
            if hasattr(self.port,'prepare_navigation'):self.view('prepare_navigation')
            choices=[]
            for pose in self.port.survey_candidates():
                key=(round(pose['x']/.15),round(pose['y']/.15),round(pose['yaw']/.5))
                if key in visited:continue
                proposal=self.port.plan(pose,held=False)
                if proposal.get('reachable') is True:
                    # Coverage is a proposal over already scanned floor cells,
                    # not a claim that an unseen target is at this waypoint.
                    gain=max(0.,float(pose.get('information_gain',0.)))
                    yaw=self.port.pose()[2];turn=abs(math.atan2(math.sin(pose['yaw']-yaw),math.cos(pose['yaw']-yaw)))
                    choices.append(((proposal.get('cost',float('inf'))+.12+.08*turn)/(1.+gain*.08),key,pose))
            if not choices:raise ValueError('No unvisited reachable observation point; targets not found')
            _,key,goal=min(choices,key=lambda row:row[0]);visited.add(key)
            try:self.navigate(goal,'search_move')
            except ViewUnavailable as error:self.event('search_route_unavailable',dict(reason=str(error),goal=goal))
        raise ValueError('Search budget exhausted; sock and basket not both observed')

    def align(self,kind,*,held=False):
        target=self.targets[kind];tried=set()
        for attempt in range(self.limits.realignments):
            if hasattr(self.port,'prepare_navigation'):self.view('prepare_navigation',held=held)
            plans=[]
            for goal in approach_candidates(target,self.port.pose(),source=kind=='sock',limits=self.limits):
                key=(round(goal['x']/.025),round(goal['y']/.025),round(goal['yaw']/.1))
                if key in tried:continue
                plan=self.port.plan(goal,held=held)
                if plan.get('reachable') is True:plans.append((plan.get('cost',float('inf')),goal))
            if not plans:raise ValueError('No collision-checked approach to observed '+kind)
            goal=min(plans,key=lambda r:r[0])[1];tried.add((round(goal['x']/.025),round(goal['y']/.025),round(goal['yaw']/.1)))
            try:self.navigate(goal,'approach_'+kind,held=held)
            except ViewUnavailable as error:
                self.event('approach_route_unavailable',dict(kind=kind,reason=str(error)));continue
            self.targets.pop(kind,None)
            try:self.view('look_at',target,held=held)
            except ViewUnavailable as error:
                self.event('approach_view_unavailable',dict(kind=kind,reason=str(error)));continue
            try:self.observe();self.observe()
            except ViewUnavailable as error:
                self.event('approach_reobservation_unavailable',dict(kind=kind,reason=str(error)));continue
            fresh=self.targets.get(kind)
            if fresh is None:
                self.event('approach_target_missing',dict(kind=kind));continue
            if math.dist(point(target['center']),point(fresh['center']))>.06:
                self.event('target_moved',dict(kind=kind,previous=target['center'],current=fresh['center']))
                target=fresh;continue
            admissible=self.port.local_admission(kind,fresh)
            if admissible.get('inside_envelope') is True:return fresh
            try:local=self.port.local_align(kind,fresh,check=self.check,held=held)
            except ViewUnavailable as error:
                self.event('local_approach_unavailable',dict(kind=kind,reason=str(error)));target=fresh;continue
            if local.get('completed') is not True:raise ValueError('Observed local alignment failed')
            self.base_revision+=1;self.last_motion_at=self.clock();self.port.set_base_revision(self.base_revision)
            self.targets.pop(kind,None);self.seen.clear()
            try:self.view('look_at',fresh,held=held);self.observe();self.observe()
            except ViewUnavailable as error:
                self.event('local_reobservation_unavailable',dict(kind=kind,reason=str(error)));target=fresh;continue
            confirmed=self.targets.get(kind)
            if confirmed and self.port.local_admission(kind,confirmed).get('inside_envelope') is True:return confirmed
            target=fresh
        raise ValueError('Fresh geometric alignment did not converge: '+kind)

    def verify(self,kind):
        self.check();result=self.port.verify(kind,check=self.check)
        evidence=result.get('evidence',{})
        if (result.get('outcome')!='success' or evidence.get('source')!='rgbd_temporal'
                or evidence.get('confirmed') is not True or evidence.get('scope')!=self.scope
                or len(set(evidence.get('frame_ids',[])))<2 or evidence.get('base_revision')!=self.base_revision
                or type(evidence.get('captured_at')) not in (int,float)
                or not self.last_motion_at<evidence['captured_at']<=self.clock()
                or self.clock()-evidence['captured_at']>self.limits.target_max_age_s):
            raise ValueError('Observed '+kind+' is unconfirmed; motion is stopped')
        self.event('verify_'+kind,result);return result

    def manipulate(self,phase,target):
        self.check()
        if target['base_revision']!=self.base_revision or self.clock()-target['captured_at']>self.limits.target_max_age_s:
            raise ValueError('Reobserve the target before learned manipulation')
        self.event('policy_'+phase+'_running');result=self.port.learned_manipulation(phase,target,check=self.check)
        if (result.get('policy_sha256')!=self.policy_sha256 or result.get('teacher_fallback') is not False
                or result.get('completed') is not True):raise ValueError('Learned manipulation did not complete with the frozen policy')
        # Any local base motion also invalidates old object observations.
        if result.get('base_moved'):
            self.base_revision+=1;self.last_motion_at=self.clock();self.port.set_base_revision(self.base_revision)
        self.event('policy_'+phase,result)

    def run(self):
        self.deadline=self.clock()+self.limits.task_timeout_s
        result=dict(state='failed',placed_verified=False,physical_ready=False)
        try:
            admission=self.port.preflight()
            if admission.get('ready') is not True:raise ValueError('; '.join(admission.get('blocked_by',[])))
            if not admission.get('mission_contract_sha256'):raise ValueError('New mission composition is not bound to a contract')
            self.policy_sha256=admission['policy_sha256'];self.scope=self.port.scope()
            self.port.begin(check=self.check);self.search()
            source=self.align('sock');self.view('prepare_policy','pick')
            # Camera/arm preparation may change the view even without a drive.
            self.targets.pop('sock',None);self.observe();self.observe();source=self.targets.get('sock')
            if source is None:raise ValueError('Sock lost during policy preparation')
            self.manipulate('pick',source);self.verify('hold');self.held=True
            basket=self.align('basket',held=True)
            self.view('prepare_policy','place',held=True)
            self.targets.pop('basket',None);self.observe();self.observe();basket=self.targets.get('basket')
            if basket is None:raise ValueError('Basket lost during policy preparation')
            self.manipulate('place',basket);self.verify('place');self.held=False
            result.update(state='succeeded',placed_verified=True,policy_sha256=self.policy_sha256)
        except Exception as error:result.update(reason=str(error),cancelled=isinstance(error,InterruptedError))
        finally:
            try:self.port.stop()
            except Exception as error:result.update(state='failed',placed_verified=False,stop_error=str(error))
        result.update(events=self.events,held_verified=self.held)
        self.record(result);return result
