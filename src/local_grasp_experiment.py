"""Bounded physical LEARN_GRASP_LOCAL loop using the existing arm executor."""
import json
from pathlib import Path
import threading
import time
import numpy as np
from delivery_vision import FeatureObject,reacquire_candidate
from grasp_verification import verify_lift,verify_place
from learning_stack import CandidateScorer
from semantic_world import guarded_closure


class LocalGraspExperiment:
    def __init__(self,root,workflows,readiness,finder,vision,arm,trajectory,model):
        self.root=Path(root);self.workflows=workflows;self.readiness=readiness;self.finder=finder
        self.vision=vision;self.arm=arm;self.trajectory=trajectory;self.model=model
        self.scorer=CandidateScorer(root);self.lock=threading.Lock();self.cancelled=threading.Event()
        self.state={'phase':'idle','physical_execution':False};self.active=None

    def status(self):return dict(self.state,busy=self.lock.locked(),workflow_id=self.active)

    def start(self,workflow_id):
        workflow=self.workflows.get(workflow_id)
        if workflow['skill']!='grasp' or workflow['state'] not in ('queued_autonomous','interrupted'):
            raise ValueError('Workflow is not ready for local grasp trials')
        row={item['id']:item for item in self.readiness.status()['capabilities']}['LEARN_GRASP_LOCAL']
        if not row['experimental_ready']:
            raise ValueError('LEARN_GRASP_LOCAL is not ready: '+', '.join(row['runtime_missing']+row['evidence_missing']+row['permission_missing']))
        if not self.lock.acquire(blocking=False):raise ValueError('Local grasp experiment is already running')
        self.cancelled.clear();self.active=workflow_id;self.state={'phase':'starting','physical_execution':False}
        threading.Thread(target=self.run,args=(workflow_id,),daemon=True).start()
        return self.status()

    def stop(self):
        self.cancelled.set();self.trajectory.stop();self.vision.stop();self.finder.cancel(self.finder.status().get('id'))
        self.state.update(phase='stopping',reason='Operator stop requested')
        return self.status()

    def permit(self):
        if self.cancelled.is_set():raise ValueError('Local grasp experiment cancelled')
        row={item['id']:item for item in self.readiness.status()['capabilities']}['LEARN_GRASP_LOCAL']
        if not row['experimental_ready']:
            raise ValueError('Local grasp permit changed: '+', '.join(row['runtime_missing']+row['permission_missing']))
        status=json.loads((self.root/'data/status.json').read_text())
        if status.get('stop_latched') is not True or any(abs(float(v))>.001 for v in status.get('velocity',[])):
            raise ValueError('Local grasp requires stationary STOP-latched base')

    def settings(self):
        profile=json.loads((self.root/'config/local-grasp-profile.json').read_text())
        handeye=json.loads((self.root/'config/handeye-accepted.json').read_text())
        if profile.get('experimental_execution_authorized') is not True or handeye.get('execution_authorized') is not True:
            raise ValueError('Experimental grasp profile or hand-eye is not authorized')
        return dict(open_deg=profile['open_command_deg'],close_deg=profile['hold_command_deg'],
            gripper_linkage_rad=profile['gripper_linkage_reference_rad'],
            grasp_quaternion_xyzw=profile['grasp_quaternion_xyzw'],approach_height_m=profile['approach_height_m'],
            lift_height_m=profile['lift_height_m'],grasp_tcp_offset_m=profile['grasp_tcp_offset_m'],
            object_kind=profile['object_kind'],camera_to_mount=handeye['camera_to_mount_reference'],
            handeye_execution_authorized=True,handeye_physical_validation_record=handeye['physical_validation_record'],
            floor_plane_base=[0.,0.,1.,0.])

    def detect(self,target,settings):
        self.state.update(phase='detecting',target=target);request=self.finder.start(target);deadline=time.monotonic()+105
        while self.finder.status().get('busy') and time.monotonic()<deadline:
            self.permit();time.sleep(.1)
        result=self.finder.status()
        if result.get('id')!=request['id'] or result.get('phase')!='ready':
            raise ValueError('tracking_lost: '+str(result.get('error','object search timeout')))
        objects=[item for item in result['result'].get('objects',[]) if item.get('position') and item.get('confidence',0)>=.35]
        if len(objects)!=1:raise ValueError('geometry_unknown: expected exactly one depth-localized target')
        with np.load(self.root/'data/object-searches'/request['id']/'rgbd.npz',allow_pickle=False) as raw:initial=dict(raw)
        fresh=self.vision.snapshot();candidate=reacquire_candidate(initial,fresh,objects[0]['bbox'])
        tracker=FeatureObject(fresh,candidate['bbox'])
        transform,_,_=self.vision.geometry(fresh,settings)
        plane=np.asarray(objects[0]['depth_evidence']['evidence']['plane_camera'],dtype=float)
        plane_base=np.linalg.inv(transform).T@plane;plane_base/=np.linalg.norm(plane_base[:3])
        settings['floor_plane_base']=plane_base.tolist();self.vision.start_tracker(tracker,settings)
        end=time.monotonic()+3
        while time.monotonic()<end:
            self.permit()
            try:return self.vision.latest(),settings
            except ValueError:time.sleep(.05)
        raise ValueError('tracking_lost: no fresh tracked target')

    def candidates(self,observation,settings,trial_number):
        point=np.asarray(observation['object_xyz'],dtype=float);point[2]+=settings['grasp_tcp_offset_m']
        uncertainty=float(observation.get('object_position_uncertainty_m',1))
        if uncertainty>.012:raise ValueError('geometry_unknown: object position uncertainty is too high')
        rows=[];seed=self.arm.reference()['servo_deg'][:5]
        for lateral in (0.,-.01,.01):
            candidate=point+[0,lateral,0];above=candidate+[0,0,settings['approach_height_m']]
            solved=self.model().ik(above,seed,settings['gripper_linkage_rad'],settings['grasp_quaternion_xyzw'])
            if solved.get('solved') and not solved.get('collision'):
                rows.append(dict(approach_x=float(candidate[0]),approach_y=float(candidate[1]),approach_z=float(candidate[2]),
                    aperture_m=0.,roll_rad=0.,base_shift_m=0.,clearance_m=settings['approach_height_m']-uncertainty,
                    trajectory_cost=float(solved.get('position_error_m',0))+abs(lateral)*10,point=candidate.tolist()))
        if not rows:raise ValueError('unreachable: no collision-free IK candidate')
        extent=observation.get('object_extent_xyz_m',[.05,.04,.025])
        context=dict(object_width_m=float(extent[0]),object_height_m=float(extent[2]),
            distance_m=float(np.linalg.norm(point)),visibility=float(observation.get('confidence',0)),
            depth_uncertainty_m=uncertainty,target_distance_m=0.,softness=1.,scene_clutter=0.,previous_failures=0.)
        candidate=self.scorer.latest_candidate();evaluation_arm='accepted'
        if candidate is not None:
            evaluation_arm='candidate' if trial_number%2 else 'baseline'
        selected_checkpoint=candidate if evaluation_arm=='candidate' else None
        ranked=self.scorer.choose(context,rows,exploration=.15,seed=int(time.time()),checkpoint=selected_checkpoint)
        if evaluation_arm=='baseline' and self.scorer.version!='geometry-baseline':
            baseline=CandidateScorer(self.root);baseline.weights[:]=0;baseline.bias=0;baseline.version='geometry-baseline'
            ranked=baseline.choose(context,rows,exploration=.15,seed=int(time.time()))
        ranked['evaluation_arm']=evaluation_arm;ranked['candidate_checkpoint']=candidate['id'] if candidate else None
        ranked['context']=context
        return ranked,ranked['rows'][ranked['selected']]['candidate']['point']

    def move(self,xyz,grip,settings):
        self.permit();current=self.arm.reference()['servo_deg']
        solved=self.model().ik(xyz,current[:5],settings['gripper_linkage_rad'],settings['grasp_quaternion_xyzw'])
        if not solved.get('solved') or solved.get('collision'):raise ValueError('collision_rejected: IK or collision check failed')
        goal=[int(round(v)) for v in solved['servo_deg']]+[int(round(grip))]
        if goal==[int(round(v)) for v in current]:return {'already_at_goal':True,'goal_deg':goal}
        plan=self.trajectory.plan(goal);started=self.trajectory.start_local(plan['plan_id'],self.permit)
        session=started['session'];deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            self.permit();state=self.trajectory.status()
            if state.get('session')!=session:raise ValueError('trajectory owner changed')
            if not state['busy']:
                if state.get('command_completed') is not True and state.get('reached') is not True:
                    raise ValueError('trajectory failed: '+str(state.get('reason')))
                return dict(state)
            time.sleep(.04)
        self.trajectory.stop();raise ValueError('trajectory timeout')

    def close_guarded(self,point,settings):
        angle=float(settings['open_deg']);commands=[]
        while angle<float(settings['close_deg']):
            step=6. if angle<130 else 3.;angle=min(float(settings['close_deg']),angle+step)
            commands.append(self.move(point,angle,settings))
            frames=self.vision.observe(self.permit,.18);decision=guarded_closure(frames,soft=True)
            if decision['action'] in ('hold','stop'):
                return dict(decision=decision,final_deg=angle,commands=commands)
        return dict(decision={'action':'hold','reason':'experimental_hold_limit'},final_deg=angle,commands=commands)

    @staticmethod
    def reset_zone(point,plane):
        x,y,_=point;a,b,c,d=plane
        if abs(c)<.5:raise ValueError('geometry_unknown: support plane is not horizontal enough')
        z=-(a*x+b*y+d)/c
        return dict(center_xyz=[float(x),float(y),float(z)],radius_m=.08,support_tolerance_m=.018)

    def trial(self,workflow):
        settings=self.settings();first,settings=self.detect(workflow['target'],settings)
        trial_number=workflow['autonomous_done']+1
        ranked,point=self.candidates(first,settings,trial_number);scene=str(first['object_id'])+'-'+str(first['at'])
        episode=self.workflows.begin_autonomous_episode(workflow['id'],scene,ranked['policy_version'],'rgbd_lift_v1')
        store=self.workflows.episodes;store.append(episode['id'],'observation',first)
        store.append(episode['id'],'proposed_action',{'grasp_selection':ranked})
        self.state.update(phase='approach',episode_id=episode['id'],physical_execution=True)
        point=np.asarray(point,dtype=float);above=point+[0,0,settings['approach_height_m']]
        self.move(above,settings['open_deg'],settings);self.move(point,settings['open_deg'],settings)
        before=self.vision.observe(self.permit,.4);closure=self.close_guarded(point,settings)
        store.append(episode['id'],'gripper_command',{'close':closure})
        self.move(point+[0,0,settings['lift_height_m']],closure['final_deg'],settings)
        after=self.vision.observe(self.permit,1.05);verdict=verify_lift(before,after,True)
        outcome=verdict['outcome'];reason=verdict['evidence'].get('reason','')
        store.append(episode['id'],'next_observation',after[-1])
        reward=1. if outcome=='success' else 0. if outcome=='failure' else None
        reset={'state':'not_required'}
        if outcome=='success':
            held=after;zone=self.reset_zone(point,settings['floor_plane_base']);self.state.update(phase='self_reset')
            self.move(point+[0,0,settings['approach_height_m']],closure['final_deg'],settings)
            self.move(point,closure['final_deg'],settings);self.move(point,settings['open_deg'],settings)
            self.move(above,settings['open_deg'],settings);placed=self.vision.observe(self.permit,1.05)
            reset_verdict=verify_place(held,placed,zone,True)
            reset={'state':reset_verdict['outcome'],'verdict':reset_verdict,'zone':zone}
        else:
            try:self.move(above,settings['open_deg'],settings)
            except (OSError,ValueError,KeyError):reset={'state':'unknown','reason':'failure withdrawal not confirmed'}
        result=self.workflows.finish_autonomous_episode(workflow['id'],outcome,
            {'verifier':verdict,'grasp_selection':ranked,'reset':reset},reason,reward)
        self.learn(episode['id'],ranked,outcome)
        if outcome=='success' and reset['state']!='success':self.workflows.needs_reset(workflow['id'])
        return result

    def learn(self,episode_id,ranked,outcome):
        if outcome not in ('success','failure'):return
        selected=ranked['rows'][ranked['selected']]['candidate'];sample=dict(episode_id=episode_id,
            context=ranked['context'],action=selected,outcome=outcome,policy_version=ranked['policy_version'],
            recorded_at=time.time())
        path=self.scorer.root/'samples.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:stream.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
        candidate_id=ranked.get('candidate_checkpoint');arm=ranked.get('evaluation_arm')
        if candidate_id and arm in ('baseline','candidate'):
            evaluation=self.scorer.record_evaluation(candidate_id,arm,outcome,episode_id)
            if evaluation['ready']:
                checkpoint=json.loads((self.scorer.root/(candidate_id+'.json')).read_text())
                self.scorer.promote(checkpoint)
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows)>=20 and len(rows)%10==0:self.scorer.train(rows)

    def run(self,workflow_id):
        try:
            while not self.cancelled.is_set():
                workflow=self.workflows.get(workflow_id)
                if workflow['state'] in ('complete','cancelled','failed','needs_reset','training_blocked'):break
                self.state.update(phase='trial_start',trial=workflow['autonomous_done']+1)
                self.trial(workflow)
            self.state.update(phase=self.workflows.get(workflow_id)['state'],physical_execution=False)
        except Exception as exc:
            try:
                workflow=self.workflows.get(workflow_id)
                if workflow.get('active_episode'):
                    self.workflows.finish_autonomous_episode(workflow_id,'infrastructure_error',
                        {'reason':str(exc)},'infrastructure_error',None)
                self.workflows.fail(workflow_id,str(exc))
            except (OSError,ValueError,KeyError,TypeError):pass
            self.state.update(phase='blocked',reason=str(exc),physical_execution=False)
        finally:
            self.vision.stop();self.active=None;self.lock.release()
