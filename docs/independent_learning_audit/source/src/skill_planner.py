"""State-dependent skill registry and bounded cost search for Explorer."""
from dataclasses import replace
import heapq
import json
from pathlib import Path
from autonomy_contracts import SkillSpec, Truth, record


BUILTINS=(
 SkillSpec('observe_scene','1',{}, {'scene_observed':Truth.TRUE.value},1,'observe','fresh_observation',resolves=('scene_observed',)),
 SkillSpec('locate_object','1',{'scene_observed':Truth.TRUE.value},{'object_localized':Truth.TRUE.value},2,'object_finder','tracked_rgbd',resolves=('object_localized',)),
 SkillSpec('navigate','1',{'object_localized':Truth.TRUE.value,'localization_valid':Truth.TRUE.value},{'base_aligned':Truth.TRUE.value},3,'missions','pose_goal',permissions=('base_motion',)),
 SkillSpec('relocalize','1',{}, {'localization_valid':Truth.TRUE.value},2,'missions','localization_health',resolves=('localization_valid',),recovery_for=('localization_lost',)),
 SkillSpec('inspect_reachability','1',{'object_localized':Truth.TRUE.value},{'grasp_reachable':Truth.TRUE.value},1.5,'delivery_robot','ik_and_collision',resolves=('grasp_reachable',)),
 SkillSpec('change_view','1',{'scene_observed':Truth.TRUE.value},{'object_localized':Truth.TRUE.value},2.2,'delivery_robot','tracked_rgbd',permissions=('base_motion',),recovery_for=('track_lost','bad_depth')),
 SkillSpec('prepare_scene','1',{'object_localized':Truth.TRUE.value,'contact_allowed':Truth.TRUE.value},{'grasp_reachable':Truth.TRUE.value,'scene_changed':Truth.TRUE.value},4,'delivery_robot','scene_change_v1',permissions=('arm_motion','target_contact'),recovery_for=('unreachable',)),
 SkillSpec('align_gripper','1',{'base_aligned':Truth.TRUE.value,'grasp_reachable':Truth.TRUE.value},{'gripper_aligned':Truth.TRUE.value},2,'delivery_robot','alignment_geometry',permissions=('arm_motion',)),
 SkillSpec('grasp','1',{'gripper_aligned':Truth.TRUE.value},{'grasp_attempted':Truth.TRUE.value},3,'delivery_robot','rgbd_lift_v1',permissions=('arm_motion',)),
 SkillSpec('verify_hold','1',{'grasp_attempted':Truth.TRUE.value},{'object_held':Truth.TRUE.value},1,'delivery_robot','rgbd_lift_v1',resolves=('object_held',)),
 SkillSpec('regrasp','1',{'object_localized':Truth.TRUE.value},{'grasp_attempted':Truth.TRUE.value},3.5,'delivery_robot','rgbd_lift_v1',permissions=('arm_motion',),recovery_for=('empty_grasp','slip')),
 SkillSpec('transport','1',{'object_held':Truth.TRUE.value,'destination_localized':Truth.TRUE.value,'localization_valid':Truth.TRUE.value},{'at_destination':Truth.TRUE.value},4,'delivery_robot','pose_goal',permissions=('base_motion','arm_motion')),
 SkillSpec('observe_destination','1',{}, {'destination_localized':Truth.TRUE.value},1.5,'semantic_world','fresh_observation',resolves=('destination_localized',)),
 SkillSpec('support_object','1',{'at_destination':Truth.TRUE.value},{'object_supported':Truth.TRUE.value},2,'delivery_robot','support_geometry',permissions=('arm_motion',)),
 SkillSpec('release','1',{'object_supported':Truth.TRUE.value},{'release_attempted':Truth.TRUE.value},1,'delivery_robot','rgbd_place_v2',permissions=('arm_motion',)),
 SkillSpec('verify_place','2',{'release_attempted':Truth.TRUE.value},{'placed':Truth.TRUE.value},1,'delivery_robot','rgbd_place_v2',resolves=('placed',)),
 SkillSpec('reset_scene','1',{'placed':Truth.TRUE.value},{'reset_ready':Truth.TRUE.value},3,'delivery_robot','reset_v1',permissions=('arm_motion',)),
)


class SkillRegistry:
    def __init__(self, root):
        self.root=Path(root);self.path=self.root/'data/skills/registry.json'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.skills={skill.id:skill.validate() for skill in BUILTINS}
        self.load_discovered()

    def load_discovered(self):
        try:rows=json.loads(self.path.read_text()).get('skills',[])
        except (OSError,ValueError,TypeError):rows=[]
        for row in rows:
            try:
                skill=SkillSpec(**row).validate()
                if skill.source=='discovered' and skill.id not in self.skills:self.skills[skill.id]=skill
            except (TypeError,ValueError):pass

    def register(self, skill):
        skill.validate()
        if skill.source!='discovered':raise ValueError('Runtime registry only accepts discovered skills')
        if skill.id in {item.id for item in BUILTINS}:raise ValueError('Cannot replace a built-in skill')
        self.skills[skill.id]=skill
        rows=[record(item) for item in self.skills.values() if item.source=='discovered']
        temporary=self.path.with_suffix('.tmp');temporary.write_text(json.dumps({'skills':rows},ensure_ascii=False,indent=2))
        temporary.replace(self.path)
        return record(skill)

    def catalog(self):return [record(skill) for skill in self.skills.values()]


class TaskPlanner:
    def __init__(self, registry, maximum_nodes=2000):
        self.registry=registry;self.maximum_nodes=maximum_nodes

    @staticmethod
    def key(state):return tuple(sorted(state.items()))

    def plan(self, predicates, targets, scene_version, allowed_skills=None, permissions=()):
        initial={key:(value.value if isinstance(value,Truth) else value) for key,value in predicates.items()}
        if any(value not in {item.value for item in Truth} for value in initial.values()):
            raise ValueError('Planner state must be three-valued')
        permitted=set(permissions);allowed=set(allowed_skills or self.registry.skills)
        queue=[(0.,0,self.key(initial),initial,[])];best={self.key(initial):0.};serial=0;expanded=0
        while queue and expanded<self.maximum_nodes:
            cost,_,_,state,path=heapq.heappop(queue);expanded+=1
            if all(state.get(key)==Truth.TRUE.value for key in targets):
                return dict(scene_version=scene_version,cost=cost,steps=path,expanded=expanded,
                            target_predicates=targets,planner='bounded_uniform_cost_v1')
            for skill in self.registry.skills.values():
                if skill.id in allowed and set(skill.permissions)<=permitted:
                    false=[key for key,value in skill.preconditions.items() if state.get(key,Truth.UNKNOWN.value)==Truth.FALSE.value and value==Truth.TRUE.value]
                    unknown=[key for key,value in skill.preconditions.items() if state.get(key,Truth.UNKNOWN.value)==Truth.UNKNOWN.value and value==Truth.TRUE.value]
                    applicable=not false and not unknown
                    if applicable:
                        nxt=dict(state);nxt.update(skill.effects);new_cost=cost+skill.cost;state_key=self.key(nxt)
                        if new_cost<best.get(state_key,float('inf')):
                            best[state_key]=new_cost;serial+=1
                            step=dict(skill=skill.id,version=skill.version,executor=skill.executor,
                                      verifier=skill.verifier,cost=skill.cost,scene_version=scene_version,
                                      revalidate=list(skill.preconditions))
                            heapq.heappush(queue,(new_cost,serial,state_key,nxt,path+[step]))
                    elif not false and unknown:
                        probes=[item for item in self.registry.skills.values() if set(unknown)&set(item.resolves)]
                        for probe in probes:
                            probe_ready=all(state.get(key,Truth.UNKNOWN.value)==value for key,value in probe.preconditions.items())
                            if probe.id in allowed and set(probe.permissions)<=permitted and probe_ready:
                                optimistic=dict(state);optimistic.update(probe.effects)
                                for key in unknown:optimistic[key]=Truth.TRUE.value
                                new_cost=cost+probe.cost;state_key=self.key(optimistic)
                                if new_cost<best.get(state_key,float('inf')):
                                    best[state_key]=new_cost;serial+=1
                                    step=dict(skill=probe.id,version=probe.version,executor=probe.executor,
                                              verifier=probe.verifier,cost=probe.cost,scene_version=scene_version,
                                              resolves=unknown,replan_after=True,revalidate=[])
                                    heapq.heappush(queue,(new_cost,serial,state_key,optimistic,path+[step]))
        return dict(scene_version=scene_version,cost=None,steps=[],expanded=expanded,target_predicates=targets,
                    planner='bounded_uniform_cost_v1',blocked=True,reason='No plan satisfies predicates and permissions')


def recovery_plan(failure_kind, registry, attempted=()):
    candidates=[skill for skill in registry.skills.values() if failure_kind in skill.recovery_for and skill.id not in attempted]
    return [record(skill) for skill in sorted(candidates,key=lambda item:item.cost)]
