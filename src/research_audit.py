"""Evidence-backed status of the research-derived runtime capabilities."""
import json
from pathlib import Path
import time


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError,TypeError):return {}


def audit(root,world,perception=None,status=None,mission=None,delivery=None,learning=None):
    root=Path(root);perception=perception or read(root/'data/perception.json')
    status=status or read(root/'data/status.json');flags=status.get('commissioning',{})
    world_state=world.status();mission=mission or {};delivery=delivery or {};learning=learning or {}
    def item(ident,name,state,evidence,blockers=(),metrics=None):
        return dict(id=ident,name=name,state=state,evidence=evidence,blockers=list(blockers),metrics=metrics or {})
    nav_missing=[k for k in ('base_commissioned','mcu_watchdog_verified','lidar_tf_validated','localization_verified') if flags.get(k) is not True]
    arm_missing=[k for k in ('arm_commissioned','camera_tf_validated') if flags.get(k) is not True]
    camera_3d=flags.get('camera_tf_validated') is True and perception.get('world_coordinates_validated') is True
    gripper=read(root/'config/gripper-accepted.json')
    grasp_missing=[]
    if flags.get('arm_feedback_measured') is not True:grasp_missing.append('No measured robotio joint feedback')
    if not (gripper.get('execution_authorized') is True and gripper.get('aperture_mm_calibrated') is True):
        grasp_missing.append('No accepted gripper aperture/contact/retention calibration')
    learning_jobs=learning.get('jobs',[]) if isinstance(learning,dict) else []
    entries=[
      item(1,'Dynamic semantic world memory','DONE' if camera_3d else 'PARTIAL',
           'Persistent episodes, places, actions and conservative entity lifecycle are active.',
           [] if camera_3d else ['No accepted camera→map object transform'],world_state),
      item(2,'Spatial orientation and exploration','DONE' if not nav_missing else 'PARTIAL',
           'SLAM Toolbox, EKF, Nav2, place memory and reachable frontier search are installed.',nav_missing,
           {'mission':mission.get('active'),'pose_fresh':time.time()-status.get('at',0)<1}),
      item(3,'Active perception','PARTIAL','Next-view policy selects target inspection or frontier exploration; execution keeps existing safety gates.',list(dict.fromkeys(arm_missing+nav_missing))),
      item(4,'Guarded/contact-aware gripper','PARTIAL','Visual guarded-closure policy handles contact, deformation, slip and stale tracking.',grasp_missing+['No force/tactile sensor; physical visual thresholds not accepted']),
      item(5,'Grasp planning and retry','PARTIAL','RGB-D candidates, IK/collision checks, tracking and outcome verification exist. Stationary hand-eye is physically accepted.',grasp_missing),
      item(6,'Mobile manipulation','PARTIAL','Delivery owns base and arm in one transaction and checks stationary handoffs.',delivery.get('blocked_by',[])),
      item(7,'Reactive manipulation','PARTIAL','30 Hz feature/depth tracking and freshness checks exist; local action policy is available.',grasp_missing),
      item(8,'Learning and interventions','PARTIAL','Demonstrations, outcome/intervention replay, trainable scorer, outcome model, bounded actor/critic and ACT training persist. Physical learned-policy trials are not yet accepted.',[],
           {'jobs':len(learning_jobs),'framework':learning.get('backend',{}).get('lerobot')}),
      item(9,'Hierarchical task execution','PARTIAL','The persistent supervisor uses three-valued predicates and bounded cost search over a skill registry; the accepted delivery remains its physical executor.',delivery.get('blocked_by',[])),
      item(10,'Learned navigation concepts','DONE','Applicable concepts are implemented as Nav2 local feedback, risk/clearance-aware frontier ranking, replanning and measured goal checks; no unvalidated NavDP weights are loaded.'),
      item(11,'RoboBrain/RoboOS upper layer','DONE','Kept out of actuator control: current local tool planner already exposes bounded skills, while no compatible validated model offers a measured advantage on 8 GB.'),
      item(12,'Continual world model','PARTIAL','Visual episodes, entities, places, actions and relations update persistent memory with a scene generation. A trained high-dimensional world model still needs physical transitions.',[],world_state),
    ]
    return dict(at=time.time(),items=entries,summary={s:sum(e['state']==s for e in entries) for s in ('DONE','PARTIAL','MISSING')},
                physical_experiment_ready=not nav_missing and not arm_missing and not delivery.get('blocked_by'))
