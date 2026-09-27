"""Reference-only kinematics and strict MoveIt collision previews. No actuator IO."""
from pathlib import Path
import math, time, threading, json, hashlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from moveit.core.robot_model import RobotModel
from moveit.core.robot_state import RobotState
from moveit.core.planning_scene import PlanningScene
from moveit.core.collision_detection import CollisionRequest,CollisionResult
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from servo_coordinates import to_radians,to_servo
ROOT=Path('/home/vlad/Explorer')
LOW=np.radians([-90]*5);HIGH=np.radians([90,90,90,90,180])

class ArmModel:
    def __init__(self):
        manifest=json.loads((ROOT/'config/arm_model.json').read_text())
        if not manifest.get('asset_sha256'):raise ValueError('No verified geometry assets')
        for name,digest in manifest['asset_sha256'].items():
            path=ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Geometry missing or changed: '+name)
        self.model=RobotModel(str(ROOT/'config/explorer.urdf'),str(ROOT/'config/explorer.srdf'))
        self.state=RobotState(self.model)
        self.scene=PlanningScene(self.model)
        floor=CollisionObject();floor.header.frame_id='base_footprint';floor.id='floor'
        box=SolidPrimitive();box.type=SolidPrimitive.BOX;box.dimensions=[10.,10.,.02]
        pose=Pose();pose.orientation.w=1.;pose.position.z=-.02
        floor.primitives=[box];floor.primitive_poses=[pose];floor.operation=CollisionObject.ADD
        self.scene.apply_collision_object(floor)
        self.lock=threading.Lock()

    @staticmethod
    def vector(values,n):
        a=np.asarray(values,dtype=float)
        if a.shape!=(n,) or not np.all(np.isfinite(a)):raise ValueError(f'{n} finite values required')
        return a

    def set_state(self,servo_deg,gripper_rad):
        deg=self.vector(servo_deg,5);q=to_radians(deg)
        if np.any(q<LOW) or np.any(q>HIGH):raise ValueError('Servo reference range exceeded')
        if not math.isfinite(gripper_rad) or not -1.54<=gripper_rad<=0:raise ValueError('Invalid reference gripper linkage angle')
        self.state.set_to_default_values()
        self.state.set_joint_group_positions('arm',q)
        self.state.joint_positions={'rlink1_Joint':gripper_rad}
        self.state.update()
        return q

    def collision(self):
        req=CollisionRequest();res=CollisionResult()
        self.scene.check_collision(req,res,self.state)
        return bool(res.collision)

    def fk(self,servo_deg,gripper_rad):
        with self.lock:
            self.set_state(servo_deg,gripper_rad)
            t=self.state.get_global_link_transform('Gripping')
            return dict(frame='base_footprint',xyz=t[:3,3].tolist(),quaternion_xyzw=Rotation.from_matrix(t[:3,:3]).as_quat().tolist(),
                        collision=self.collision(),reference_only=True,measured_state=False,executed=False,
                        warning='Nominal CAD and servo convention, not measured joint state. Environment and hand-eye calibration still require validation. Not an execution permission.')

    def ik(self,xyz,seed_deg,gripper_rad,quaternion=None):
        target=self.vector(xyz,3)
        if np.linalg.norm(target)>1:raise ValueError('Target outside reference workspace')
        rot=None
        if quaternion is not None:
            quat=self.vector(quaternion,4)
            if abs(np.linalg.norm(quat)-1)>.001:raise ValueError('Unit quaternion required')
            rot=Rotation.from_quat(quat)
        with self.lock:
            seed=self.set_state(seed_deg,gripper_rad);start=time.monotonic()
            # Position-only IK must not exploit tiny CAD eccentricities by
            # spinning the wrist through its range. Keep the requested roll.
            def expand(q):return np.r_[q,seed[4]] if rot is None else q
            def residual(q):
                if time.monotonic()-start>2:raise TimeoutError('IK time budget exceeded')
                self.state.set_joint_group_positions('arm',expand(q));self.state.update()
                t=self.state.get_global_link_transform('Gripping')
                r=(t[:3,3]-target).tolist()
                if rot is not None:r.extend((.1*(rot.inv()*Rotation.from_matrix(t[:3,:3])).as_rotvec()).tolist())
                return r
            size=4 if rot is None else 5
            fit=least_squares(residual,np.clip(seed[:size],LOW[:size]+1e-8,HIGH[:size]-1e-8),bounds=(LOW[:size],HIGH[:size]),max_nfev=100,
                              ftol=1e-8,xtol=1e-8,gtol=1e-8)
            err=residual(fit.x);poserr=float(np.linalg.norm(err[:3]));angerr=float(np.linalg.norm(err[3:])/.1) if rot is not None else None
            collided=self.collision()
            return dict(solved=poserr<=.002 and (angerr is None or angerr<=.02),servo_deg=to_servo(expand(fit.x)).tolist(),
                        position_error_m=poserr,orientation_error_rad=angerr,collision=collided,
                        reference_only=True,executed=False,execution_allowed=False)

    def path(self,start_deg,goal_deg,gripper_rad):
        with self.lock:
            a=self.set_state(start_deg,gripper_rad);b=self.set_state(goal_deg,gripper_rad)
            n=max(2,int(np.ceil(np.max(np.abs(b-a))/np.radians(.5)))+1)
            for i,u in enumerate(np.linspace(0,1,n)):
                self.state.set_joint_group_positions('arm',a+(b-a)*u);self.state.update()
                if self.collision():return dict(valid=False,collision_fraction=float(u),checked=i+1,reference_only=True,executed=False)
            return dict(valid=True,checked=n,start_deg=list(start_deg),goal_deg=list(goal_deg),reference_only=True,
                        executed=False,execution_allowed=False,warning='Discrete geometric preview only; no environmental depth integration or verified calibration.')
