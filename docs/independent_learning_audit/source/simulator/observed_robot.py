"""Same observed controller on live sensors; MuJoCo is robot-only FK here.

No simulated camera, object, contact evaluation or dynamics step is available.
The transport owns the existing hardware permissions and verified executors.
"""
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .explorer_model import ARM_JOINTS,load_model,servo_to_joint


def rigid(value):
    matrix=np.asarray(value,float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]) or
            not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(matrix[:3,:3])<.999):
        raise ValueError('Invalid measured robot transform')
    return matrix


class ObservedRobotEnv:
    def __init__(self,transport,*,mesh_dir=None):
        self.transport=transport;self.model,self.data=load_model(scene='empty',mesh_dir=mesh_dir)
        self.joint_q=np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.pad_ids=[self.model.geom(n).id for n in ('finger_left','finger_right')]
        self.wheel_ids=np.array([self.model.actuator('wheel'+str(i)).id for i in range(4)])
        motion=json.loads((Path(__file__).resolve().parents[1]/'config/arm-motion.json').read_text())
        self.velocity_limit=np.asarray(motion['streaming']['velocity_deg_s'],float)
        self.command=np.zeros(9);self.pending_command=np.zeros(9);self.phase='ready';self.steps=0
        self.last_error=None;self._scope=None;self._last_frame=None;self._frame=None
        self._sync(self.transport.state())

    def _sync(self,state):
        now=time.monotonic();stamp=float(state['acquired_monotonic'])
        if not 0<=now-stamp<.4:raise ValueError('Live joint/odometry sample is stale')
        if state.get('floor_transform_validated') is not True:raise ValueError('Measured floor transform is required')
        if state.get('joint_state_source') not in ('measured','accepted_command_estimate'):
            raise ValueError('Unaccepted arm state provenance')
        scope=(state['boot_id'],state['map_epoch'],state['calibration_sha256'])
        if any(not isinstance(x,str) or not x for x in scope):raise ValueError('Missing live state provenance')
        if self._scope is not None and self._scope!=scope:raise ValueError('Robot boot, map or calibration changed')
        self._scope=scope
        command=np.asarray([*state['servo_deg'],*state['body_velocity']],float)
        if command.shape!=(9,) or not np.isfinite(command).all():raise ValueError('Incomplete live robot state')
        q=servo_to_joint(command[:6]);base=rigid(state['base_to_world'])
        self.data.qpos[:3]=base[:3,3]
        xyzw=Rotation.from_matrix(base[:3,:3]).as_quat();self.data.qpos[3:7]=xyzw[[3,0,1,2]]
        self.data.qpos[self.joint_q]=q
        for n,m in [('rlink2_Joint',1),('llink1_Joint',1),('llink2_Joint',-1),('rlink3_Joint',-1),('llink3_Joint',1)]:
            self.data.qpos[self.model.joint(n).qposadr[0]]=m*q[5]
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data);mujoco.mj_camlight(self.model,self.data)
        self.command=command;self.pending_command=command.copy();self.data.time=stamp

    def jaw_center(self,data=None):return np.mean((data or self.data).geom_xpos[self.pad_ids],axis=0)

    def base_pose(self):
        body=self.data.body('base_link');r=body.xmat.reshape(3,3)
        return [float(body.xpos[0]),float(body.xpos[1]),float(np.arctan2(r[1,0],r[0,0]))]

    def solve_ik(self,*args,**kwargs):
        from .environment import ExplorerEnv
        return ExplorerEnv.solve_ik(self,*args,**kwargs)

    def step(self,action):
        from .environment import LOW,HIGH,PERIOD
        action=np.asarray(action,float)
        if action.shape!=(9,) or not np.isfinite(action).all() or np.any(action<LOW) or np.any(action>HIGH):
            raise ValueError('Live action outside qualified command contract')
        self.transport.permit()
        state=self.transport.apply(action.copy(),reference=self.command.copy(),period=PERIOD)
        self._sync(state);self.steps+=1;self.transport.permit()

    def render_rgb(self,camera='wrist'):
        if camera!='wrist' or self._frame is None:raise ValueError('Live robot has no synthetic overview camera')
        return self._frame['rgb'].copy()

    def _robot_depth(self,frame):
        """Raycast known robot CAD at measured intrinsics, without OpenGL."""
        k=frame['intrinsics'];h,w=frame['depth'].shape;row,col=np.indices((h,w))
        optical=np.stack(((col-k['cx'])/k['fx'],(row-k['cy'])/k['fy'],np.ones((h,w))),-1).reshape(-1,3)
        norm=np.linalg.norm(optical,axis=1);transform=rigid(frame['optical_to_world'])
        vectors=(optical/norm[:,None])@transform[:3,:3].T
        distance=np.full(h*w,-1.,float);ids=np.full(h*w,-1,np.int32)
        mujoco.mj_multiRay(self.model,self.data,transform[:3,3].copy(),vectors.ravel().copy(),
            np.array([0,0,1,0,0,0],np.uint8),True,-1,ids,distance,None,h*w,3.)
        return np.where(distance>=0,distance/norm,np.nan).reshape(h,w)

    def capture_registered_frame(self,raw_output=None):
        from .rgbd_observation import deproject_depth,extract_candidates
        self.transport.permit();frame=self.transport.capture(after_frame=self._last_frame)
        if frame['frame_id']==self._last_frame:raise ValueError('RGB-D exposure did not advance')
        if frame.get('registered') is not True or frame.get('arm_stationary') is not True or frame.get('depth_unit')!='metre':
            raise ValueError('Live RGB-D must be registered and acquired after arm motion completed')
        self._sync(frame['robot_state']);stamp=float(frame['acquired_monotonic'])
        if not 0<=time.monotonic()-stamp<.4 or abs(stamp-self.data.time)>.075:
            raise ValueError('RGB-D and robot kinematics are not synchronized')
        depth=np.asarray(frame['depth'],float);rgb=np.asarray(frame['rgb'])
        if depth.ndim!=2 or rgb.shape!=(*depth.shape,3) or rgb.dtype!=np.uint8:raise ValueError('Malformed registered RGB-D')
        k=frame['intrinsics'];transform=rigid(frame['optical_to_world'])
        if (k.get('height'),k.get('width'))!=depth.shape:raise ValueError('RGB-D intrinsics/image size mismatch')
        # Validate intrinsics before known-CAD projection; never rescale a real
        # camera to synthetic intrinsics or substitute rendered sensor pixels.
        deproject_depth(depth,k,transform)
        camera=self.model.camera('wrist_depth').id
        expected=self.data.cam_xmat[camera].reshape(3,3)@np.diag([1.,-1.,-1.])
        if (np.linalg.norm(self.data.cam_xpos[camera]-transform[:3,3])>.008 or
                Rotation.from_matrix(expected.T@transform[:3,:3]).magnitude()>np.deg2rad(2.)):
            raise ValueError('Live camera extrinsics differ from qualified robot geometry')
        prediction=self._robot_depth(frame)
        result=extract_candidates(depth,k,transform,robot_depth=prediction,rgb=rgb)
        self._frame=dict(frame,rgb=rgb.copy());self._last_frame=frame['frame_id']
        if raw_output is not None:raw_output.update(depth=depth.copy(),rgb=rgb.copy(),robot_depth=prediction)
        return dict(schema=1,frame_id=frame['frame_id'],simulation_time=stamp,captured_monotonic=stamp,
            intrinsics=k,optical_to_world=transform.tolist(),source='live_registered_RGBD',simulation_only=False,
            calibration=dict(floor_status='measured_and_validated',floor_prior_z_m=0.,real_calibration_validated=True),
            robot_self_filter=dict(available=True,source='robot_only_CAD_metric_ray_projection',depth_tolerance_m=.008),**result)

    def lidar_scan(self,scanner=0,rays=180):
        if scanner not in (0,1) or not 8<=rays<=720:raise ValueError('Invalid live scanner request')
        value=self.transport.scan(scanner)
        if not 0<=time.monotonic()-value['acquired_monotonic']<.6:raise ValueError('Live lidar is stale')
        # Preserve the actual scan angular grid. Consumers already derive
        # directions from these fields; no interpolation fabricates free space.
        if len(value['ranges'])<8:raise ValueError('Incomplete live lidar')
        site=self.data.site('laser'+str(scanner)+'_frame')
        transform=rigid(value['sensor_to_world'])
        if np.linalg.norm(site.xpos-transform[:3,3])>.012 or np.linalg.norm(site.xmat.reshape(3,3)-transform[:3,:3])>.03:
            raise ValueError('Live lidar frame differs from qualified geometry')
        return dict(value,simulation_only=False)

    def stop_motion(self):self.transport.stop()

    def close(self):self.stop_motion()


def run_candidate(folder,transport,output,*,model_root,mesh_dir=None,progress=lambda _:None):
    """Explicit observed trial; importing or validating a package cannot call it."""
    from .observed_transfer import validate
    from .skill_workshop import execution_fingerprint
    from .grounded_skill_port import GroundedSkillPort
    from src.task_executor import TaskExecutor
    checked=validate(folder);policy=checked['policy']
    if execution_fingerprint()['sha256']!=policy['controller_sha256']:
        raise ValueError('Installed execution code differs from independently qualified code')
    transport.preflight(checked)
    env=None;port=None
    try:
        env=ObservedRobotEnv(transport,mesh_dir=mesh_dir)
        port=GroundedSkillPort(env,output,transport.cancelled,progress,parameters=policy['parameters'],
            repair_program=policy['controller_program'],experience=checked['experience'],
            semantic_root=model_root,query=policy['scene_recipe']['query'])
        executor=TaskExecutor(port.operations(),port.observe,guard=lambda ctx:transport.permit(),
            cancelled=transport.cancelled,stop=lambda ctx:transport.stop(),total_timeout=900.,stage_timeout=180.,
            max_find_attempts=1,max_grasp_attempts=3)
        return executor.execute(dict(task_id=policy['task_id']))
    finally:
        try:
            if port is not None:port.close()
        finally:transport.stop()
