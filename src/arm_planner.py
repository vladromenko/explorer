"""MoveIt OMPL planning with explicit reference state; never publishes actuators."""
import math
import threading
from pathlib import Path
import numpy as np
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from arm_model import ArmModel, ROOT
from servo_coordinates import to_radians,to_servo


class ArmPlanner:
    def __init__(self):
        self.guard = ArmModel()  # verify pinned assets before loading plugins
        self.lock = threading.Lock()
        config = {
            'robot_description': (ROOT/'config/explorer.urdf').read_text(),
            'robot_description_semantic': (ROOT/'config/explorer.srdf').read_text(),
            'planning_scene_monitor_options': {
                'name': 'explorer_arm_reference_scene',
                'robot_description': 'robot_description',
                'joint_state_topic': '/explorer/unavailable_measured_arm_states',
                'wait_for_initial_state_timeout': 0.0,
            },
            'planning_pipelines': {'pipeline_names': ['ompl']},
            'ompl': {
                'planning_plugins': ['ompl_interface/OMPLPlanner'],
                'request_adapters': ['default_planning_request_adapters/CheckStartStateBounds',
                                     'default_planning_request_adapters/CheckStartStateCollision'],
                'response_adapters': ['default_planning_response_adapters/AddTimeOptimalParameterization',
                                      'default_planning_response_adapters/ValidateSolution'],
                'planner_configs': {'RRTConnectkConfigDefault': {'type': 'geometric::RRTConnect', 'range': 0.1}},
                'arm': {'planner_configs': ['RRTConnectkConfigDefault'],
                        'longest_valid_segment_fraction': 0.002},
            },
            'plan_request_params': {
                'planning_pipeline': 'ompl', 'planner_id': 'RRTConnectkConfigDefault',
                'planning_attempts': 1, 'planning_time': 2.0,
                'max_velocity_scaling_factor': 0.1, 'max_acceleration_scaling_factor': 0.1,
            },
            'trajectory_execution': {'manage_controllers': False},
            'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
            'robot_description_planning': {'joint_limits': {
                f'arm{i}_Joint': {'has_velocity_limits': True, 'max_velocity': 0.3,
                                 'has_acceleration_limits': True, 'max_acceleration': 0.5}
                for i in range(1, 6)}},
        }
        self.robot = MoveItPy(node_name='explorer_arm_planner', config_dict=config,
                              provide_planning_service=False)
        self.component = self.robot.get_planning_component('arm')
        self.monitor = self.robot.get_planning_scene_monitor()

    def state(self, values, gripper):
        self.guard.set_state(values, gripper)
        state = RobotState(self.robot.get_robot_model())
        state.set_to_default_values()
        state.set_joint_group_positions('arm', to_radians(values))
        state.joint_positions = {'rlink1_Joint': gripper}
        state.update()
        return state

    def plan(self, start_deg, goal_deg, gripper_rad, obstacles=()):
        """Obstacles are explicit base-frame boxes, not inferred from stale images."""
        with self.lock:
            start = self.state(start_deg, gripper_rad)
            goal = self.state(goal_deg, gripper_rad)
            boxes = [dict(center=[0., 0., -.02], size=[10., 10., .02]), *obstacles]
            with self.monitor.read_write() as scene:
                scene.remove_all_collision_objects()
                self.guard.scene.remove_all_collision_objects()
                for i, box in enumerate(boxes):
                    center = ArmModel.vector(box['center'], 3)
                    size = ArmModel.vector(box['size'], 3)
                    if np.any(size <= 0):raise ValueError('Positive obstacle size required')
                    obj = CollisionObject(); obj.id = 'obstacle_'+str(i)
                    obj.header.frame_id = 'base_footprint'; obj.operation = CollisionObject.ADD
                    primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size.tolist())
                    pose = Pose(); pose.orientation.w = 1.
                    pose.position.x, pose.position.y, pose.position.z = center.tolist()
                    obj.primitives = [primitive]; obj.primitive_poses = [pose]
                    scene.apply_collision_object(obj)
                    self.guard.scene.apply_collision_object(obj)
                scene.current_state.update()
            self.component.set_start_state(robot_state=start)
            self.component.set_goal_state(robot_state=goal)
            result = self.component.plan()
            if not result:
                return dict(planned=False, executed=False, execution_allowed=False,
                            planner='MoveIt2/OMPL/RRTConnect', reason='No valid path')
            trajectory = result.trajectory.get_robot_trajectory_msg().joint_trajectory
            # Servo packet interpolation differs from MoveIt timing. Recheck each
            # returned edge at <=0.5 degrees, including other plausible jaw shapes.
            names = [f'arm{i}_Joint' for i in range(1, 6)]
            order = [list(trajectory.joint_names).index(n) for n in names]
            points = [to_servo([p.positions[i] for i in order]).tolist()
                      for p in trajectory.points]
            # URDF limits use rounded radians (1.5708), while the servo API
            # uses exact degrees. Accept only sub-millidegree numerical drift.
            normalized=[]
            for point in points:
                a=np.asarray(point);upper=np.array([180.,180.,180.,180.,270.])
                if not np.isfinite(a).all() or np.any(a < -.001) or np.any(a > upper+.001):
                    raise ValueError('MoveIt path exceeds physical servo limits')
                normalized.append(np.clip(a,0.,upper).tolist())
            points=normalized
            if len(points) < 2:raise ValueError('Incomplete trajectory')
            if not np.allclose(points[0], start_deg, atol=.1) or not np.allclose(points[-1], goal_deg, atol=.1):
                raise ValueError('Plan endpoints disagree with request')
            for a, b in zip(points, points[1:]):
                for q in (0., -.2, -.4, -.6, -.8):
                    if not self.guard.path(a, b, q)['valid']:
                        raise ValueError('Trajectory fails conservative jaw-envelope check')
            return dict(planned=True, planner='MoveIt2/OMPL/RRTConnect', servo_waypoints=points,
                        time_from_start_s=[p.time_from_start.sec+p.time_from_start.nanosec/1e9
                                           for p in trajectory.points],
                        start_deg=list(start_deg), goal_deg=list(goal_deg),
                        obstacles=list(obstacles), measured_state=False, reference_only=True,
                        executed=False, execution_allowed=False,
                        timing_validated=False, environment_calibration_validated=False)
