"""MuJoCo Explorer environment with physical contact and portable observations.

The policy sees commanded state and wrist RGB, as on robotio. Ground truth is
available only to the simulator's teacher, diagnostics and outcome evaluator.
No ROS, SSH, serial, simulated object welding or chassis teleportation.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .explorer_model import (ARM_JOINTS, load_model, model_config, model_metadata,
                             servo_to_joint, joint_to_servo, wheel_targets)

ROOT = Path(__file__).resolve().parents[1]
PERIOD = .05
LOW = np.array([0., 0., 0., 0., 0., 30., -.2, -.2, -.6])
HIGH = np.array([180., 180., 180., 180., 270., 170., .2, .2, .6])


class ExplorerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config=None, *, scene="sock", mesh_dir=None, render=True):
        super().__init__()
        self.config = config or {}
        self.scene, self.mesh_dir = scene, mesh_dir
        self.render_enabled = render
        self.action_space = gym.spaces.Box(LOW.astype(np.float32), HIGH.astype(np.float32))
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "wrist": gym.spaces.Box(0, 255, (240, 320, 3), np.uint8)})
        motion = json.loads((ROOT / "config/arm-motion.json").read_text())
        self.velocity_limit = np.array(motion["streaming"]["velocity_deg_s"], float)
        self.acceleration_limit = np.array(motion["streaming"]["acceleration_deg_s2"], float)
        self.jerk_limit = np.array(motion["streaming"]["jerk_deg_s3"], float)
        self.renderer = None
        self.overview_renderer = None
        self._rgb_cache = {}
        self.render_options = mujoco.MjvOption()
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = False
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = False
        self.reset(seed=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        from .curriculum import task_spec
        self.task_id = options.get('task_id', getattr(self, 'task_id', 'sock_basket'))
        spec = task_spec(self.task_id)
        if self.scene != 'empty' and self.scene != 'navigation':
            self.scene = spec['scene']
        self._rgb_cache.clear()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.overview_renderer is not None:
            self.overview_renderer.close()
            self.overview_renderer = None
        varied = options.get("difficulty") == "varied"
        from .domain_randomization import PROFILE, sample_sock_scene
        profile = options.get('scene_profile', 'baseline')
        if profile not in ('baseline', PROFILE):
            raise ValueError('Unknown scene profile')
        sampled = sample_sock_scene(int(seed or 0)) if profile == PROFILE and self.scene in ('sock', 'cloth') else None
        sock_default = [self.np_random.uniform(.28, .36), self.np_random.uniform(-.025, .025), .015] if varied else [.31, 0., .015]
        bin_default = [.44, self.np_random.uniform(.20, .26), 0.] if varied else [.44, .24, 0.]
        if spec['rooms']:
            sock_default = [self.np_random.uniform(1.02,1.10),self.np_random.uniform(-.025,.025),.025]
            bin_default = [.35,.25,0.]
        if sampled:
            sock_default, bin_default = sampled['sock_position'], sampled['bin_position']
        self.sock_position = np.array(options.get("sock_position", sock_default), float)
        self.bin_position = np.array(options.get("bin_position", bin_default), float)
        self.model, self.data = load_model(mesh_dir=self.mesh_dir, scene=self.scene,
            sock_position=self.sock_position, bin_position=self.bin_position,
            scene_parameters=sampled['parameters'] if sampled else None)
        self.domain_variation = {"enabled": varied, "kind": "appearance_and_placement"}
        if varied:
            # Visual variation does not pretend to identify real contact physics.
            palette = np.array([[.82, .27, .23], [.16, .18, .23], [.72, .72, .67], [.18, .36, .70]])
            color = palette[self.np_random.integers(len(palette))]
            self.model.flex_rgba[:, :3] = color
            self.model.light_diffuse[:] *= self.np_random.uniform(.65, 1.15)
            self.model.geom("floor").rgba[:3] = self.np_random.uniform(.12, .38, 3)
            self.domain_variation["sock_rgb"] = color.tolist()
        if sampled:
            self.domain_variation = dict(sampled, enabled=True, kind='bounded_geometry_material_appearance_placement')
            self.model.flex_rgba[:, :3] = sampled['sock_rgb']
            self.model.light_diffuse[:] *= sampled['lighting_scale']
            self.model.geom('floor').rgba[:3] = sampled['floor_rgb']
            for index in range(self.model.ngeom):
                if self.model.geom(index).name.startswith('bin_'):
                    self.model.geom(index).rgba[:3] = sampled['basket_rgb']
        self.model.vis.global_.offwidth = 800
        self.model.vis.global_.offheight = 600
        self.joint_q = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.servo_ids = np.array([self.model.actuator(f"servo{i}").id for i in range(1, 7)])
        self.wheel_ids = np.array([self.model.actuator(f"wheel{i}").id for i in range(4)])
        self.pad_ids = [self.model.geom(name).id for name in ("finger_left", "finger_right")]
        self.sock_ids = np.array([i for i in range(self.model.nbody)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("sock_")],dtype=int)
        self.rigid_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'task_object')
        if self.scene!='empty':
            self.bin_half_size = np.array(sampled['parameters']['bin']['inner_half_size_m']) if sampled else self.model.geom('bin_bottom').size[:2]-model_config()['bin']['wall_thickness_m']
            self.bin_height = 2*self.model.geom('bin_left').size[2]
            self.bin_shape = sampled['parameters']['bin']['shape'] if sampled else 'rectangular'
        if self.rigid_id >= 0:
            self.sock_ids = np.array([self.rigid_id])
        self.command = np.array([90., 90., 90., 90., 90., 30., 0., 0., 0.])
        self.pending_command = self.command.copy()
        self.arm_velocity = np.zeros(6)
        self.arm_acceleration = np.zeros(6)
        self.phase = "ready"
        self.steps = 0
        self.lift_hold_s = 0.
        self.placement_hold_s = 0.
        self.ever_lifted = False
        self.ever_grasped = False
        self.contacts = [False, False]
        self.task_success = False
        self.last_error = None
        # Let gravity settle the cloth and wheels, without changing commands.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.start_height = self._sock_points()[:, 2].mean() if len(self.sock_ids) else 0.
        self.initial_sock_points = self._sock_points().copy()
        return self._observation(), self.state()

    def set_manual(self, command9):
        value = np.asarray(command9, dtype=float)
        if value.shape != (9,) or not np.isfinite(value).all():
            raise ValueError("Нужны шесть конечных углов и три скорости шасси")
        if np.any(value < LOW) or np.any(value > HIGH):
            raise ValueError("Команды вне диапазонов симуляции")
        self.pending_command = value.copy()

    def _advance_commands(self, dt):
        delta = self.pending_command[:6] - self.command[:6]
        desired_v = np.clip(delta / dt, -self.velocity_limit, self.velocity_limit)
        desired_a = np.clip((desired_v-self.arm_velocity)/dt,
                            -self.acceleration_limit, self.acceleration_limit)
        self.arm_acceleration += np.clip(desired_a-self.arm_acceleration,
                                         -self.jerk_limit*dt, self.jerk_limit*dt)
        self.arm_velocity = np.clip(self.arm_velocity+self.arm_acceleration*dt,
                                    -self.velocity_limit, self.velocity_limit)
        step = self.arm_velocity*dt
        reached = (np.sign(step) == np.sign(delta)) & (np.abs(step) >= np.abs(delta))
        self.command[:6] = np.clip(self.command[:6]+np.where(reached, delta, step), LOW[:6], HIGH[:6])
        self.arm_velocity[reached] = 0.
        self.arm_acceleration[reached] = 0.
        self.command[6:] += np.clip(self.pending_command[6:]-self.command[6:], -.5*dt, .5*dt)
        self.data.ctrl[self.servo_ids] = servo_to_joint(self.command[:6])
        self.data.ctrl[self.wheel_ids] = wheel_targets(*self.command[6:])

    def step(self, action, *, applied=False):
        self.set_manual(action)
        substeps = max(1, round(PERIOD / self.model.opt.timestep))
        start = self.command.copy()
        # Recorded actions are the endpoints already issued by the host profile.
        # Running that profile a second time changes the demonstrated path.
        # Interpolate issued endpoints with the existing speed limits; never
        # write qpos or attach the object. Manual/teacher goals retain the planner.
        endpoint = start.copy()
        endpoint[:6] += np.clip(self.pending_command[:6]-start[:6],
                                -self.velocity_limit*PERIOD, self.velocity_limit*PERIOD)
        endpoint[6:] += np.clip(self.pending_command[6:]-start[6:], -.5*PERIOD, .5*PERIOD)
        for index in range(substeps):
            if applied:
                self.command = start + (endpoint-start)*(index+1)/substeps
                self.data.ctrl[self.servo_ids] = servo_to_joint(self.command[:6])
                self.data.ctrl[self.wheel_ids] = wheel_targets(*self.command[6:])
            else:
                self._advance_commands(self.model.opt.timestep)
            mujoco.mj_step(self.model, self.data)
        if applied:
            self.arm_velocity = (self.command[:6]-start[:6])/PERIOD
            self.arm_acceleration[:] = 0.
        self.steps += 1
        self._evaluate_contact(PERIOD)
        if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
            self.last_error = "Численная нестабильность физики; попытка не принимается"
        info = self.state()
        reward = float(self.ever_grasped) + 2.*float(self.ever_lifted) + 5.*float(self.task_success)
        terminated = self.task_success or self.last_error is not None
        truncated = self.steps >= 2400
        return self._observation(), reward, terminated, truncated, info

    def _observation(self):
        image = self.render_rgb("wrist") if self.render_enabled else np.zeros((240, 320, 3), np.uint8)
        return {"state": self.command.astype(np.float32).copy(), "wrist": image}

    def _sock_points(self):
        if self.rigid_id >= 0:
            half = self.model.geom('task_object_contact').size
            corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
            return corners @ self.data.xmat[self.rigid_id].reshape(3,3).T + self.data.xpos[self.rigid_id]
        return self.data.xpos[self.sock_ids].copy()

    def jaw_center(self, data=None):
        d = data or self.data
        return np.mean(d.geom_xpos[self.pad_ids], axis=0)

    def _evaluate_contact(self, dt):
        points = self._sock_points()
        pads = [False, False]
        force = np.zeros(6)
        for contact_index, contact in enumerate(self.data.contact[:self.data.ncon]):
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            flex = list(contact.flex)
            rigid_contact = self.rigid_id >= 0 and self.model.geom('task_object_contact').id in contact.geom
            if any(i >= 0 for i in flex) or rigid_contact:
                for i, pad in enumerate(self.pad_ids):
                    if pad in contact.geom and contact.dist <= .001 and force[0] > .0001:
                        pads[i] = True
        self.contacts = pads
        grasped = all(pads)
        self.ever_grasped = self.ever_grasped or grasped
        if len(points):
            raised = points[:, 2].mean() - self.start_height > .045
            near_jaw = np.linalg.norm(points.mean(axis=0)-self.jaw_center()) < .12
            held = raised and near_jaw and grasped
            self.lift_hold_s = self.lift_hold_s + dt if held else 0.
            self.ever_lifted = self.ever_lifted or self.lift_hold_s >= 1.
            half = self.bin_half_size
            relative = points[:, :2]-self.bin_position[:2]
            inside_fraction = np.mean(np.linalg.norm(relative, axis=1) < half[0]) if self.bin_shape == 'round' else np.mean(np.all(np.abs(relative) < half, axis=1))
            height = self.bin_height
            within_height = np.all((points[:, 2] >= self.bin_position[2]+.003)
                                   & (points[:, 2] < self.bin_position[2]+height+.012))
            if self.rigid_id>=0:
                # A rigid package may stand above a serving tray's low wall.
                # Its lowest corner must settle on the tray bottom, not float.
                support=self.bin_position[2]+model_config()['bin']['wall_thickness_m']
                within_height=support-.005<=points[:,2].min()<=support+.012
            sock_speed = np.linalg.norm(self.data.cvel[self.sock_ids, 3:], axis=1)
            settled = np.max(sock_speed) < .12
            self.placement_diagnostics = {"inside_fraction": float(inside_fraction),
                "within_height": bool(within_height), "sock_speed_max_m_s": float(np.max(sock_speed)),
                "sock_height_bounds_m": [float(points[:, 2].min()), float(points[:, 2].max())]}
            in_bin = (self.ever_lifted and inside_fraction >= .9 and within_height
                      and not any(pads) and settled)
            self.placement_hold_s = self.placement_hold_s+dt if in_bin else 0.
            self.task_success = self.task_success or self.placement_hold_s >= 1.

    def base_pose(self):
        pos = self.data.body("base_link").xpos
        rot = self.data.body("base_link").xmat.reshape(3, 3)
        return [float(pos[0]), float(pos[1]), math.atan2(rot[1, 0], rot[0, 0])]

    def state(self):
        points = self._sock_points()
        return {"command": self.command.tolist(), "target": self.pending_command.tolist(),
            "measured_sim": joint_to_servo(self.data.qpos[self.joint_q]).tolist(),
            "base_pose": self.base_pose(), "sim_time": float(self.data.time),
            "jaw_center_m": self.jaw_center().tolist(),
            "sock_center_m": points.mean(axis=0).tolist() if len(points) else None,
            "task": {"id": self.task_id, "phase": self.phase, "grasp_contact": all(self.contacts),
                     "finger_contacts": self.contacts, "lifted": self.ever_lifted,
                     "placed": self.task_success, "success": self.task_success,
                     "lift_hold_s": self.lift_hold_s},
            "physics_validated": False, "joint_state_source": "command_estimate",
            "simulation_truth_separate": True, "domain_variation": self.domain_variation, "error": self.last_error,
            "provenance": {"geometry": "checksum_verified_CAD", "dynamics": "estimated",
                "camera": model_config()["camera"]["intrinsics_status"],
                "real_hardware_accepted": False}}

    def render_rgb(self, camera="wrist"):
        cached = self._rgb_cache.get(camera)
        if cached is not None and cached[0] == self.data.time:
            return cached[1].copy()
        if camera == "overview":
            if self.overview_renderer is None:
                self.overview_renderer = mujoco.Renderer(self.model, height=600, width=800)
            self.overview_renderer.update_scene(self.data, camera=camera, scene_option=self.render_options)
            image = self.overview_renderer.render().copy()
            self._rgb_cache[camera] = (self.data.time, image)
            return image.copy()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.render_options)
        image = self.renderer.render().copy()
        self._rgb_cache[camera] = (self.data.time, image)
        return image.copy()

    def render_depth(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="wrist_depth", scene_option=self.render_options)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return depth

    def lidar_scan(self, scanner=0, rays=180):
        if scanner not in (0, 1) or not 8 <= rays <= 720:
            raise ValueError("Неверный сканер или число лучей")
        site = self.data.site(f"laser{scanner}_frame")
        angles = np.linspace(-math.pi, math.pi, rays, endpoint=False)
        matrix = site.xmat.reshape(3, 3)
        output = []
        geom_id = np.zeros(1, np.int32)
        for angle in angles:
            direction = matrix @ np.array([math.cos(angle), math.sin(angle), 0.])
            dist = mujoco.mj_ray(self.model, self.data, site.xpos, direction,
                                np.array([1, 1, 0, 1, 1, 1], np.uint8), 1, -1, geom_id)
            output.append(float(dist) if .03 <= dist <= 24. else None)
        return {"frame": f"laser{scanner}_frame", "ranges": output,
                "angle_min": -math.pi, "angle_increment": 2*math.pi/rays,
                "simulation_only": True}

    def capture_sample(self):
        return {"at": time.time(), "observation": self.command.tolist(),
                "action": self.pending_command.tolist(), "source": "simulation",
                "simulation_time": float(self.data.time)}

    def solve_ik(self, target, seed=None, *, downward=True, gripper=30.):
        target = np.asarray(target, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Три конечных координаты цели обязательны")
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.
        seed = np.asarray(seed if seed is not None else self.command[:5], float)
        def residual(servo):
            q = servo_to_joint([*servo, gripper])
            scratch.qpos[self.joint_q] = q
            # A kinematic candidate must include every mimic joint. Equality
            # constraints are applied by dynamics, not mj_forward.
            for name, multiplier in (("rlink2_Joint", 1), ("llink1_Joint", 1),
                                     ("llink2_Joint", -1), ("rlink3_Joint", -1),
                                     ("llink3_Joint", 1)):
                scratch.qpos[self.model.joint(name).qposadr[0]] = multiplier*q[5]
            mujoco.mj_forward(self.model, scratch)
            position = (self.jaw_center(scratch)-target)*10.
            axis = scratch.body("arm5").xmat.reshape(3, 3)[:, 2]
            orientation = (axis-np.array([0., 0., -1.]))*.65 if downward else np.zeros(3)
            return np.r_[position, orientation, (servo-seed)*.00003]
        fits = []
        for candidate in (seed, [90., 125., 165., 150., 90.], [90., 45., 20., 45., 90.]):
            fit = least_squares(residual, np.clip(candidate, .001, [179.999]*4+[269.999]),
                bounds=([0.]*5, [180.]*4+[270.]), max_nfev=100)
            error = np.linalg.norm(residual(fit.x)[:3])/10.
            fits.append((float(np.linalg.norm(residual(fit.x))), error, fit.x.copy()))
        _, error, angles = min(fits, key=lambda pair: pair[0])
        if error > .008:
            raise ValueError(f"IK: цель недостижима, остаток {error:.3f} м")
        return np.r_[angles, gripper]

    def make_episode_trial(self, output_dir, seed=0, progress=None, policy=None, task_id='sock_basket', scene_profile='baseline'):
        """Attempt real physical pickup/place; failed trials remain failed data.

        Teacher may inspect simulated object truth for curriculum generation.
        A learned actor receives only the declared nine state values and RGB.
        """
        from src.mobile_policy_contract import CONTRACT_SHA256, SAMPLE_CONTRACT
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "frames").mkdir()
        rng = np.random.default_rng(seed)
        from .curriculum import task_spec
        spec = task_spec(task_id)
        rooms = spec['rooms']
        source = [rng.uniform(1.02, 1.10) if rooms else rng.uniform(.28, .36), rng.uniform(-.025,.025), .025 if spec['scene']=='cube' else .015]
        options = dict(task_id=task_id, scene_profile=scene_profile)
        if scene_profile == 'baseline' or spec['scene'] not in ('sock', 'cloth'):
            options.update(sock_position=source, bin_position=[.35, .25, 0.] if rooms else [.44, .23, 0.])
        self.reset(seed=seed, options=options)
        manifest = {"id": folder.name, "source": "simulation", "task": spec['title'], "task_id": task_id,
            "task_spec": spec, "teacher_navigation_source": "simulated_pose_only" if rooms else "local_pose_only",
            "action_alignment": "same_row_pre_action", "physics_validated": False,
            "sample_period_s": PERIOD, "fps": round(1/PERIOD), "seed": int(seed), "success": False,
            "sample_contract_sha256": CONTRACT_SHA256, "sample_contract": SAMPLE_CONTRACT,
            "generator": "policy" if policy is not None else "ground_truth_teacher",
            "model_provenance": model_metadata(), "started_at": time.time()}
        manifest['scene_profile'] = scene_profile if spec['scene'] in ('sock', 'cloth') else 'baseline'
        manifest['domain_variation'] = self.domain_variation
        steps = 0
        handle = (folder / "samples.jsonl").open("w")
        def tick(action):
            nonlocal steps
            self.set_manual(action)
            row = self.capture_sample()
            row["sample_contract_sha256"] = CONTRACT_SHA256
            row["action_alignment"] = "same_row_pre_action"
            row["wrist_rgb"] = f"frames/{steps:06d}.jpg"
            row["phase"] = self.phase
            snapshot = self.state()
            row["simulated_outcome"] = dict(snapshot["task"])
            row["simulator_diagnostics"] = {key: snapshot[key] for key in
                ("jaw_center_m", "sock_center_m", "base_pose")}
            row["simulator_diagnostics"]["placement"] = getattr(self, "placement_diagnostics", {})
            Image.fromarray(self.render_rgb()).save(folder / row["wrist_rgb"], quality=92)
            row["requested_action"] = np.asarray(action, float).tolist()
            self.step(action, applied=policy is not None)
            # The learner's target is the endpoint actually applied after the
            # host's bounded motion profile, paired with pre-action observation.
            row["action"] = self.command.tolist()
            handle.write(json.dumps(row)+"\n")
            steps += 1
            if progress is not None and steps % 5 == 0:
                progress(self.state())
            if self.last_error:
                raise RuntimeError(self.last_error)
        def arm_to(goal, duration=2.):
            start = self.pending_command[:6].copy()
            count = max(round(duration/PERIOD), round(np.max(np.abs(goal-start)/self.velocity_limit)/PERIOD)+10)
            for n in range(count):
                u = (n+1)/count
                blend = u*u*u*(10.-15.*u+6.*u*u)
                tick(np.r_[start+(goal-start)*blend, 0., 0., 0.])
            for _ in range(12):
                tick(np.r_[goal, 0., 0., 0.])
        def drive_to(point, arm, limit=1000, speed=.12):
            arrived = False
            for _ in range(limit):
                pose = np.asarray(self.base_pose())
                error = np.asarray(point)-pose[:2]
                c,s = math.cos(pose[2]),math.sin(pose[2])
                velocity = np.clip(np.array([[c,s],[-s,c]]) @ error*1.5, -speed,speed)
                tick(np.r_[arm,velocity,np.clip(-pose[2],-.3,.3)])
                if np.linalg.norm(error)<.012:
                    arrived = True
                    break
            for _ in range(16):
                tick(np.r_[arm,0.,0.,0.])
            if not arrived:
                raise RuntimeError('Путь через дверь не завершён')
        try:
            if policy is not None:
                policy.reset()
                self.phase = "policy"
                # A complete teacher cycle takes roughly 50 simulated seconds.
                for _ in range(4800 if rooms else 1600):
                    obs = self._observation()
                    action = np.clip(np.asarray(policy(obs["state"], obs["wrist"]), float), LOW, HIGH)
                    tick(action)
                    if self.task_success:
                        break
            else:
                # Inspection is a fixed chassis-relative pose, selected without
                # looking at privileged object coordinates. The camera observes
                # the floor before object-dependent alignment begins.
                self.phase = "inspect_floor"
                base = np.asarray(self.base_pose(), float)
                inspection = self.solve_ik([base[0]+.23, base[1], .15],
                                           self.command[:5], gripper=30., downward=True)
                arm_to(inspection)
                for _ in range(12):
                    tick(np.r_[inspection, 0., 0., 0.])
                manifest["initial_inspection"] = {
                    "pose_source": "fixed_chassis_relative_without_object_truth",
                    "jaw_goal_relative_m": [.23, 0., .15],
                    "phase": "inspect_floor", "before_object_alignment": True,
                    "frame": f"frames/{steps-1:06d}.jpg"}
                self.phase = "align_base"
                if rooms:
                    self.phase = 'navigate_source'
                    drive_to([.38,0.], inspection)
                    drive_to([.82,0.], inspection)
                    self.phase = 'align_base'
                target_base = np.r_[self._sock_points().mean(axis=0)[:2]-[.20, 0.], 0.]
                for _ in range(180):
                    pose = np.array(self.base_pose())
                    error = target_base[:2]-pose[:2]
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    body = np.array([[c, s], [-s, c]]) @ error
                    velocity = np.clip(body*1.7, -.12, .12)
                    tick(np.r_[self.pending_command[:6], velocity, np.clip(-pose[2], -.3, .3)])
                    if np.linalg.norm(error) < .008:
                        break
                for _ in range(16):
                    tick(np.r_[self.pending_command[:6], 0., 0., 0.])
                center = self._sock_points().mean(axis=0)
                self.phase = "approach"
                approach = self.solve_ik([*center[:2], .13])
                arm_to(approach)
                self.phase = "lower"
                lowered = self.solve_ik([*center[:2], max(.024, float(center[2])+.018)], approach[:5])
                arm_to(lowered)
                self.phase = "close"
                grasp_height = max(.006,float(center[2])) if self.rigid_id >= 0 else .017
                closed = self.solve_ik([*center[:2], grasp_height], lowered[:5], gripper=170.)
                arm_to(closed, duration=2.)
                self.phase = "lift"
                diversified = manifest['scene_profile'] == 'sock_variants_v1'
                # A common carry height clears every rim in this curriculum;
                # the actor need not infer a distant basket before pickup.
                lift_height = self.bin_position[2]+.42 if diversified else .22
                lift_goal = [self.base_pose()[0]+.23, self.base_pose()[1], .15] if rooms else [*center[:2], lift_height]
                lifted = self.solve_ik(lift_goal, closed[:5], gripper=170., downward=rooms)
                arm_to(lifted)
                for _ in range(30):
                    tick(np.r_[lifted, 0., 0., 0.])
                if not self.ever_lifted:
                    raise RuntimeError("Контактное удержание при подъёме не подтверждено")
                self.phase = "transport"
                if rooms:
                    self.phase = 'navigate_destination'
                    drive_to([.82,0.], lifted, speed=.04)
                    drive_to([-.02,0.], lifted, speed=.04)
                    self.phase = 'transport'
                # Base carries the held object; the arm never drags cloth at floor height.
                target_base = self.bin_position[:2]-(self.jaw_center()[:2]-np.array(self.base_pose()[:2]))
                if diversified:
                    # Leave the chassis hull outside the front wall. The wrist
                    # can pitch for an elevated rim instead of insisting on an
                    # unreachable vertical-down pose. IK tolerance is unchanged.
                    stand_off = self.bin_half_size[0]+.008+.106+.020
                    target_base = self.bin_position[:2]-[stand_off, 0.]
                for _ in range(200):
                    pose = np.array(self.base_pose())
                    error = target_base-pose[:2]
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    body = np.array([[c, s], [-s, c]]) @ error
                    carrying_speed = .04 if rooms else .1
                    tick(np.r_[lifted, np.clip(body*1.5, -carrying_speed, carrying_speed), np.clip(-pose[2], -.2, .2)])
                    if np.linalg.norm(error) < .01:
                        break
                self.phase = "place"
                release_height = max(.14, self.bin_position[2]+self.bin_height+.035) if diversified else .14
                # For a wide laundry basket, drop inside the near edge instead
                # of demanding that this short arm reach its distant centre.
                drop_xy = self.bin_position[:2].copy()
                if diversified:
                    drop_xy[0] -= max(0., self.bin_half_size[0]-.09)
                placement = self.solve_ik([*drop_xy, release_height], lifted[:5], gripper=170., downward=not diversified)
                arm_to(placement)
                opened = self.solve_ik([*drop_xy, release_height], placement[:5], gripper=30., downward=not diversified)
                arm_to(opened, 2.)
                for _ in range(60):
                    tick(np.r_[opened, 0., 0., 0.])
                # Withdraw the open fingers along their approach direction;
                # cloth can drape on one pad and must release through contact.
                self.phase = "release_retreat"
                center = self.jaw_center()
                retreat = self.solve_ik([center[0]-.035, center[1], center[2]+.07],
                                        opened[:5], gripper=30., downward=False)
                arm_to(retreat, 2.)
                opened = retreat
                for _ in range(160):
                    tick(np.r_[opened, 0., 0., 0.])
                if diversified and not self.task_success:
                    # A fabric loop may drape over one rounded fingertip after
                    # opening. Try a bounded wrist roll; success still requires
                    # real release and settling, with every command recorded.
                    self.phase = 'release_unhook'
                    reference = opened.copy()
                    for roll in (15., -15., 0.):
                        unhook = reference.copy()
                        unhook[4] = np.clip(reference[4]+roll, LOW[4], HIGH[4])
                        arm_to(unhook, 1.)
                        for _ in range(40):
                            tick(np.r_[unhook, 0., 0., 0.])
                        if self.task_success:
                            break
            manifest["success"] = self.task_success
            manifest["outcome"] = self.state()["task"]
            manifest["physics_outcome_diagnostics"] = getattr(self, "placement_diagnostics", {})
        except Exception as exc:
            manifest["error"] = str(exc)
            manifest["outcome"] = self.state()["task"]
            manifest["physics_outcome_diagnostics"] = getattr(self, "placement_diagnostics", {})
        finally:
            handle.close()
            self.pending_command[6:] = 0.
            self.command[6:] = 0.
            self.data.ctrl[self.wheel_ids] = 0.
            manifest.update(finished_at=time.time(), frames=steps, simulated=True,
                            hardware_accepted=False)
            (folder / "episode.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    def validate(self, episodes=3, output_dir=None, policy=None, progress=None):
        if output_dir is None:
            raise ValueError("Укажите папку результатов проверки")
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        results = [self.make_episode_trial(folder / f"heldout-{i:04d}", seed=10000+i,
                   policy=policy, progress=progress) for i in range(episodes)]
        report = {"validation_kind": "closed_loop_policy" if policy else "teacher_physics_diagnostics",
                  "episodes": episodes, "successes": sum(bool(r["success"]) for r in results),
                  "physics_validated": False, "real_hardware_accepted": False,
                  "results": [{k: r.get(k) for k in ("id", "success", "error", "outcome")} for r in results]}
        (folder / "validation.json").write_text(json.dumps(report, indent=2))
        return report

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.overview_renderer is not None:
            self.overview_renderer.close()
            self.overview_renderer = None
