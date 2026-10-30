"""Bounded arm segments for keyboard and gamepad teleoperation.

The historical class name is retained for the manual-control backend, but this
module contains no recording, dataset, training or autonomous execution code.
"""
import json
import math
from pathlib import Path
import threading
import time

from arm_commissioning import coordinated_status


class TeachingController:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.move = None
        self.stop_revision = lambda: 0
        self.measured_reference = None
        self.arm_reference = None

    def pose(self):
        if self.arm_reference is not None:
            return self.arm_reference()['servo_deg']
        state = json.loads((self.root / 'data/status.json').read_text())
        coordinated_status(state, time.time())
        if self.measured_reference is not None:
            return self.measured_reference()['servo_deg']
        arm = json.loads((self.root / 'data/arm-state.json').read_text())
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        streaming = (arm.get('phase') == 'command_in_progress'
                     and arm.get('motion_profile') == 'ruckig_community_velocity'
                     and time.time() - arm.get('updated_at', 0) < .5)
        if arm.get('boot_id') != boot or (arm.get('phase') != 'command_elapsed_observation_required' and not streaming):
            raise ValueError('Исходное положение руки не подтверждено в текущем сеансе')
        fault = self.root / 'data/arm-telemetry-fault.json'
        if fault.exists() and json.loads(fault.read_text()).get('at', 0) > arm['at']:
            raise ValueError('После потери связи заново подготовьте руку')
        return arm['servo_deg']

    def teleop(self, model, xyz_delta, joint_delta, observing, deadline=None,
               precision=False, arm_mode='cartesian', arm_speed='normal'):
        if observing is not True:
            raise ValueError('Подтвердите присутствие рядом с роботом')
        if arm_mode not in ('cartesian', 'joint') or arm_speed not in ('normal', 'fast'):
            raise ValueError('Неверный режим ручного управления')
        if len(xyz_delta) != 3 or len(joint_delta) != 6:
            raise ValueError('Неверный вектор ручного управления')
        if not all(math.isfinite(float(value)) for value in xyz_delta) or any(type(value) is not int for value in joint_delta):
            raise ValueError('Неверный шаг ручного управления')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Предыдущий сегмент руки ещё выполняется')
        try:
            if self.move is None:
                raise ValueError('Контроллер руки не готов')
            pose = self.pose()
            goal = list(pose)
            cartesian_error = None
            magnitude = math.sqrt(sum(float(value) ** 2 for value in xyz_delta))
            if arm_mode == 'cartesian' and magnitude > 0:
                from cartesian_jog import propose_delta
                try:
                    goal = propose_delta(model, pose, [float(value) for value in xyz_delta], maximum_distance=.010)['goal_deg']
                except ValueError as exc:
                    cartesian_error = str(exc)
            indices = range(6) if arm_mode == 'joint' else (5,)
            for index in indices:
                goal[index] = max(pose[index] - 10, min(pose[index] + 10, goal[index] + joint_delta[index]))
            if goal == pose:
                raise ValueError('Нет исполнимого движения руки')
            result = self.move(
                pose, goal, deadline=deadline, expected_stop_revision=self.stop_revision(),
                source='coordinated_operator',
                speed='precision' if precision else 'teleop_fast' if arm_speed == 'fast' else 'teleop',
            )
            if cartesian_error:
                result['cartesian_rejected'] = cartesian_error
            return result
        finally:
            self.lock.release()
