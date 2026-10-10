"""Local Community Ruckig velocity mode; all state is commanded, never measured.

No intermediate-waypoint/cloud API is used. The factory endpoint transport
remains integer degrees; rounding happens only after the continuous state and
short lookahead have been generated. Actual servo interpolation is a separate
physical contract and must be observed on the installed controller.
"""
import numpy as np
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig, Trajectory
from arm_commissioning import HARD_LIMITS


def finite_vector(values, size, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Invalid " + name)
    return result


class VelocityGenerator:
    """Continuous q/v/a with a bounded, replaceable future endpoint."""
    def __init__(self, start, config=None):
        config = config or {}
        self.period = float(config.get("publish_period_s", 0.04))
        self.horizon = float(config.get("lookahead_s", 0.08))
        if not 0.04 <= self.period <= 0.15 or not self.period <= self.horizon <= 0.15:
            raise ValueError("Invalid velocity stream timing")
        self.lower = np.array([pair[0] for pair in HARD_LIMITS], dtype=float)
        self.upper = np.array([pair[1] for pair in HARD_LIMITS], dtype=float)
        q = finite_vector(start, 6, "stream starting position")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Stream start exceeds servo limits")
        self.maximum = finite_vector(config.get("velocity_deg_s", [48, 48, 48, 60, 60, 72]), 6, "velocity limits")
        self.acceleration = finite_vector(config.get("acceleration_deg_s2", [800, 800, 800, 1120, 1120, 1280]), 6, "acceleration limits")
        self.jerk = finite_vector(config.get("jerk_deg_s3", [16000, 16000, 16000, 22400, 22400, 28800]), 6, "jerk limits")
        if any(np.any(values <= 0) for values in (self.maximum, self.acceleration, self.jerk)):
            raise ValueError("Nonpositive stream limits")
        self.input = InputParameter(6)
        self.input.control_interface = ControlInterface.Velocity
        self.input.current_position = q.tolist()
        self.input.current_velocity = [0.0] * 6
        self.input.current_acceleration = [0.0] * 6
        self.input.target_acceleration = [0.0] * 6
        self.input.max_velocity = self.maximum.tolist()
        self.input.max_acceleration = self.acceleration.tolist()
        self.input.max_jerk = self.jerk.tolist()
        self.output = OutputParameter(6)
        self.ruckig = Ruckig(6, self.period)
        self.previous_quantized = np.rint(q).astype(int)
        self.steps = 0

    @property
    def position(self):
        return np.asarray(self.input.current_position, dtype=float)

    def _bounded_velocity(self, requested):
        velocity = np.clip(requested, -self.maximum, self.maximum)
        braking = InputParameter(6)
        for field in ("control_interface", "current_position", "current_velocity", "current_acceleration",
                "max_velocity", "max_acceleration", "max_jerk", "target_acceleration"):
            setattr(braking, field, getattr(self.input, field))
        braking.target_velocity = [0.0] * 6
        trajectory = Trajectory(6)
        result = self.ruckig.calculate(braking, trajectory)
        if result not in (Result.Working, Result.Finished):
            raise ValueError("Unable to calculate bounded braking")
        # Ruckig velocity-mode position_extrema initializes an unused minimum
        # at zero. Sample the actual braking path instead of using that field.
        points = np.asarray([trajectory.at_time(float(t))[0]
            for t in np.linspace(0.0, trajectory.duration, 33)])
        future_margin = np.abs(velocity) * (self.horizon + self.period) + 0.51
        low = points.min(axis=0) - future_margin
        high = points.max(axis=0) + future_margin
        velocity[(velocity < 0.0) & (low <= self.lower)] = 0.0
        velocity[(velocity > 0.0) & (high >= self.upper)] = 0.0
        return velocity

    def step(self, requested):
        requested = finite_vector(requested, 6, "requested joint velocity")
        current_q = list(self.input.current_position)
        current_v = list(self.input.current_velocity)
        current_a = list(self.input.current_acceleration)
        target = self._bounded_velocity(requested)
        self.input.target_velocity = target.tolist()
        result = self.ruckig.update(self.input, self.output)
        if result not in (Result.Working, Result.Finished):
            raise ValueError("Ruckig velocity generation failed")
        q, v, a = self.output.trajectory.at_time(self.output.time + self.horizon - self.period)
        endpoint = finite_vector(q, 6, "stream endpoint")
        current = np.asarray(self.output.new_position)
        if np.any(current < self.lower - 1e-7) or np.any(current > self.upper + 1e-7):
            raise ValueError("Braking path exceeds servo limits")
        if np.any(endpoint < self.lower - 1e-7) or np.any(endpoint > self.upper + 1e-7):
            raise ValueError("Stream lookahead exceeds servo limits")
        quantized = np.rint(endpoint).astype(int)
        # Small fluctuations around half-degrees do not toggle a target.
        hold = np.abs(endpoint - self.previous_quantized) < 0.55
        quantized[hold] = self.previous_quantized[hold]
        if np.any(quantized < self.lower) or np.any(quantized > self.upper):
            raise ValueError("Quantized endpoint exceeds servo limits")
        changed = bool(np.any(quantized != self.previous_quantized))
        self.previous_quantized = quantized
        self.output.pass_to_input(self.input)
        self.steps += 1
        settled = bool(np.max(np.abs(self.output.new_velocity)) < 1e-6
            and np.max(np.abs(self.output.new_acceleration)) < 1e-6
            and np.max(np.abs(target)) < 1e-6)
        return dict(pose=quantized.tolist(), changed=changed, settled=settled,
            runtime_ms=round(self.horizon * 1000), q_estimated_deg=current_q,
            velocity_deg_s=current_v, acceleration_deg_s2=current_a,
            requested_velocity_deg_s=requested.tolist(), limited_velocity_deg_s=target.tolist(),
            q_endpoint_deg=endpoint.tolist(), generation_step=self.steps,
            measured=False, attained=False, state_source="command_trajectory_estimate")


def cartesian_joint_velocity(model, position, xyz_velocity, direct_velocity, maximum):
    """Damped local Jacobian; keeps fractional q and never invokes long IK."""
    q = finite_vector(position, 6, "Cartesian seed")
    xyz = finite_vector(xyz_velocity, 3, "Cartesian velocity")
    direct = finite_vector(direct_velocity, 6, "direct joint velocity")
    bounds = finite_vector(maximum, 6, "joint velocity bounds")
    if not np.any(xyz):
        return np.clip(direct, -bounds, bounds).tolist()
    jacobian = np.zeros((3, 5))
    for index in range(5):
        low, high = HARD_LIMITS[index]
        a, b = q[:5].copy(), q[:5].copy()
        a[index] = max(low, q[index] - 0.2)
        b[index] = min(high, q[index] + 0.2)
        denominator = b[index] - a[index]
        if denominator <= 0.0:
            raise ValueError("No local Cartesian range")
        pa = np.asarray(model.fk(a.tolist(), -0.3)["xyz"])
        pb = np.asarray(model.fk(b.tolist(), -0.3)["xyz"])
        jacobian[:, index] = (pb - pa) / denominator
    # Operator pitch/roll are independent requested components. Position
    # compensation uses the first three joints when pitch is explicitly held.
    active = 3 if abs(direct[3]) > 1e-8 else 4
    matrix = jacobian[:, :active]
    residual = xyz - jacobian @ direct[:5]
    damping = 0.0001
    correction = matrix.T @ np.linalg.solve(matrix @ matrix.T + np.eye(3) * damping ** 2, residual)
    result = direct.copy()
    result[:active] += correction
    scale = max(1.0, float(np.max(np.abs(result) / bounds)))
    result /= scale
    predicted = jacobian @ result[:5]
    if np.dot(predicted, xyz) <= 0.0 or np.linalg.norm(predicted) < 0.1 * np.linalg.norm(xyz):
        raise ValueError("Cartesian direction is singular or unreachable; use joint mode")
    return result.tolist()
