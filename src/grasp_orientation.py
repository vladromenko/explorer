"""Orient a radial arm toward its target without changing grasp pitch or roll."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def radial_orientation(point, arm_origin, reference_quaternion):
    target = np.asarray(point, dtype=float)
    origin = np.asarray(arm_origin, dtype=float)
    quaternion = np.asarray(reference_quaternion, dtype=float)
    if (target.shape != (3,) or origin.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(np.r_[target, origin, quaternion]).all()
            or abs(np.linalg.norm(quaternion) - 1.) > .001):
        raise ValueError("Invalid target, arm origin or reference grasp orientation")
    delta = target - origin
    if math.hypot(delta[0], delta[1]) < .03:
        raise ValueError("Target too close to the arm yaw axis")
    reference = Rotation.from_quat(quaternion)
    bearing = math.atan2(delta[1], delta[0])
    reference_yaw = reference.as_euler("xyz")[2]
    return (Rotation.from_euler("z", bearing - reference_yaw) * reference).as_quat().tolist()
