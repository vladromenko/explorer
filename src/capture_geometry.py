"""Validate measured arm samples bracketing a camera exposure, without actuation."""
import math
import numpy as np


def stationary_exposure(before, after, exposure_monotonic):
    if before.get('measured') is not True or after.get('measured') is not True:
        raise ValueError('Camera geometry requires measured joints')
    if before['boot_id'] != after['boot_id'] or before['session'] != after['session']:
        raise ValueError('Controller restarted or session changed during exposure')
    a, b = np.asarray(before['position_rad']), np.asarray(after['position_rad'])
    if a.shape != (6,) or b.shape != (6,) or not np.isfinite([a,b]).all():
        raise ValueError('Incomplete joint measurements')
    first, last = np.asarray(before['acquired_monotonic']), np.asarray(after['acquired_monotonic'])
    if first.shape != (6,) or last.shape != (6,) or not np.isfinite([first,last]).all():
        raise ValueError('Missing acquisition times')
    if (not math.isfinite(exposure_monotonic) or np.any(first > exposure_monotonic) or
        np.any(last < exposure_monotonic) or np.any(last-first > .35)):
        raise ValueError('Measurements do not bracket the exposure')
    if np.max(np.abs(a-b)) > math.radians(.15):
        raise ValueError('Arm moved during stationary capture')
    return dict(servo_deg=after['servo_deg'],position_rad=b.tolist(),measured_joint_positions=True,
                controller_boot_id=after['boot_id'],controller_session=after['session'],
                maximum_joint_change_rad=float(np.max(np.abs(a-b))),
                exposure_monotonic=exposure_monotonic)
