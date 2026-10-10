import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from grasp_orientation import radial_orientation


def test_preserves_pitch_roll_and_uses_arm_axis_instead_of_chassis_origin():
    reference = Rotation.from_euler("xyz", [.0001, math.pi / 3, .035]).as_quat()
    point = [.22637, -.0426, .03475]
    origin = [.033617, -.0000124, .185]
    rotation = Rotation.from_quat(radial_orientation(point, origin, reference)).as_euler("xyz")
    np.testing.assert_allclose(rotation[:2], [.0001, math.pi / 3], atol=1e-10)
    assert rotation[2] == pytest.approx(math.atan2(point[1] - origin[1], point[0] - origin[0]))
    assert abs(rotation[2] - math.atan2(point[1], point[0])) > .025


def test_front_target_has_no_forced_side_heading_and_invalid_geometry_fails():
    reference = Rotation.from_euler("y", 1.).as_quat()
    np.testing.assert_allclose(radial_orientation([.25, 0, .02], [0, 0, .185], reference), reference)
    for point in ([0, 0, .02], [float("nan"), 0, .02]):
        with pytest.raises(ValueError):
            radial_orientation(point, [0, 0, .185], reference)
