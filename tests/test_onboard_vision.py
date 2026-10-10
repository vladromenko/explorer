import time
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from onboard_vision import OnboardVision, OpticalTarget, color_objects, deproject, fit_plane, tag_objects


def test_measured_cloud_units_and_plane():
    depth = np.full((120, 160), 1.25)
    k = np.array([[130., 0, 80], [0, 130., 60], [0, 0, 1.]])
    xyz, pixels = deproject(depth, k, np.zeros(5), 4)
    assert np.allclose(xyz[:, 2], 1.25)
    assert len(xyz) == len(pixels)
    plane = fit_plane(xyz)
    assert plane["inlier_fraction"] > .99
    assert plane["is_floor"] is False
    assert abs(plane["coefficients_camera"][3]) == pytest.approx(1.25)
    with pytest.raises(ValueError):
        deproject(np.zeros((20, 20)), k, np.zeros(5))


def test_color_shape_and_apriltag_identity():
    image = np.zeros((300, 400, 3), np.uint8)
    cv2.rectangle(image, (30, 40), (130, 100), (0, 0, 255), -1)
    rows = color_objects(image)
    assert rows[0]["color"] == "red"
    assert rows[0]["shape_2d"] == "quadrilateral"
    assert rows[0]["semantic_identity_verified"] is False
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    generator = getattr(cv2.aruco, "generateImageMarker", None)
    marker = generator(dictionary, 17, 100) if generator else cv2.aruco.drawMarker(dictionary, 17, 100)
    canvas = np.full((160, 160, 3), 255, np.uint8)
    canvas[30:130, 30:130] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    assert tag_objects(canvas)[0]["tag_id"] == 17


def test_optical_target_follows_translation_not_future_or_stale():
    rng = np.random.default_rng(2)
    image = np.zeros((200, 240, 3), np.uint8)
    image[50:130, 60:140] = rng.integers(20, 240, (80, 80, 3), dtype=np.uint8)
    tracker = OpticalTarget(image, [60, 50, 140, 130], 10.)
    shifted = cv2.warpAffine(image, np.array([[1., 0, 4.], [0, 1., -3.]]), (240, 200))
    observation = tracker.update(shifted, 10.12)
    assert observation["center_px"] == pytest.approx([104, 87], abs=.2)
    assert observation["target_id"] == tracker.identifier
    with pytest.raises(ValueError, match="Camera gap"):
        tracker.update(shifted, 11.)
    assert tracker.phase == "lost"


def test_low_texture_roi_rejected():
    with pytest.raises(ValueError, match="texture"):
        OpticalTarget(np.zeros((100, 100, 3), np.uint8), [10, 10, 50, 50], time.time())


def test_cloud_requires_synchronized_depth_not_only_fresh_rgb():
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder) / "data"
        directory.mkdir()
        stamp = time.time()
        sample = dict(rgb=np.full((120, 160, 3), 100, np.uint8), depth=np.full((120, 160), 1.25),
                      k=np.array([[130., 0, 80], [0, 130., 60], [0, 0, 1.]]), d=np.zeros(5),
                      stamp=stamp, depth_stamp=stamp, frame="camera_optical_frame")
        path = directory / "rgbd-snapshot.npz"
        np.savez(path, **sample)
        result = OnboardVision(folder).cloud(80)
        assert result["depth_stamp"] == stamp
        assert result["units"] == "metres"
        assert len(result["xyz"]) <= 80
        sample["depth_stamp"] = stamp - .1
        np.savez(path, **sample)
        with pytest.raises(ValueError, match="not synchronized"):
            OnboardVision(folder).cloud(80)
