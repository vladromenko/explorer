import cv2
import numpy as np
import pytest
from grounded_targets import GroundedTargets
from grounded_targets import StationaryDepthTarget


def frame(stamp=10., shift=0):
    rng = np.random.default_rng(9)
    rgb = np.full((180, 240, 3), 120, np.uint8)
    rgb[45:110, 65+shift:130+shift] = rng.integers(20, 240, (65, 65, 3), dtype=np.uint8)
    depth = np.full((180, 240), .5)
    depth[45:110, 65+shift:130+shift] = .47
    return {"rgb": rgb, "depth": depth, "k": np.array([[200.,0,120],[0,200.,90],[0,0,1]]),
        "d": np.zeros(5), "stamp": stamp, "frame": "optical"}


def recognized():
    return [{"label": "black sock", "requested_label": "носок", "confidence": .6,
        "bbox": [60,40,135,115], "position": {"z": .47}}]


def test_fresh_association_preserves_query_and_measures_new_point_without_motion():
    tracker = GroundedTargets()
    tracker.install("a"*32, frame(), frame(10.1), recognized(), now=10.1)
    output = tracker.update(frame(10.2, 3), now=10.2)
    assert output["phase"] == "tracking"
    target = output["objects"][0]
    assert target["label"] == "носок" and target["model_label"] == "black sock"
    assert target["bbox"] == pytest.approx([63,40,138,115], abs=.3)
    assert target["position"]["z"] == pytest.approx(.47)
    assert target["image_stamp"] == 10.2 and target["recognition_stamp"] == 10.
    assert target["identity_verified"] is False and target["motion_authorized"] is False


def test_gap_and_changed_calibration_do_not_reuse_old_label_on_new_object():
    for mode in ("gap", "calibration", "occlusion", "timeout"):
        tracker = GroundedTargets()
        tracker.install("b"*32, frame(), frame(10.1), recognized(), now=10.1)
        sample = frame(10.2)
        now = 10.2
        if mode == "gap":
            sample["stamp"] = 11.
        if mode == "calibration":
            sample["k"][0,0] = 220
        if mode == "occlusion":
            sample["rgb"][:] = 120
        if mode == "timeout":
            now = 131.
        output = tracker.update(sample, now=now)
        assert output["objects"] == [], mode
        assert output["phase"] in ("lost", "expired")
        assert tracker.update(frame(11.1), now=131.1)["objects"] == []


def test_reacquisition_rejects_moved_or_unlocalized_hypothesis():
    tracker = GroundedTargets()
    tracker.install("c"*32, frame(), frame(10.1, 20), recognized(), now=10.1)
    assert not tracker.targets

    item = recognized()[0]
    item["position"] = None
    tracker.install("d"*32, frame(), frame(10.1), [item], now=10.1)
    assert not tracker.targets


def test_depth_reobservation_never_follows_a_moving_observer_or_survives_occlusion():
    original=frame()
    original["observer_key"]=("boot",(90,90,30,0))
    target=StationaryDepthTarget(original,recognized()[0]["bbox"])
    current=frame(10.1)
    current["observer_key"]=original["observer_key"]
    observation=target.update(current)
    assert observation["point_camera"][2] == pytest.approx(.47)
    assert observation["association_fraction"] is None
    assert observation["object_position_uncertainty_m"] is None
    repeated=frame(10.8)
    repeated["observer_key"]=original["observer_key"]
    assert target.update(repeated)["point_camera"][2] == pytest.approx(.47)
    with pytest.raises(ValueError,match="frame gap"):
        target.update(dict(repeated,stamp=13.))
    current=frame(10.9)
    current["observer_key"]=("boot",(90,91,30,0))
    with pytest.raises(ValueError,match="Observer moved"):
        target.update(current)
    current["observer_key"]=original["observer_key"]
    current["depth"][:]=.5
    with pytest.raises(ValueError,match="separated object"):
        target.update(current)
