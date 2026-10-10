from types import SimpleNamespace
import numpy as np
from rgbd_snapshot import registered_pair, save_snapshot


def samples():
    rgb = (np.full((30, 40, 3), 80, np.uint8), 10., "camera")
    depth = (np.full((30, 40), 1250, np.uint16), 10.01, "camera", .001)
    info = SimpleNamespace(k=[30., 0, 20., 0, 30., 15., 0, 0, 1.], d=[0.]*5,
                           width=40, height=30, header=SimpleNamespace(frame_id="camera"), distortion_model="plumb_bob")
    return rgb, depth, info


def test_latest_pair_is_selected_without_detector_result():
    rgb, depth, info = samples()
    newer = (rgb[0], 10.1, "camera")
    later = (depth[0], 10.11, "camera", .001)
    pair = registered_pair([rgb, newer], [depth, later], info, now=10.2, previous=10.)
    assert pair[0][1] == 10.1
    assert pair[1][1] == 10.11


def test_cache_rejects_stale_unsynchronized_and_unregistered_pairs():
    rgb, depth, info = samples()
    assert registered_pair([rgb], [depth], info, now=11.) is None
    assert registered_pair([rgb], [(depth[0], 10.06, "camera", .001)], info, now=10.2) is None
    assert registered_pair([rgb], [(depth[0], 10.01, "other_camera", .001)], info, now=10.2) is None
    info.width = 41
    assert registered_pair([rgb], [depth], info, now=10.2) is None


def test_cache_persists_depth_units_and_both_actual_timestamps(tmp_path):
    rgb, depth, info = samples()
    path = tmp_path / "snapshot.npz"
    assert save_snapshot(path, (rgb, depth, info)) == 10.
    with np.load(path, allow_pickle=False) as raw:
        assert float(raw["stamp"]) == 10.
        assert float(raw["depth_stamp"]) == 10.01
        assert np.allclose(raw["depth"], 1.25)
        assert len(str(raw["frame_id"])) == 32
        assert len(str(raw["camera_info_version"])) == 64
    assert not path.with_suffix(".tmp").exists()
