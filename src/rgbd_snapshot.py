"""Fresh registered RGB-D cache independent of object inference latency."""
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np


def registered_pair(rgbs, depths, info, now=None, previous=0):
    now = time.time() if now is None else now
    if info is None or info.k[0] <= 0 or info.k[4] <= 0:
        return None
    for rgb in reversed(rgbs):
        image, stamp, frame = rgb
        if stamp <= previous:
            break
        if 0 <= now-stamp <= .5:
            candidates = [depth for depth in depths if abs(stamp-depth[1]) <= .04 and depth[2] == frame
                          and depth[0].shape == image.shape[:2] and 0 <= now-depth[1] <= .5]
            if candidates and info.width == image.shape[1] and info.height == image.shape[0] and info.header.frame_id == frame:
                return rgb, min(candidates, key=lambda depth: abs(stamp-depth[1])), info
    return None


def save_snapshot(path, pair):
    rgb, depth, info = pair
    image, stamp, frame = rgb
    signature = json.dumps({"k": list(info.k), "d": list(info.d), "width": info.width,
                            "height": info.height, "frame": info.header.frame_id,
                            "distortion_model": info.distortion_model}, sort_keys=True)
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, rgb=image, depth=depth[0].astype(np.float32)*depth[3],
                            k=np.array(info.k).reshape(3, 3), d=np.array(info.d), stamp=stamp,
                            depth_stamp=depth[1], frame=frame, frame_id=uuid.uuid4().hex,
                            camera_info_version=hashlib.sha256(signature.encode()).hexdigest())
    temporary.replace(path)
    return stamp
