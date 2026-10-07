"""Convert a real map-frame pose to the displayed occupancy image."""
import math


def pose_pixel(pose, metadata, scale=1):
    resolution = float(metadata["resolution"])
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Некорректный масштаб карты")
    origin = metadata["origin"]
    yaw = float(metadata.get("origin_yaw", 0))
    dx, dy = pose["x"] - origin[0], pose["y"] - origin[1]
    x = (math.cos(yaw) * dx + math.sin(yaw) * dy) / resolution
    y = (-math.sin(yaw) * dx + math.cos(yaw) * dy) / resolution
    return {"x": x * scale, "y": (metadata["height"] - 1 - y) * scale,
            "yaw": pose["yaw"] - yaw}
