"""Bounded live scan display with source timestamps and the actual sensor TF."""
import math
import threading
import time


class LidarView:
    def __init__(self, transform, source_clock=time.time, display_limit=360):
        self.transform = transform
        self.source_clock = source_clock
        self.display_limit = display_limit
        self.lock = threading.Lock()
        self.scans = {}

    def receive(self, name, message):
        now = time.monotonic()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        points = []
        for index, value in enumerate(message.ranges):
            distance = float(value)
            if math.isfinite(distance) and message.range_min < distance < message.range_max:
                angle = message.angle_min + index * message.angle_increment
                points.append([distance * math.cos(angle), distance * math.sin(angle)])
        stride = max(1, math.ceil(len(points) / self.display_limit))
        with self.lock:
            previous = self.scans.get(name)
            period = now - previous["received"] if previous else None
            rate = 1 / period if period and period > 0 else None
            if rate and previous and previous.get("rate_hz"):
                rate = .8 * previous["rate_hz"] + .2 * rate
            self.scans[name] = {"name": name, "frame": message.header.frame_id,
                "stamp": stamp, "received": now, "sequence": previous["sequence"] + 1 if previous else 1,
                "range_min_m": float(message.range_min), "range_max_m": float(message.range_max),
                "valid_returns": len(points), "total_returns": len(message.ranges),
                "rate_hz": rate, "sensor_points": points[::stride], "display_stride": stride}

    def status(self):
        now = time.monotonic()
        source_now = self.source_clock()
        with self.lock:
            snapshots = {key: dict(value) for key, value in self.scans.items()}
        sensors = []
        for name in ("scan0", "scan1"):
            record = snapshots.get(name)
            if record is None:
                sensors.append({"name": name, "fresh": False, "reason": "Нет сканов", "points_m": []})
            else:
                receipt_age = now - record["received"]
                source_age = source_now - record["stamp"]
                fresh = 0 <= receipt_age < .7 and -.1 <= source_age < .7
                result = {key: value for key, value in record.items() if key not in ("received", "sensor_points")}
                result.update(receipt_age_s=round(receipt_age, 3), source_age_s=round(source_age, 3),
                              fresh=fresh, reason=None if fresh else "Скан устарел", points_m=[])
                try:
                    translation, quaternion = self.transform(record["frame"])
                    tx, ty, tz = translation
                    if not all(math.isfinite(value) for value in translation):
                        raise ValueError("Некорректное смещение TF лидара")
                    x, y, z, w = quaternion
                    norm = math.sqrt(x*x + y*y + z*z + w*w)
                    if not math.isfinite(norm) or norm < 1e-9:
                        raise ValueError("Некорректный TF лидара")
                    x, y, z, w = (value / norm for value in (x, y, z, w))
                    r00, r01 = 1-2*(y*y+z*z), 2*(x*y-z*w)
                    r10, r11 = 2*(x*y+z*w), 1-2*(x*x+z*z)
                    result["points_m"] = [[round(tx+r00*px+r01*py, 4), round(ty+r10*px+r11*py, 4)]
                                           for px, py in record["sensor_points"]]
                    result["sensor_origin_m"] = [tx, ty, tz]
                    result["transform_valid"] = True
                except (OSError, ValueError, TypeError) as exc:
                    result.update(transform_valid=False, reason=str(exc))
                sensors.append(result)
        return {"at": source_now, "frame": "base_footprint", "sensors": sensors,
                "display_only": True, "mask_applied": False,
                "all_fresh": all(sensor.get("fresh") and sensor.get("transform_valid") for sensor in sensors)}
