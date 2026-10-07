"""Separate continuous SLAM exploration from localization on a restored map."""
import math

BASE_FLAGS=("base_commissioned","mcu_watchdog_verified","lidar_tf_validated")


def scope_blockers(config,scope,health=None,now=None):
    if scope not in ("localized","mapping"):return ["invalid_navigation_scope"]
    reasons=[name for name in BASE_FLAGS if config.get(name) is not True]
    if scope=="localized" and config.get("localization_verified") is not True:
        reasons.append("localization_verified")
    if scope=="mapping":
        for name in ("navigation","planning"):
            record=(health or {}).get(name,{})
            if not isinstance(record,dict):record={}
            stamp=record.get("at")
            if (type(stamp) not in (int,float) or not math.isfinite(stamp) or now is None or
                    not 0<=now-stamp<3 or record.get("stage")!="active" or record.get("inputs_ready") is not True):
                reasons.append(name+"_runtime_unavailable")
    return reasons


def bound_mapping_velocity(values):
    if len(values)!=3 or any(type(value) not in (int,float) or not math.isfinite(value) for value in values):
        raise ValueError("Three finite navigation velocities required")
    speed=math.hypot(values[0],values[1]);scale=min(1.,.10/speed) if speed else 1.
    return [values[0]*scale,values[1]*scale,max(-.25,min(.25,values[2]))]
