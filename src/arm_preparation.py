"""Confirm the base STOP request before supervised arm preparation."""
import json
import math
from pathlib import Path
import time

from arm_commissioning import stationary_status


def stop_base_before_prepare(root, stop, timeout=3):
    request=stop()
    deadline=time.monotonic()+timeout
    steady_since=None
    first_stamp=None
    while time.monotonic()<deadline:
        try:
            state=json.loads((Path(root)/'data/status.json').read_text())
            ack=state.get('last_request',{})
            if ack.get('id')!=request['id'] or ack.get('ok') is not True:
                raise ValueError('STOP not acknowledged')
            stationary_status(state,time.time())
            speeds=state.get('odom_velocity')
            stamp=state.get('odom_received_monotonic')
            now=time.monotonic()
            if not isinstance(speeds,list) or len(speeds)!=3 or any(
                not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>limit
                for v,limit in zip(speeds,(.005,.005,.02))):
                raise ValueError('Base has not settled')
            # Core persists status every 0.5 s; account for that snapshot latency.
            # stationary_status also checks the age at persistence, and a second
            # distinct odometry receipt is required before returning.
            if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not 0<=now-stamp<.9:
                raise ValueError('Odometry is stale')
            if steady_since is None:
                steady_since=now
                first_stamp=stamp
            if now-steady_since>=.3 and stamp>first_stamp:
                return state
        except (OSError,ValueError,KeyError,TypeError):
            steady_since=None
            first_stamp=None
        time.sleep(.02)
    raise ValueError('Нет подтверждённой остановки шасси и свежей нулевой скорости по одометрии. Команда подготовки руки не отправлена')
