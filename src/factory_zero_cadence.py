"""Reduce redundant zeros only after fresh odometry confirms stationary chassis.

A host accommodation for the known 1.0 shared battery/command counter. It does
not alter moving commands, clear STOP, suppress urgent stop, or prove recovery.
The independent host heartbeat and MCU expiry remain active.
"""
class StationaryZeroCadence:
    def __init__(self):self.last_sent=-float('inf')
    def publish_due(self,velocity,stationary_since,now):
        established=(stationary_since is not None and 0<=now-stationary_since and now-stationary_since>=.4)
        redundant_zero=established and len(velocity)==3 and all(v==0 for v in velocity)
        due=not redundant_zero or now-self.last_sent>=1.0
        if due:self.last_sent=now
        return due
