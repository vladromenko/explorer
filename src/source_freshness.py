"""Reject duplicate/old acquisition stamps; receipt time does not refresh them."""
import math

class SourceFreshness:
    # micro-ROS robotio stamps are sampled on the controller clock. Observed
    # host/controller phase error reaches about 160 ms while stamps still
    # advance normally; 250 ms admits that bounded skew, never old samples.
    FUTURE_TOLERANCE_S=.25
    def __init__(self):
        self.last={}
        self.diagnostics={}

    def accept(self, key, stamp, wall, max_age):
        age=wall-stamp
        reason=None
        if not math.isfinite(stamp) or stamp<=0 or not math.isfinite(age):reason='INVALID_STAMP'
        elif not -self.FUTURE_TOLERANCE_S<=age<max_age:reason='SOURCE_CLOCK_OR_AGE_FAULT'
        elif stamp<=self.last.get(key,0):reason='NONADVANCING_STAMP'
        self.diagnostics[key]=dict(stamp=stamp,age_at_receipt_s=age,reason=reason)
        if reason is not None:return False
        self.last[key]=stamp
        return True
