"""Reject duplicate/old acquisition stamps; receipt time does not refresh them."""
import math

class SourceFreshness:
    def __init__(self):
        self.last={}
        self.diagnostics={}

    def accept(self, key, stamp, wall, max_age):
        age=wall-stamp
        reason=None
        if not math.isfinite(stamp) or stamp<=0 or not math.isfinite(age):reason='INVALID_STAMP'
        elif not -.1<=age<max_age:reason='SOURCE_CLOCK_OR_AGE_FAULT'
        elif stamp<=self.last.get(key,0):reason='NONADVANCING_STAMP'
        self.diagnostics[key]=dict(stamp=stamp,age_at_receipt_s=age,reason=reason)
        if reason is not None:return False
        self.last[key]=stamp
        return True
