"""Bounded heading PI using measured chassis yaw, without scene truth."""
import math
import numpy as np


class HeadingFeedback:
    def __init__(self):self.integral=0.;self.velocity=0.
    def update(self,error,dt):
        if not math.isfinite(error) or not math.isfinite(dt) or not 0<dt<=.1:raise ValueError('Invalid measured heading')
        error=math.atan2(math.sin(error),math.cos(error))
        if error*self.integral<0:self.integral=0.
        self.integral=float(np.clip(self.integral+error*dt,-.25,.25))
        desired=float(np.clip(1.2*error+.4*self.integral,-.15,.15))
        self.velocity+=float(np.clip(desired-self.velocity,-.4*dt,.4*dt))
        return self.velocity
