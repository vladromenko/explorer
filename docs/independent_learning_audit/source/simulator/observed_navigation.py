"""Bounded local translation feedback using own odometry, never object truth."""
import math
import numpy as np


class TranslationFeedback:
    def __init__(self,*,maximum_speed=.06,maximum_acceleration=.10):
        if not 0.<maximum_speed<=.06 or not 0.<maximum_acceleration<=.10:
            raise ValueError('Invalid local motion limits')
        self.maximum_speed=maximum_speed;self.maximum_acceleration=maximum_acceleration
        self.integral=np.zeros(2);self.velocity=np.zeros(2)

    def update(self,error_world,yaw,dt):
        error=np.asarray(error_world,float)
        if error.shape!=(2,) or not np.isfinite(np.r_[error,yaw,dt]).all() or not 0.<dt<=.1:
            raise ValueError('Invalid odometry feedback')
        # Limited integral overcomes small persistent rolling resistance. A
        # reversal discards accumulated drive rather than accelerating past it.
        self.integral[error*self.integral<0.]=0.
        self.integral=np.clip(self.integral+error*dt,-.04,.04)
        desired=error*1.2+self.integral*.4
        norm=float(np.linalg.norm(desired))
        if norm>self.maximum_speed:desired*=self.maximum_speed/norm
        change=desired-self.velocity;norm=float(np.linalg.norm(change))
        maximum=self.maximum_acceleration*dt
        if norm>maximum:change*=maximum/norm
        self.velocity+=change
        c,s=math.cos(yaw),math.sin(yaw)
        return np.array([[c,s],[-s,c]])@self.velocity
