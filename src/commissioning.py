"""A single supervised, low-speed commissioning pulse; not a driving mode."""
import math

class Pulse:
    def __init__(self, velocity, duration, now):
        limits=[.05,.05,.18]
        if len(velocity)!=3 or not all(math.isfinite(v) and abs(v)<=m for v,m in zip(velocity,limits)):
            raise ValueError('Probe velocity exceeds commissioning limits')
        if sum(v!=0 for v in velocity)!=1:
            raise ValueError('Commission one axis at a time')
        if not math.isfinite(duration) or not .1<=duration<=.7:
            raise ValueError('Probe duration must be 0.1..0.7 seconds')
        self.target=list(velocity)
        self.output=[0.,0.,0.]
        self.deadline=now+duration
        self.last_lease=now
        self.finished=False

    def tick(self, now, dt, estop, sensors_ok, collision):
        reason=None
        if estop:reason='STOP LATCHED'
        elif not sensors_ok:reason='SENSOR OR BATTERY FAULT'
        elif collision:reason='OBSTACLE'
        elif now>=self.deadline:reason='PROBE COMPLETE'
        elif now-self.last_lease>.15:reason='PROBE LEASE EXPIRED'
        if self.finished or reason:
            self.finished=True
            self.output=[0.,0.,0.]
            return self.output,reason or 'PROBE COMPLETE'
        dt=max(0.,min(dt,.04))
        self.output=[v+max(-a*dt,min(a*dt,t-v)) for v,t,a in zip(self.output,self.target,[.25,.25,.6])]
        return self.output,'SUPERVISED COMMISSIONING'
