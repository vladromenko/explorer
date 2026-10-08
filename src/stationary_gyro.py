"""Zero angular rate update, conditional on fresh wheel and command evidence.

Assumes the chassis is resting on its wheels. Carrying/repositioning the robot
requires re-localization; this is not an inertial navigation system.
"""
import math

class StationaryGyro:
    def __init__(self,bias):
        self.bias=bias;self.wheel_at=-1e9;self.command_at=-1e9;self.last_motion=-1e9
        self.started=None;self.wheel_zero=False;self.command_zero=False

    def wheel(self,velocity,now):
        self.wheel_at=now
        self.wheel_zero=all(math.isfinite(v) and abs(v)<.002 for v in velocity)
        if not self.wheel_zero:self.last_motion=now

    def command(self,velocity,now):
        self.command_at=now
        self.command_zero=all(math.isfinite(v) and abs(v)<1e-8 for v in velocity)
        if not self.command_zero:self.last_motion=now

    def correct(self,gyro,acceleration,now):
        if not all(math.isfinite(v) for v in (*gyro,*acceleration)):
            self.started=None;return None,False
        # The mobile base cannot turn at 5 rad/s under its accepted speed
        # limits. Reject corrupt IMU frames rather than feeding an impossible
        # angular impulse into the heading filter. Raw IMU stays untouched.
        if max(abs(v) for v in gyro)>5. or math.sqrt(sum(v*v for v in acceleration))>40.:
            self.started=None;return None,False
        still=(self.wheel_zero and self.command_zero and now-self.wheel_at<.25 and
               now-self.command_at<.35 and now-self.last_motion>1. and
               max(abs(gyro[0]),abs(gyro[1]),abs(gyro[2]-self.bias))<.025 and
               9.2<math.sqrt(sum(v*v for v in acceleration))<10.4)
        if not still:self.started=None
        elif self.started is None:self.started=now
        qualified=still and now-self.started>1.
        if qualified:
            self.bias+=.002*(gyro[2]-self.bias)
            return 0.,True
        return gyro[2]-self.bias,False

    def controller(self,status,now):
        """Fresh sole-owner command state supplements the 1 Hz factory zero packet.

        Never keep stationarity alive from an expired snapshot or desired goal.
        """
        stamp=status.get("monotonic")
        velocity=status.get("velocity")
        if (type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<=now-stamp<.3 or
                not isinstance(velocity,list) or len(velocity)!=3 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in velocity)):
            return False
        self.command(velocity,now)
        return True
