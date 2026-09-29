"""Time-based quintic execution of the complete MoveIt joint path.

Jetson owns interpolation; MCU sends immediate servo targets. Internal servo
target replacement and attainable streaming rate still require bench acceptance.
No degree rounding and no stop/dwell inserted at internal trajectory points.
"""
from dataclasses import dataclass
import bisect
import hashlib
import json
import math
import numpy as np


def derivative(coefficients):
    return coefficients[1:] * np.arange(1, len(coefficients))[:, None]


def bernstein_bounds(coefficients):
    """Convex hull bound on a polynomial over u in [0,1], all joints at once."""
    degree = len(coefficients) - 1
    bernstein = np.zeros_like(coefficients)
    for k in range(degree + 1):
        for i in range(k + 1):
            bernstein[k] += coefficients[i] * math.comb(k, i) / math.comb(degree, i)
    return bernstein.min(axis=0), bernstein.max(axis=0)


def exact_bounds(coefficients):
    """Tight numeric extrema of a power-basis polynomial on [0, 1]."""
    coefficients=np.asarray(coefficients,dtype=float)
    low=np.empty(coefficients.shape[1]);high=np.empty(coefficients.shape[1])
    slope=derivative(coefficients)
    for joint in range(coefficients.shape[1]):
        roots=np.polynomial.polynomial.polyroots(slope[:,joint]) if len(slope)>1 else []
        points=[0.,1.,*(float(r.real) for r in roots if abs(r.imag)<1e-9 and 0<r.real<1)]
        values=[np.polynomial.polynomial.polyval(x,coefficients[:,joint]) for x in points]
        low[joint],high[joint]=min(values),max(values)
    return low,high


def quintic(q0, q1, v0, v1, a0, a1, duration):
    c = np.zeros((6, len(q0)))
    c[0], c[1], c[2] = q0, v0 * duration, a0 * duration**2 / 2
    delta = q1 - c[:3].sum(axis=0)
    velocity = v1 * duration - c[1] - 2 * c[2]
    acceleration = a1 * duration**2 - 2 * c[2]
    c[3] = 10 * delta - 4 * velocity + acceleration / 2
    c[4] = -15 * delta + 7 * velocity - acceleration
    c[5] = 6 * delta - 3 * velocity + acceleration / 2
    return c


@dataclass(frozen=True)
class Limits:
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    def validate(self, count):
        arrays = [np.asarray(a, dtype=float) for a in
                  (self.lower, self.upper, self.velocity, self.acceleration, self.jerk)]
        if any(a.shape != (count,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('finite limits required for every joint')
        if np.any(arrays[0] >= arrays[1]) or any(np.any(a <= 0) for a in arrays[2:]):
            raise ValueError('invalid mechanical/dynamic limits')


class TimedPath:
    def __init__(self, names, times, positions, velocities, accelerations, limits, collision_free):
        self.names = tuple(names)
        q, v, a = (np.asarray(x, dtype=float) for x in (positions, velocities, accelerations))
        t = np.asarray(times, dtype=float)
        count = len(names)
        limits.validate(count)
        if not callable(collision_free):
            raise ValueError('collision validation of interpolated path is mandatory')
        if not 2 <= len(t) <= 2000 or len(set(names)) != count or not count:
            raise ValueError('incomplete trajectory or repeated joint names')
        if q.shape != (len(t), count) or v.shape != q.shape or a.shape != q.shape:
            raise ValueError('MoveIt position, velocity and acceleration arrays must be retained')
        if not all(np.isfinite(x).all() for x in (t, q, v, a)):
            raise ValueError('non-finite trajectory')
        if abs(t[0]) > 1e-9 or np.any(np.diff(t) < .001) or t[-1] > 600:
            raise ValueError('strictly increasing trajectory times starting at zero required')
        if np.any(q < limits.lower) or np.any(q > limits.upper):
            raise ValueError('trajectory exceeds joint soft limits')
        if np.max(np.abs(v[[0, -1]])) > 1e-5 or np.max(np.abs(a[[0, -1]])) > 1e-4:
            raise ValueError('start/finish must be at rest; replanning a moving path needs braking first')
        self.source_sha256 = hashlib.sha256(json.dumps(dict(
            names=list(names), times=t.tolist(), positions=q.tolist(), velocities=v.tolist(),
            accelerations=a.tolist()), sort_keys=True, allow_nan=False).encode()).hexdigest()
        segments, scale = [], 1.0
        for index, dt in enumerate(np.diff(t)):
            c = quintic(q[index], q[index+1], v[index], v[index+1], a[index], a[index+1], dt)
            low, high = bernstein_bounds(c)
            if np.any(low < limits.lower - 1e-9) or np.any(high > limits.upper + 1e-9):
                raise ValueError('spline control hull crosses a soft limit')
            d = c
            for order, maximum in enumerate((limits.velocity, limits.acceleration, limits.jerk), 1):
                d = derivative(d)
                # The Bernstein hull is ideal for collision subdivision but is
                # overly conservative for motion timing. Exact derivative
                # extrema avoid turning a smooth 24 deg/s request into 5 deg/s.
                low, high = exact_bounds(d)
                bound = np.maximum(np.abs(low), np.abs(high)) / dt**order
                scale = max(scale, float(np.max((bound / maximum)**(1 / order))))
            segments.append(c)
        # Uniform time dilation leaves the entire polynomial path unchanged,
        # including intermediate positions and nonzero waypoint velocities.
        self.times = t * scale
        self.coefficients = segments
        self.duration = float(self.times[-1])
        if self.duration > 600:
            raise ValueError('dynamically limited path is too long')
        self.scale = scale
        self.positions = q.copy()
        self.limits = limits
        self.validate_collisions(collision_free)

    def sample(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid monotonic elapsed time')
        index = min(len(self.coefficients)-1, max(0, bisect.bisect_right(self.times, elapsed)-1))
        dt = self.times[index+1]-self.times[index]
        u = min(1., max(0., (elapsed-self.times[index])/dt))
        c = self.coefficients[index]
        result = []
        for order in range(4):
            result.append(sum(c[i] * u**i for i in range(len(c))) / dt**order)
            c = derivative(c)
        return dict(position=result[0], velocity=result[1], acceleration=result[2], jerk=result[3],
                    finished=elapsed >= self.duration)

    def validate_collisions(self, collision_free):
        checks = 0
        previous = self.positions[0]
        for index, c in enumerate(self.coefficients):
            low, high = bernstein_bounds(derivative(c))
            # Limit the maximum joint displacement of each checked interval to
            # 0.25 degrees using a conservative bound, not endpoint difference.
            count = max(1, math.ceil(float(np.max(np.maximum(np.abs(low), np.abs(high)))) / math.radians(.25)))
            checks += count
            if checks > 100000:
                raise ValueError('collision-check budget exceeded')
            for k in range(1, count+1):
                u = k / count
                current = sum(c[i] * u**i for i in range(6))
                if collision_free(previous.tolist(), current.tolist()) is not True:
                    raise ValueError('interpolated MoveIt path is not collision-free')
                previous = current


class Execution:
    """Nonblocking execution state; only fresh feedback can produce reached."""
    def __init__(self, path, send, cancel, feedback, *, period_s, tolerance=.035, settle_s=.3):
        if not .02 <= period_s <= .1:
            raise ValueError('period must come from accepted servo-bus timing')
        self.path, self.send, self.cancel_command, self.feedback = path, send, cancel, feedback
        self.period, self.tolerance, self.settle = period_s, tolerance, settle_s
        self.state, self.reason, self.generation = 'accepted', '', 0
        self.started = self.next_send = self.inside_since = None
        self.last_sent = None

    def tick(self, now):
        if self.state in ('reached', 'cancelled', 'fault'):
            return self.state
        measurement = self.feedback()
        acquired = np.asarray(measurement['acquired_monotonic'], dtype=float)
        actual = np.asarray(measurement['position'], dtype=float)
        valid = np.asarray(measurement['valid'], dtype=bool)
        if (actual.shape != (len(self.path.names),) or acquired.shape != actual.shape or
            valid.shape != actual.shape or not valid.all() or not np.isfinite(actual).all() or
            not np.isfinite(acquired).all() or np.any(now-acquired < 0) or np.any(now-acquired > .25)):
            return self.fail('invalid or stale measured joint state')
        if self.started is None:
            if np.max(np.abs(actual-self.path.positions[0])) > self.tolerance:
                return self.fail('measured start differs from planned start')
            self.started, self.next_send = now, now
        elapsed = now-self.started
        if elapsed < 0:
            return self.fail('monotonic clock reversed')
        if self.last_sent is not None and now-self.last_sent > .15:
            return self.fail('stream deadline missed')
        expected = self.path.sample(elapsed)
        # No catch-up burst: send the current sample once. Old samples are gone.
        if now >= self.next_send:
            self.generation += 1
            self.send(expected['position'].tolist(), now, now+.15, self.generation)
            self.last_sent, self.next_send = now, now+self.period
            self.state = 'sent' if elapsed == 0 else 'executing'
        if expected['finished']:
            inside = np.max(np.abs(actual-self.path.positions[-1])) <= self.tolerance
            self.inside_since = (self.inside_since if self.inside_since is not None else now) if inside else None
            if self.inside_since is not None and now-self.inside_since >= self.settle:
                self.state = 'reached'
            elif elapsed > self.path.duration + 3:
                return self.fail('feedback never reached the final pose')
        return self.state

    def cancel(self):
        self.generation += 1
        self.cancel_command()
        self.state = 'cancelled'
        self.reason = 'profile invalidated; MCU measured hold requested, physical stop is separate'

    def fail(self, reason):
        self.generation += 1
        self.cancel_command()
        self.state, self.reason = 'fault', reason
        return self.state
