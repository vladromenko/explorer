"""Pure startup/liveness policy for owned Nav2 server processes.

Readiness here means fresh ROS inputs and active software services. It never
sets commissioning flags or grants a motion lease.
"""
import math

SENSOR_MAX_AGE = .6


def fresh(ros_age, receipt_age, limit=SENSOR_MAX_AGE):
    return (isinstance(ros_age, (int, float)) and isinstance(receipt_age, (int, float))
            and math.isfinite(ros_age) and math.isfinite(receipt_age)
            and -.15 <= ros_age <= limit and 0 <= receipt_age <= limit)


def power_errors(status, report, now, stop_voltage, sample_timeout=3.):
    """Use the existing power policy's state/latch; never infer its recovery.

    Either current core status or current power-manager output is sufficient.
    If both are fresh, the more restrictive observation wins, including a
    latched CRITICAL state after voltage rises again.
    """
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    if not number(stop_voltage) or not 5 < stop_voltage < 15 or not number(sample_timeout) or sample_timeout <= 0:
        return ['power_config_invalid']
    observations = []
    for item, core in ((status, True), (report, False)):
        if isinstance(item, dict):
            stamp = item.get('at')
            if number(stamp) and 0 <= now-stamp < 2.:
                policy = item.get('power') if core else item
                if isinstance(policy, dict):
                    voltage = item.get('battery') if core else item.get('battery_voltage_v')
                    ages = item.get('sensor_age')
                    age = (ages.get('battery') if isinstance(ages, dict) else None) if core else 0.
                    observations.append((policy, voltage, age, now-stamp))
    if not observations:
        return ['power_stale']
    errors = []
    allowed = ('NORMAL', 'IDLE', 'LOW_POWER', 'PERFORMANCE')
    for policy, voltage, age, record_age in observations:
        state = policy.get('state', 'UNKNOWN')
        if state not in allowed:
            errors.append('power_'+str(state).lower())
        if policy.get('motion_allowed') is False:
            errors.append('power_motion_prohibited')
        if policy.get('charging') is True:
            errors.append('power_charging')
        if not number(voltage) or not 5 <= voltage <= 15:
            errors.append('battery_unknown')
        elif voltage <= stop_voltage:
            errors.append('battery_below_stop')
        if not number(age) or age < 0 or age+record_age >= sample_timeout:
            errors.append('battery_stale')
    return list(dict.fromkeys(errors))


class LaunchHealth:
    def __init__(self, nodes, startup_timeout=90., data_loss_timeout=2., state_loss_timeout=8.):
        self.nodes = tuple(nodes)
        self.startup_timeout = startup_timeout
        self.data_loss_timeout = data_loss_timeout
        self.state_loss_timeout = state_loss_timeout
        self.started = None
        self.active_seen = False
        self.data_bad_since = None
        self.state_bad_since = None

    def launched(self, now):
        self.started = now
        self.active_seen = False
        self.data_bad_since = self.state_bad_since = None

    def evaluate(self, now, input_errors, states):
        if self.started is None:
            return ('wait', list(input_errors)) if input_errors else ('start', [])
        if input_errors:
            if self.data_bad_since is None:
                self.data_bad_since = now
            if now - self.data_bad_since >= self.data_loss_timeout:
                return 'restart', list(input_errors)
        else:
            self.data_bad_since = None
        missing = [name for name in self.nodes if name not in states
                   or states[name][0] != 3 or not 0 <= now-states[name][1] <= 3.]
        if not missing:
            self.active_seen = True
            self.state_bad_since = None
        elif self.active_seen:
            if self.state_bad_since is None:
                self.state_bad_since = now
            if now-self.state_bad_since >= self.state_loss_timeout:
                return 'restart', ['lifecycle_inactive:'+name for name in missing]
        elif now-self.started >= self.startup_timeout:
            return 'restart', ['lifecycle_start_timeout:'+name for name in missing]
        return ('active' if self.active_seen and not missing and not input_errors else 'starting', list(input_errors))
