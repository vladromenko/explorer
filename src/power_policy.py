"""Pure local power policy. Time arguments are monotonic; no hardware effects."""
import math


class PowerPolicy:
    def __init__(self, config):
        self.c = config['policy']
        self.filtered = None
        self.last_sample = None
        self.last_activity = 0.
        self.low_since = None
        self.critical_since = None
        self.low = False
        self.critical = False
        self.state = 'UNKNOWN'

    def sample(self, voltage, now):
        if type(voltage) not in (int, float) or not math.isfinite(voltage) or not 5 <= voltage <= 15:
            self.last_sample = None
            return
        dt = 0. if self.last_sample is None else max(0., now-self.last_sample)
        if self.filtered is None or self.last_sample is None:
            self.filtered = voltage
        else:
            alpha = 1.-math.exp(-dt/self.c['filter_tau_s'])
            self.filtered += alpha*(voltage-self.filtered)
        self.last_sample = now
        if voltage <= self.c['critical_v']:
            if self.critical_since is None:self.critical_since = now
            if now-self.critical_since >= self.c['critical_delay_s']:self.critical = True
        else:self.critical_since = None
        if self.filtered <= self.c['low_v']:
            if self.low_since is None:self.low_since = now
            if now-self.low_since >= self.c['low_delay_s']:self.low = True
        else:
            self.low_since = None
            if self.filtered >= self.c['low_v']+self.c['recovery_hysteresis_v']:self.low = False

    def evaluate(self, now, active=False, performance=False, charging=None):
        if active:self.last_activity = now
        fresh = self.last_sample is not None and 0 <= now-self.last_sample < self.c['sample_timeout_s']
        if self.critical:state = 'CRITICAL'
        elif charging is True:state = 'CHARGING'
        elif not fresh:state = 'UNKNOWN'
        elif self.low:state = 'LOW_POWER'
        elif performance:state = 'PERFORMANCE'
        elif now-self.last_activity >= self.c['idle_timeout_s']:state = 'IDLE'
        else:state = 'NORMAL'
        self.state = state
        allowed = fresh and state not in ('CRITICAL', 'CHARGING', 'UNKNOWN')
        return dict(state=state, motion_allowed=allowed,
                    speed_scale=self.c['low_speed_scale'] if state == 'LOW_POWER' else 1. if allowed else 0.,
                    warning=fresh and self.filtered <= self.c['warning_v'],
                    filtered_voltage=self.filtered if fresh else None,
                    charging=charging, shutdown_required=self.critical)
