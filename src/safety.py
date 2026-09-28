"""Pure deterministic velocity gate, independent of the language model."""
import math


class SafetyGate:
    def __init__(self, config):
        self.config = config
        self.estop = True
        self.mode = "MANUAL"
        self.command = [0., 0., 0.]
        self.output = [0., 0., 0.]
        self.command_at = -1e9
        self.source = None

    def stop(self):
        self.mode = "MANUAL"
        self.estop = True
        self.command_at = -1e9
        self.command = [0., 0., 0.]
        self.output = [0., 0., 0.]

    def observe_controller_link(self, healthy):
        """A lost motor-board link revokes the session; recovery never rearms it.

        This cannot stop a disconnected MCU. It prevents new commands on recovery.
        Joystick connectivity is deliberately not part of this signal.
        """
        if not healthy:
            self.stop()

    def submit(self, values, source, now):
        if source not in ("manual","autonomy"):raise ValueError("Invalid command source")
        if len(values) != 3 or not all(math.isfinite(x) for x in values):
            raise ValueError("Three finite velocities required")
        if source == "autonomy" and self.mode != "AUTONOMOUS":
            return False
        if source == "manual":
            self.mode = "MANUAL"
        self.command = [max(-m, min(m, x)) for x, m in zip(values, self.config["max_velocity"])]
        self.command_at = now
        self.source = source
        return True

    def tick(self, now, dt, sensors_ok, collision=False):
        reason = None
        if self.estop:
            reason = "STOP LATCHED"
        elif not self.config["base_commissioned"] or not self.config["mcu_watchdog_verified"]:
            reason = "BASE COMMISSIONING REQUIRED"
        elif not sensors_ok:
            reason = "SENSOR OR BATTERY FAULT"
        elif collision:
            reason = "OBSTACLE"
        elif now - self.command_at > 0.25:
            reason = "COMMAND EXPIRED"
        if reason:
            self.output = [0., 0., 0.]
            # A fault must not leave a pending command that resumes later.
            if reason not in ("COMMAND EXPIRED",):
                self.command_at = -1e9
            return self.output, reason
        dt = max(0., min(dt, 0.1))
        self.output = [v + max(-a * dt, min(a * dt, target - v))
                       for v, target, a in zip(self.output, self.command, self.config["max_acceleration"])]
        return self.output, "ACTIVE"
