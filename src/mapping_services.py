"""Start absent mapping components without resetting a healthy map or UART owner."""
import subprocess
import math
import time
from service_diagnostics import inspect_services


UNITS = {"geometry": "explorer-geometry.service", "ekf": "explorer-ekf.service",
         "slam": "explorer-slam.service", "mapview": "explorer-mapview.service"}


class MappingServices:
    def __init__(self, runner=subprocess.run):
        self.runner = runner
        self.cached = None
        self.checked = 0
        self.probe_error = None

    def status(self, force=False):
        if self.cached is None or force or time.monotonic() - self.checked > 10:
            result = inspect_services(UNITS, self.runner)
            self.cached = result["services"]
            self.probe_error = result["services_error"]
            self.checked = time.monotonic()
        return dict(self.cached)

    def recover(self, state):
        age = time.time() - state.get("at", 0)
        velocity = state.get("velocity")
        if not 0 <= age < 2 or state.get("stop_latched") is not True or state.get("mission"):
            raise ValueError("Для восстановления потока включите STOP и завершите текущую миссию")
        if (not isinstance(velocity, list) or len(velocity) != 3 or
                any(type(value) not in (int,float) or not math.isfinite(value) or abs(value) > .001 for value in velocity)):
            raise ValueError("Шасси должно стоять")
        services = self.status(force=True)
        if self.probe_error:
            raise ValueError("Состояние служб недоступно; перезапуск карты не выполнялся: " + self.probe_error)
        started = []
        for name, unit in UNITS.items():
            if services[name] not in ("active", "activating"):
                result = self.runner(["systemctl", "--user", "start", unit],
                                     capture_output=True, text=True, timeout=10)
                if result.returncode:
                    raise ValueError("Не удалось запустить " + name + ": " + result.stderr[-300:])
                started.append(name)
        return {"started": started, "services": self.status(force=True),
                "healthy_map_restarted": False, "motor_commands_sent": False}
