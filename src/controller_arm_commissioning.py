"""Short, arm-only commissioning authority without accepting robot hardware.

The UART owner validates every request. This is not a navigation permit and
cannot authorize base motion, a changed calibration, or stale measurements.
"""
import hashlib
import json
import math
import time
from pathlib import Path

OPERATIONS = frozenset(('CLEAR', 'OPEN', 'CALIBRATION', 'ARM_ENABLE', 'RECOVERY_ENABLE',
                        'ARM', 'ARM_RECOVER', 'ARM_CANCEL', 'CANCEL', 'HOLD'))
MAX_PERMIT_NS = 60_000_000_000


def require_arm_test_power(root):
    """Read saved power telemetry; this neither clears a latch nor enables motion."""
    status = json.loads((Path(root)/'data/status.json').read_text())
    battery = status.get('battery')
    power = status.get('power', {})
    if (not 0 <= time.time()-status.get('at', 0) < 1 or
        power.get('motion_allowed') is not True or power.get('charging') is True or
        type(battery) not in (int, float) or not math.isfinite(battery) or battery < 11.0 or
        not 0 <= status.get('sensor_age', {}).get('battery', float('inf')) < 2):
        raise ValueError('arm commissioning requires fresh adequate power: state='+
                         str(power.get('state'))+', voltage='+str(battery))


def arm_commissioning_allowed(root, profile, identity, state, samples, calibration, request, now):
    path = Path(root)/'data/controller-arm-permit.json'
    if not path.exists():
        return False
    permit = json.loads(path.read_text())
    if request.get('source_id') != permit.get('source_id'):
        return False
    if request.get('operation') not in OPERATIONS:
        raise ValueError('arm commissioning does not authorize this operation')
    issued, expires = permit.get('issued_ns'), permit.get('expires_ns')
    if (type(issued) is not int or type(expires) is not int or
        not issued <= now < expires or not 0 < expires-issued <= MAX_PERMIT_NS):
        raise ValueError('arm commissioning permit expired')
    if not identity or any(permit.get(key) != identity.get(key) for key in ('boot', 'uid', 'source_sha256')):
        raise ValueError('arm commissioning board or boot changed')
    digest = hashlib.sha256((Path(root)/'config/controller-calibration.json').read_bytes()).hexdigest()
    if (digest != profile.get('calibration_sha256') or digest != permit.get('calibration_sha256') or
        identity.get('source_sha256') != profile.get('firmware_source_sha256')):
        raise ValueError('arm commissioning calibration or image changed')
    wheels = state.get('wheels', [])
    if (len(wheels) != 4 or any(w.get('pwm') != 0 or w.get('target_rad_s') != 0 or
        not isinstance(w.get('measured_rad_s'), (int, float)) or
        not math.isfinite(w['measured_rad_s']) or abs(w['measured_rad_s']) > .1 for w in wheels)):
        raise ValueError('arm commissioning requires a stationary base')
    # Cancellation and session setup must remain possible without arm samples.
    if request['operation'] in ('ARM', 'ARM_RECOVER', 'ARM_ENABLE', 'RECOVERY_ENABLE'):
        if len(samples) != 6:
            raise ValueError('arm commissioning requires all six measured joints')
        for index, sample in enumerate(samples):
            acquired = sample.get('acquired_monotonic_ns')
            if (sample.get('joint') != index+1 or sample.get('error') != 0 or
                sample.get('device_error') != 0 or not sample.get('raw_valid') or
                not sample.get('position_valid') or type(acquired) is not int or
                not 0 <= now-acquired < 250_000_000):
                raise ValueError('arm commissioning measurement invalid: '+str(index+1))
        if request['operation'] in ('ARM', 'ARM_RECOVER'):
            require_arm_test_power(root)
            positions = request.get('position_rad', [])
            if len(positions) != 6:
                raise ValueError('six arm targets required')
            for sample, cal, q in zip(samples, calibration, positions):
                measured = cal.observe(sample['raw_ticks'])['position_rad']
                if not math.isfinite(q) or abs(q-measured) > .03:
                    raise ValueError('arm commissioning target exceeds measured excursion')
                if request['operation'] == 'ARM':
                    cal.target_raw(q)
                else:
                    coordinate = (q-cal.radians_at_raw_zero)/cal.radians_per_tick
                    raw = int(coordinate+.5)
                    in_work = (cal.command_min <= raw <= cal.command_max and cal.lower <= q <= cal.upper)
                    if not in_work and not (cal.recovery_min and cal.recovery_min <= raw <= cal.recovery_max):
                        raise ValueError('arm commissioning recovery corridor not accepted')
    return True
