"""Approximate voltage reserve, not a coulomb-counted state of charge.

The stock ROSMASTER pack is 3S (12.6 V fully charged). 10.8 V is Explorer's
existing operational stop threshold. No current or charger telemetry exists.
The linear estimate spans the usable voltage interval; load and charging bias it.
"""
import math

EMPTY_V = 10.8
FULL_V = 12.6


def battery_percent(voltage):
    if type(voltage) not in (int, float) or not math.isfinite(voltage):
        return None
    if not 5 <= voltage <= 15:
        return None
    return round(max(0., min(100., 100.*(voltage-EMPTY_V)/(FULL_V-EMPTY_V))))


def battery_summary(voltage, sample_age):
    fresh = (type(sample_age) in (int, float) and math.isfinite(sample_age)
             and 0 <= sample_age < 3)
    return dict(percent=battery_percent(voltage) if fresh else None,
                approximate=True, method='usable_voltage_range',
                empty_voltage=EMPTY_V, full_voltage=FULL_V,
                charging=None, remaining_minutes=None)
