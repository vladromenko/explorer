"""Display-only voltage reserve. This is not a measured state of charge.

0% means the software stopping voltage; 100% means the standard pack's full
voltage. Motor load and charging affect it. Never use it for motion permission,
charge detection, remaining runtime or ROS BatteryState.percentage.
"""
import math

EMPTY_V = 10.8
FULL_V = 12.6

def battery_percent(voltage):
    if type(voltage) not in (int, float) or not math.isfinite(voltage) or not 9 <= voltage <= 13:
        return None
    span = max(0., min(1., (voltage-EMPTY_V)/(FULL_V-EMPTY_V)))
    return int(5*round(span*20))


def battery_summary(voltage, sample_age):
    fresh = (type(sample_age) in (int, float) and math.isfinite(sample_age)
             and 0 <= sample_age < 3)
    percent = battery_percent(voltage) if fresh else None
    return dict(percent=percent, approximate=percent is not None,
                method='operating_voltage_reserve', state_of_charge_percent=None,
                voltage_valid=percent is not None, charging=None, remaining_minutes=None,
                empty_voltage_v=EMPTY_V, full_voltage_v=FULL_V,
                reason='Voltage scale only: 0% at software stop, 100% at standard pack full voltage; not measured charge')
