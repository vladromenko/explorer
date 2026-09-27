"""SOC stays unknown until a pack-specific method is validated."""
import math

def battery_percent(voltage):
    return None


def battery_summary(voltage, sample_age):
    fresh = (type(sample_age) in (int, float) and math.isfinite(sample_age)
             and 0 <= sample_age < 3)
    valid = (type(voltage) in (int, float) and math.isfinite(voltage) and 5 <= voltage <= 15)
    return dict(percent=None, approximate=False, method='unavailable_uncharacterized_pack',
                voltage_valid=fresh and valid, charging=None, remaining_minutes=None,
                reason='No validated OCV curve, pack current sensor or coulomb counter')
