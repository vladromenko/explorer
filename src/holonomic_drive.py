"""Calibrated evdev axes to a bounded mecanum body velocity request."""
import math


def axis_value(value, calibration, deadzone):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Invalid gamepad axis')
    lo, center, hi = (calibration[k] for k in ('minimum', 'center', 'maximum'))
    if not lo < center < hi or not lo <= value <= hi:
        raise ValueError('Axis outside calibrated range')
    value = (value-center)/(hi-center if value >= center else center-lo)
    value = -value if calibration.get('inverted') else value
    return math.copysign(max(0., (abs(value)-deadzone)/(1-deadzone)), value)


def velocity(axes, config, profile='precision'):
    if profile not in config['profiles']:
        raise ValueError('Unknown drive profile')
    values=[]
    for name in ('left_y', 'left_x', 'right_x'):
        calibration=config['axes'][name]
        # Missing axes cannot be interpreted as a deflected stick.
        value=axes.get(str(calibration['code']), calibration['center'])
        values.append(axis_value(value, calibration, config['deadzone']))
    x,y,yaw=values
    scale=max(1.,math.hypot(x,y))
    limits=config['profiles'][profile]
    return [x/scale*limits['linear'], y/scale*limits['linear'], yaw*limits['angular']]


def blocked_by(state, config, now):
    reasons=[]
    if not 0 <= now-state.get('at', 0) < .9:reasons.append('Нет свежего состояния робота')
    flags=state.get('commissioning', {})
    for name,label in (
        ('base_commissioned','Калибровка шасси не завершена'),
        ('mcu_watchdog_verified','Остановка при потере связи не подтверждена'),
        ('lidar_tf_validated','Положение лидаров не подтверждено')):
        if flags.get(name) is not True:reasons.append(label)
    if not config.get('radio_loss_verified'):reasons.append('Потеря радиосвязи геймпада не проверена')
    if not config.get('continuous_motion_enabled'):reasons.append('Непрерывное управление ещё не разрешено')
    if state.get('stop_latched',True):reasons.append('Включён стоп шасси')
    return reasons
