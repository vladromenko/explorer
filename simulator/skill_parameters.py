"""Learnable control choices, never physics or evidence acceptance thresholds."""
import math

PARAMETERS={
    'pick_fraction':(.20,.55,.50,'pick'),
    'pick_surface_offset_m':(.007,.015,.010,'pick'),
    'lift_clearance_m':(.12,.16,.12,'pick'),
    'view_retreat_m':(.16,.26,.20,'view'),
    'view_baseline_m':(.03,.06,.035,'view'),
    'reorientation_distance_m':(.22,.42,.40,'place'),
    'reorientation_camera_height_m':(.30,.38,.30,'place'),
    'release_fraction':(.30,.65,.35,'place'),
    'release_margin_m':(.035,.075,.035,'place'),
    'withdraw_height_m':(.07,.12,.07,'place'),
}


def parameters(values=None):
    result={name:spec[2] for name,spec in PARAMETERS.items()}
    if values is None:return result
    if not isinstance(values,dict) or set(values)-set(PARAMETERS):
        raise ValueError('Unknown skill policy parameters')
    for name,value in values.items():
        lower,upper,_,_=PARAMETERS[name]
        if type(value) not in (int,float) or not math.isfinite(value) or not lower<=value<=upper:
            raise ValueError('Out-of-range skill parameter: '+name)
        result[name]=float(value)
    return result
