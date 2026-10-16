"""Learnable control choices, never physics or evidence acceptance thresholds."""
import math

PARAMETERS={
    'pick_fraction':(.20,.55,.50,'pick'),
    'pick_lateral_fraction':(.20,.80,.50,'pick'),
    'pick_standoff_m':(.18,.28,.24,'pick'),
    'pick_approach_angle_deg':(-65.,65.,0.,'pick'),
    'pick_surface_offset_m':(.007,.015,.010,'pick'),
    'lift_clearance_m':(.12,.16,.12,'pick'),
    'initial_lift_height_m':(.10,.16,.10,'pick'),
    'closure_duration_s':(3.,6.,3.,'pick'),
    'view_retreat_m':(.16,.26,.20,'view'),
    'view_baseline_m':(.03,.06,.035,'view'),
    'inspection_reach_m':(.20,.28,.23,'view'),
    'inspection_height_m':(.13,.20,.15,'view'),
    'reorientation_distance_m':(.22,.42,.40,'place'),
    'reorientation_camera_height_m':(.30,.38,.30,'place'),
    'destination_standoff_m':(.21,.29,.23,'place'),
    'release_fraction':(.15,.65,.22,'place'),
    'release_margin_m':(.035,.16,.12,'place'),
    'release_roll_deg':(0.,25.,12.,'place'),
    'withdraw_height_m':(.07,.12,.07,'place'),
    'verification_distance_m':(.18,.50,.30,'place'),
    'verification_aim_fraction':(.10,.80,.40,'place'),
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
