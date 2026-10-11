"""Plan a bounded base correction that puts an object in the arm's front workspace."""
import math


def reachable_correction(object_xyz, reachable, preferred_distance_m=.24, arrival_margin_m=0.):
    """Choose a base position using the complete arm approach/contact/lift test.

    This is geometry only. Nav2 must check the map and execute the base goal;
    after arrival the object must be detected again before arm execution.
    Sampling distances does not change any joint or collision limits.
    """
    if not math.isfinite(arrival_margin_m) or not 0. <= arrival_margin_m <= .10:
        raise ValueError("Invalid base arrival margin")
    initial = correction(object_xyz, preferred_distance_m)
    x, y, z = (float(value) for value in object_xyz)
    distance = initial["object_distance_m"]
    if abs(initial["bearing_rad"]) <= .12 and reachable([x, y, z]):
        return dict(initial, aligned=True, actions=[], reobserve_required=False,
            reachability_checked=True, nominal_arm_geometry=True)
    choices = {preferred_distance_m + delta for delta in (0., -.02, .02, -.04, .04, -.06, .06)}
    choices.add(distance)
    # When a correction is required, prefer the central working distance
    # over a minimally shorter drive that leaves the arm fully stretched.
    # An already aligned, reachable pose above still needs no base motion.
    distances = sorted(choices, key=lambda value: (abs(value - preferred_distance_m), abs(value - distance)))
    for target_distance in distances:
        samples = {target_distance, target_distance - arrival_margin_m, target_distance + arrival_margin_m}
        if .03 < target_distance <= 1. and all(.03 < value <= 1. and reachable([value, 0., z]) for value in sorted(samples)):
            plan = correction(object_xyz, target_distance, bearing_tolerance_rad=0.)
            # A small correction still matters if the current IK failed. Do not
            # discard it using correction()'s ordinary 2.5 cm range tolerance.
            if abs(distance - target_distance) > .001 and not any(
                    action["kind"] == "translate" for action in plan["actions"]):
                delta = distance - target_distance
                plan["actions"].append({"kind": "translate", "distance_m": delta,
                    "direction": "forward" if delta > 0 else "backward"})
            if plan["actions"]:
                return dict(plan, aligned=False, reobserve_required=True,
                    reachability_checked=True, nominal_arm_geometry=True, arrival_margin_m=arrival_margin_m)
    raise ValueError("Нет позиции базы с достижимыми подходом, контактом и подъёмом предмета")

def correction(object_xyz, desired_distance_m=.24, bearing_tolerance_rad=.12):
    x,y,z=(float(v) for v in object_xyz)
    distance=math.hypot(x,y);bearing=math.atan2(y,x)
    if not all(math.isfinite(v) for v in (x,y,z,distance,bearing)) or distance<.03:
        raise ValueError('Invalid object position for base alignment')
    actions=[]
    if abs(bearing)>bearing_tolerance_rad:
        actions.append(dict(kind='rotate',yaw_rad=bearing,direction='ccw' if bearing>0 else 'cw'))
    range_error=distance-desired_distance_m
    if abs(range_error)>.025:
        actions.append(dict(kind='translate',distance_m=range_error,
                            direction='forward' if range_error>0 else 'backward'))
    return dict(object_distance_m=distance,bearing_rad=bearing,desired_distance_m=desired_distance_m,
                aligned=not actions,actions=actions,reobserve_required=bool(actions),
                rule='rotate_then_range_then_reobserve')
