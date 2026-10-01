"""Plan a bounded base correction that puts an object in the arm's front workspace."""
import math

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
