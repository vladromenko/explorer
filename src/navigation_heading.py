"""Face the first useful planned segment before room exploration translation."""
import math


def departure_heading(points,pose):
    if not points:raise ValueError("Empty navigation path")
    x,y=pose["x"],pose["y"]
    for point in points:
        if len(point)!=2 or not all(type(value) in (int,float) and math.isfinite(value) for value in point):
            raise ValueError("Invalid navigation path")
    target=next((point for point in points if math.hypot(point[0]-x,point[1]-y)>=.15),points[-1])
    if math.hypot(target[0]-x,target[1]-y)<.01:return pose["yaw"]
    return math.atan2(target[1]-y,target[0]-x)
