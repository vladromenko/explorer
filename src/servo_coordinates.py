"""Servo command angles versus CAD joint coordinates.

The vendor grasp adapter sends joint1 = 180 - IK_joint1. Reversing this axis
also resolves the independent RGB-D motion survey's translation inconsistency.
Other axes remain nominal; fitted gain/zero corrections are not deployed.
"""
import numpy as np

SIGNS=np.array([-1.,1.,1.,1.,1.])

def to_radians(servo_degrees):
    return np.radians(np.asarray(servo_degrees,dtype=float)-90.)*SIGNS

def to_servo(radians):
    return np.degrees(np.asarray(radians,dtype=float))*SIGNS+90.
