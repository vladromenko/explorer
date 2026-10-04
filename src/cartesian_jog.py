"""Small operator-directed Cartesian steps through nominal MoveIt geometry.

This module has no actuator access. Integer servo quantization is checked in
Cartesian space before the persistent finite-step controller can accept it.
"""
import itertools
import math
import numpy as np


def propose_delta(model,start,delta_xyz,maximum_distance=.006):
    delta=np.asarray(delta_xyz,dtype=float)
    if (not isinstance(maximum_distance,(int,float)) or not math.isfinite(maximum_distance)
            or not 0<maximum_distance<=.020):
        raise ValueError('Неверный предел локального XYZ-сдвига')
    if delta.shape!=(3,) or not np.isfinite(delta).all() or not 0<np.linalg.norm(delta)<=maximum_distance:
        raise ValueError('Нужен конечный локальный XYZ-сдвиг в разрешённом диапазоне')
    if len(start)!=6 or any(type(v) is not int for v in start):
        raise ValueError('Нужно известное исходное положение команды')
    initial=model.fk(start[:5],-.3);origin=np.array(initial['xyz']);target=origin+delta
    # A five-axis arm cannot generally hold all three tool-orientation DOF
    # while translating in XYZ. Position IK therefore uses the current pose as
    # its seed and a strict ten-degree local bound. This preserves a manual
    # servo-4 adjustment continuously without over-constraining Y/Z motion.
    solution=model.ik(target.tolist(),start[:5],-.3,max_step_deg=10)
    if not solution['solved'] or solution['collision']:
        raise ValueError('Локальный Cartesian-сдвиг недостижим; измените направление')
    values=np.asarray(solution['servo_deg'],dtype=float)
    if values.shape!=(5,) or not np.isfinite(values).all():raise ValueError('Неверный результат IK')
    candidates=[]
    # Cartesian translation preserves the independently commanded wrist roll
    # (servo 5).  The current FK orientation remains the IK target, while only
    # servos 1-4 are quantized for the factory integer-degree interface.
    for q in itertools.product(*[sorted({math.floor(v),math.ceil(v)}) for v in values[:4]]):
        goal=[*q,start[4],start[5]];bounded=any(a!=b for a,b in zip(start,goal)) and max(abs(a-b) for a,b in zip(start,goal))<=10
        if bounded:
            try:
                result=model.fk(goal[:5],-.3);position=np.array(result['xyz']);error=float(np.linalg.norm(position-target))
                progress=float(np.dot(position-origin,delta/max(np.linalg.norm(delta),1e-9)))
                if not result['collision'] and error<=.004 and progress>=.0015:candidates.append((error,goal,position))
            except ValueError:pass
    if not candidates:raise ValueError('Целые углы не позволяют безопасный локальный сдвиг')
    error,goal,position=min(candidates,key=lambda x:x[0])
    return dict(start_deg=start,goal_deg=goal,requested_xyz_m=target.tolist(),predicted_xyz_m=position.tolist(),
                predicted_delta_m=(position-origin).tolist(),quantization_error_m=error,
                wrist_pitch_seed_deg=start[3],wrist_roll_preserved_deg=start[4],
                reference_only=True,measured=False,executed=False)


def propose(model, start, axis, direction):
    if axis not in ('x','y','z') or type(direction) is not int or direction not in (-1,1):
        raise ValueError('Выберите ось X/Y/Z и направление −1/+1')
    delta=[0.,0.,0.];delta['xyz'.index(axis)]=.005*direction
    return propose_delta(model,start,delta)
