"""Small operator-directed Cartesian steps through nominal MoveIt geometry.

This module has no actuator access. Integer servo quantization is checked in
Cartesian space before the persistent finite-step controller can accept it.
"""
import itertools
import math
import numpy as np


def propose(model, start, axis, direction):
    if axis not in ('x','y','z') or type(direction) is not int or direction not in (-1,1):
        raise ValueError('Выберите ось X/Y/Z и направление −1/+1')
    if len(start)!=6 or any(type(v) is not int for v in start):
        raise ValueError('Нужно известное исходное положение команды')
    origin=np.array(model.fk(start[:5],-.3)['xyz'])
    target=origin.copy();target['xyz'.index(axis)]+=.005*direction
    solution=model.ik(target.tolist(),start[:5],-.3,max_step_deg=2)
    if not solution['solved'] or solution['collision']:
        raise ValueError('Малый шаг в этом направлении недостижим; измените позу суставами')
    values=np.asarray(solution['servo_deg'],dtype=float)
    if values.shape!=(5,) or not np.isfinite(values).all():raise ValueError('Неверный результат IK')
    candidates=[]
    for q in itertools.product(*[sorted({math.floor(v),math.ceil(v)}) for v in values[:4]]):
        goal=[*q,start[4],start[5]]
        bounded=any(a!=b for a,b in zip(start,goal)) and max(abs(a-b) for a,b in zip(start,goal))<=2
        if bounded:
            try:
                result=model.fk(goal[:5],-.3);position=np.array(result['xyz'])
                error=float(np.linalg.norm(position-target))
                progress=(position-origin)['xyz'.index(axis)]*direction
                if not result['collision'] and error<=.003 and progress>=.002:
                    candidates.append((error,goal,position))
            except ValueError:pass
    if not candidates:raise ValueError('Точность целых углов или предел шага 2° не позволяют этот сдвиг')
    error,goal,position=min(candidates,key=lambda x:x[0])
    return dict(start_deg=start,goal_deg=goal,requested_xyz_m=target.tolist(),
                predicted_xyz_m=position.tolist(),predicted_delta_m=(position-origin).tolist(),
                quantization_error_m=error,reference_only=True,measured=False,executed=False)
