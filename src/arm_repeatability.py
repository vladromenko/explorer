"""Finite near-home repeatability plans. No ROS or actuator access."""
from arm_commissioning import HARD_LIMITS


def plan(start, joints=(1,2,3,4), approach_from_above=False, approach_degrees=2):
    if type(approach_degrees) is not int or approach_degrees not in (2,4):
        raise ValueError('Approach distance must be 2 or 4 degrees')
    if len(start)!=6 or any(type(v) is not int or not lo<=v<=hi
                           for v,(lo,hi) in zip(start,HARD_LIMITS)):
        raise ValueError('Invalid initial servo command')
    if not joints or len(set(joints))!=len(joints) or any(j not in (1,2,3,4) for j in joints):
        raise ValueError('Only camera-observable joints 1..4 are supported')
    steps=[];pose=list(start)
    for joint in joints:
        i=joint-1;lo,hi=HARD_LIMITS[i]
        center=min(hi-2,max(lo+2,start[i]))
        if abs(center-start[i]) not in (0,2):raise ValueError('Center requires unsupported step')
        if center!=pose[i]:
            delta=center-pose[i];pose[i]=center
            steps.append(dict(joint=joint,delta=delta,pose=list(pose)))
        steps.append(dict(capture=f'j{joint}_initial',pose=list(pose)))
        for delta,label in ((-2,None),(2,'from_below'),(2,None),(-2,'from_above')):
            pose[i]+=delta
            steps.append(dict(joint=joint,delta=delta,pose=list(pose)))
            if label:steps.append(dict(capture=f'j{joint}_{label}',pose=list(pose)))
        if pose[i]!=start[i]:
            delta=start[i]-pose[i];pose[i]=start[i]
            steps.append(dict(joint=joint,delta=delta,pose=list(pose)))
    if approach_from_above:
        expanded=[]
        for step in steps:
            if 'capture' in step:
                joint=int(step['capture'][1]);pose=list(step['pose'])
                if pose[joint-1]+approach_degrees>HARD_LIMITS[joint-1][1]:raise ValueError('No approach margin')
                for delta in [2]*(approach_degrees//2)+[-2]*(approach_degrees//2):
                    pose[joint-1]+=delta
                    expanded.append(dict(joint=joint,delta=delta,pose=list(pose)))
            expanded.append(step)
        return expanded
    return steps
