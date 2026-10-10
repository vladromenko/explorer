"""Bounded formation simulation over the isolated mathematical fleet."""
import math
from fleet_simulation import FleetSimulation


def simulate_formation(count=3, shape="line", spacing=.8):
    if type(count) is not int or not 2 <= count <= 8:
        raise ValueError("Simulation count must be 2..8")
    fleet=FleetSimulation(robots=[dict(robot_id="robot_"+str(i),pose=[-2.,i*.8,0.]) for i in range(count)])
    formation=fleet.formation(shape,[0.,0.,0.],spacing)
    traces=[]
    for iteration in range(120):
        arrived=True
        for identifier,assignment in formation["assignments"].items():
            pose=fleet.robots[identifier]["pose"]
            goal=assignment["pose"]
            dx,dy=goal[0]-pose[0],goal[1]-pose[1]
            distance=math.hypot(dx,dy)
            arrived=arrived and distance<.03
            velocity=[0.,0.,0.] if distance<.03 else [dx*min(.5, distance*1.5)/distance,dy*min(.5,distance*1.5)/distance,0.]
            fleet.submit(identifier,velocity,iteration+1,fleet.time+.3,fleet.session,fleet.robots[identifier]["namespace"])
        state=fleet.advance(.1)
        if iteration%5==0:traces.append(dict(time=state["time"],poses={key:row["pose"] for key,row in state["robots"].items()}))
        if arrived or state["events"]:
            break
    fleet.cancel()
    return dict(simulation_only=True,executed_on_hardware=False,formation=formation,
        reached_in_simulation=arrived,snapshot=fleet.snapshot(),trace=traces)
