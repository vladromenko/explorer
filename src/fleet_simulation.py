"""Isolated multi-robot math simulation. No ROS imports, transports or actuators."""
import copy
import math
import re
import secrets


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value) is None:
        raise ValueError("Simulation ID must be a lowercase identifier")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected finite simulation number")
    return float(value)


def _pose(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Simulation pose is [x, y, yaw]")
    return tuple(_number(item) for item in value)


def _distance_to_segment(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = dx * dx + dy * dy
    fraction = 0. if length == 0. else max(0., min(1., ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length))
    return math.hypot(start[0] + fraction * dx - point[0], start[1] + fraction * dy - point[1])


class FleetSimulation:
    """At most eight circular virtual robots, explicit simulation time and leases.

    Returned namespace/TF descriptors are data only. This class has no publisher,
    callback, file restore, physical adapter or namespace remapping facility.
    """
    def __init__(self, fleet_id="demo", robots=(), obstacles=()):
        self.fleet_id = _identifier(fleet_id)
        self.namespace = "/simulation/fleet/" + self.fleet_id
        self.session = secrets.token_hex(16)
        self.time = 0.
        self.robots = {}
        self.commands = {}
        self.sequences = {}
        self.events = []
        self.obstacles = []
        for obstacle in obstacles:
            if not isinstance(obstacle, (list, tuple)) or len(obstacle) != 3:
                raise ValueError("Simulation obstacle is [x, y, radius]")
            x, y, radius = (_number(item) for item in obstacle)
            if radius <= 0.:
                raise ValueError("Obstacle radius must be positive")
            self.obstacles.append((x, y, radius))
        for robot in robots:
            self.register(**robot)

    def register(self, robot_id, pose=(0., 0., 0.), radius=.2, namespace=None):
        robot_id = _identifier(robot_id)
        derived = self.namespace + "/" + robot_id
        if namespace is not None and namespace != derived:
            raise ValueError("Only the derived simulation namespace is permitted")
        if robot_id in self.robots or len(self.robots) >= 8:
            raise ValueError("Simulation IDs must be unique; maximum eight robots")
        pose, radius = _pose(pose), _number(radius)
        if not .01 <= radius <= 2.:
            raise ValueError("Simulation radius must be within 0.01..2 metres")
        if any(math.hypot(pose[0] - item[0], pose[1] - item[1]) <= radius + item[2] for item in self.obstacles):
            raise ValueError("Initial virtual pose intersects an obstacle")
        if any(math.hypot(pose[0] - item["pose"][0], pose[1] - item["pose"][1]) <= radius + item["radius"] for item in self.robots.values()):
            raise ValueError("Initial virtual robots intersect")
        self.robots[robot_id] = {"pose": pose, "radius": radius, "namespace": derived,
                                 "cmd_topic": derived + "/cmd_vel", "tf_topic": derived + "/tf",
                                 "map_frame": derived.lstrip("/") + "/map",
                                 "base_frame": derived.lstrip("/") + "/base_footprint"}
        self.sequences[robot_id] = 0
        return copy.deepcopy(self.robots[robot_id])

    def snapshot(self):
        return {"simulation_only": True, "execution_allowed": False, "physical_publishers": 0,
                "session": self.session, "time": self.time, "robots": copy.deepcopy(self.robots),
                "active_commands": copy.deepcopy(self.commands), "events": copy.deepcopy(self.events)}

    def submit(self, robot_id, velocity, sequence, deadline, session, namespace):
        """Accept only current-session, ordered, short-lived virtual body velocity."""
        robot = self.robots.get(robot_id)
        if robot is None or namespace != robot["namespace"] or session != self.session:
            raise ValueError("Simulation identity or namespace mismatch")
        velocity = _pose(velocity)
        deadline = _number(deadline)
        if type(sequence) is not int or sequence <= self.sequences[robot_id]:
            raise ValueError("Simulation command replay or invalid sequence")
        if not self.time < deadline <= self.time + 1.:
            raise ValueError("Simulation command must expire within one second")
        if math.hypot(velocity[0], velocity[1]) > .8 or abs(velocity[2]) > 1.5:
            raise ValueError("Simulation velocity exceeds its configured bounds")
        self.sequences[robot_id] = sequence
        self.commands[robot_id] = {"velocity": velocity, "deadline": deadline, "sequence": sequence}
        return {"simulation_only": True, "accepted": True, "sequence": sequence}

    def cancel(self, robot_id=None):
        if robot_id is None:
            self.commands.clear()
        elif robot_id in self.robots:
            self.commands.pop(robot_id, None)
        else:
            raise ValueError("Unknown virtual robot")

    def reset_session(self):
        """Keep virtual poses; invalidate every old lease, including cancelled ones."""
        self.cancel()
        self.session = secrets.token_hex(16)
        self.sequences = {robot_id: 0 for robot_id in self.robots}
        return self.session

    def advance(self, duration):
        duration = _number(duration)
        if not 0. < duration <= 10.:
            raise ValueError("Advance duration must be within 0..10 seconds")
        steps = math.ceil(duration / .02)
        dt = duration / steps
        for _ in range(steps):
            previous = {key: item["pose"] for key, item in self.robots.items()}
            proposed = {}
            for robot_id, robot in self.robots.items():
                command = self.commands.get(robot_id)
                alive_dt = min(dt, max(0., command["deadline"] - self.time)) if command else 0.
                vx, vy, wz = command["velocity"] if command else (0., 0., 0.)
                x, y, yaw = robot["pose"]
                midpoint = yaw + .5 * wz * alive_dt
                proposed[robot_id] = (x + (vx * math.cos(midpoint) - vy * math.sin(midpoint)) * alive_dt,
                                      y + (vx * math.sin(midpoint) + vy * math.cos(midpoint)) * alive_dt,
                                      math.atan2(math.sin(yaw + wz * alive_dt), math.cos(yaw + wz * alive_dt)))
            blocked = set()
            for robot_id, robot in self.robots.items():
                for x, y, radius in self.obstacles:
                    if _distance_to_segment((x, y), previous[robot_id], proposed[robot_id]) <= radius + robot["radius"]:
                        blocked.add(robot_id)
            names = sorted(self.robots)
            for index, first in enumerate(names):
                for second in names[index + 1:]:
                    relative_start = (previous[first][0] - previous[second][0], previous[first][1] - previous[second][1])
                    relative_end = (proposed[first][0] - proposed[second][0], proposed[first][1] - proposed[second][1])
                    if _distance_to_segment((0., 0.), relative_start, relative_end) <= self.robots[first]["radius"] + self.robots[second]["radius"]:
                        blocked.update((first, second))
            # Reject the complete synchronous step on collision: stopping only one
            # proposal could create a collision with a robot expected to move away.
            if blocked:
                self.cancel()
                proposed = previous
                self.events.append({"time": self.time, "reason": "simulated_collision", "robots": sorted(blocked)})
                self.events = self.events[-100:]
            for robot_id, pose in proposed.items():
                self.robots[robot_id]["pose"] = pose
            self.time += dt
            expired = [key for key, command in self.commands.items() if command["deadline"] <= self.time + 1e-12]
            for robot_id in expired:
                self.commands.pop(robot_id)
        return self.snapshot()

    def assign(self, targets):
        """Deterministic minimum-total-distance bijection, small exact subset DP.

        This proposes assignments; it does not claim an obstacle-free route.
        """
        names = sorted(self.robots)
        if not isinstance(targets, dict) or len(targets) != len(names) or not names:
            raise ValueError("One target per registered virtual robot is required")
        goals = {key: _pose(value) for key, value in targets.items()}
        target_names = sorted(_identifier(key) for key in goals)
        states = {0: (0., ())}
        for robot_id in names:
            next_states = {}
            for mask, (cost, choices) in states.items():
                for index, target_id in enumerate(target_names):
                    if not mask & (1 << index):
                        pose, goal = self.robots[robot_id]["pose"], goals[target_id]
                        value = (cost + math.hypot(pose[0] - goal[0], pose[1] - goal[1]), choices + (target_id,))
                        new_mask = mask | (1 << index)
                        if new_mask not in next_states or value < next_states[new_mask]:
                            next_states[new_mask] = value
            states = next_states
        cost, choices = states[(1 << len(names)) - 1]
        return {"simulation_only": True, "execution_allowed": False, "route_verified": False,
                "total_distance_m": cost, "assignments": {robot_id: {"target_id": target, "pose": goals[target]}
                    for robot_id, target in zip(names, choices)}}

    def formation(self, shape, anchor, spacing=.8):
        """Return rotated line/grid/V slots and assignments without sending commands."""
        anchor, spacing = _pose(anchor), _number(spacing)
        if shape not in ("line", "grid", "v") or not self.robots:
            raise ValueError("Select line, grid or v for a nonempty virtual fleet")
        if not 2. * max(item["radius"] for item in self.robots.values()) < spacing <= 10.:
            raise ValueError("Formation spacing must exceed virtual robot diameters")
        targets = {}
        columns = math.ceil(math.sqrt(len(self.robots)))
        for index in range(len(self.robots)):
            if shape == "line":
                dx, dy = 0., (index - (len(self.robots) - 1) / 2.) * spacing
            elif shape == "grid":
                dx, dy = -(index // columns) * spacing, (index % columns) * spacing
            else:
                rank = (index + 1) // 2
                dx, dy = -rank * spacing, (1. if index % 2 else -1.) * rank * spacing
            x = anchor[0] + dx * math.cos(anchor[2]) - dy * math.sin(anchor[2])
            y = anchor[1] + dx * math.sin(anchor[2]) + dy * math.cos(anchor[2])
            if any(math.hypot(x - item[0], y - item[1]) <= max(robot["radius"] for robot in self.robots.values()) + item[2] for item in self.obstacles):
                raise ValueError("Formation slot intersects a simulated obstacle")
            targets["slot_" + str(index)] = (x, y, anchor[2])
        return dict(self.assign(targets), shape=shape, targets=targets)
