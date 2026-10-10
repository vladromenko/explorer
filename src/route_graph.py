"""Map-bound permitted road network; blocked edges are excluded from routing."""
import heapq
import json
import math
from pathlib import Path
import threading
import time

from lerobot_bridge import write_json


def shortest_path(graph, start, destination, epoch):
    if graph.get("map_epoch") != epoch:
        raise ValueError("Road network belongs to another map")
    nodes = graph.get("nodes", {})
    if start not in nodes or destination not in nodes:
        raise ValueError("Unknown road network place")
    links = {name: [] for name in nodes}
    for edge in graph.get("edges", []):
        if edge.get("closed") is not True:
            links[edge["from"]].append((edge["to"], float(edge["cost"])))
            if edge.get("bidirectional", True):
                links[edge["to"]].append((edge["from"], float(edge["cost"])))
    queue = [(0., start, [start])]
    visited = set()
    while queue:
        cost, name, route = heapq.heappop(queue)
        if name not in visited:
            visited.add(name)
            if name == destination:
                return {"places": route, "cost": cost, "graph_version": graph["version"], "map_epoch": epoch}
            for neighbour, weight in links[name]:
                if neighbour not in visited:
                    heapq.heappush(queue, (cost+weight, neighbour, route+[neighbour]))
    raise ValueError("No open permitted route to the destination")


class RouteGraph:
    def __init__(self, root, places, epoch, preview):
        self.path = Path(root) / "data/road-network.json"
        self.places, self.epoch, self.preview = places, epoch, preview
        self.lock = threading.Lock()

    def status(self):
        try:
            value = json.loads(self.path.read_text())
        except FileNotFoundError:
            value = {"version": 0, "nodes": {}, "edges": [], "map_epoch": self.epoch()}
        return dict(value, compatible_map=value["map_epoch"] == self.epoch())

    def save(self, edges):
        if not isinstance(edges, list) or not 1 <= len(edges) <= 100:
            raise ValueError("Specify 1..100 permitted edges")
        available = {place["name"]: place for place in self.places() if place["compatible_map"]}
        validated = []
        nodes = {}
        for edge in edges:
            if not isinstance(edge, dict) or set(edge)-{"from", "to", "cost", "closed", "bidirectional"}:
                raise ValueError("Invalid road network edge")
            a, b = edge.get("from"), edge.get("to")
            if a == b or a not in available or b not in available:
                raise ValueError("Road network nodes must be different saved places on the current map")
            distance = math.hypot(available[a]["x"]-available[b]["x"], available[a]["y"]-available[b]["y"])
            cost = edge.get("cost", max(.001, distance))
            if type(cost) not in (int, float) or not math.isfinite(cost) or not max(.001, distance) <= cost <= 10000:
                raise ValueError("Edge cost must be finite and at least the metric distance")
            if type(edge.get("closed", False)) is not bool or type(edge.get("bidirectional", True)) is not bool:
                raise ValueError("Edge flags must be boolean")
            # Planning is checked again by Nav2 from the actual pose at execution;
            # this graph describes operator-permitted edges, not measured clearance.
            validated.append(dict(edge, cost=float(cost)))
            for name in (a, b):
                nodes[name] = {key: available[name][key] for key in ("x", "y", "yaw")}
        with self.lock:
            current = self.status()
            value = {"version": current["version"]+1, "map_epoch": self.epoch(), "saved_at": time.time(),
                     "nodes": nodes, "edges": validated, "edge_clearance_verified": False}
            write_json(self.path, value)
        return self.status()

    def route(self, start, destination):
        value = self.status()
        current = {place["name"]: place for place in self.places() if place["compatible_map"]}
        for name, saved in value["nodes"].items():
            actual = current.get(name)
            if actual is None or any(abs(actual[key]-saved[key]) > 1e-6 for key in ("x", "y", "yaw")):
                raise ValueError("Road network place changed; update the network")
        return shortest_path(value, start, destination, self.epoch())
