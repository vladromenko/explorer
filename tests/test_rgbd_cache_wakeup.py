"""Exercise the real cache loop without loading ROS/TensorRT or opening camera."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading


def test_received_pair_wakes_cache_without_polling_sleep():
    source = Path(__file__).parents[1] / "src" / "perception.py"
    function = next(item for item in ast.parse(source.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "cache_rgbd")
    calls = []
    node = SimpleNamespace(lock=threading.Lock(), rgb=[1], depth=[1], info=object())

    class Event:
        def wait(self, timeout):
            calls.append(("wait", timeout))
            # A new callback delivers the next registered exposure while idle.
            node.rgb[:] = [2]
            node.depth[:] = [2]

        def clear(self):
            calls.append(("clear",))

    node.rgbd_updated = Event()
    iterations = iter([True, True, False])

    def pair(rgb, depth, info, previous):
        return rgb[0] if rgb[0] > previous and depth[0] == rgb[0] else None

    def save(path, value):
        calls.append(("save", value))
        return value

    namespace = {"rclpy": SimpleNamespace(ok=lambda: next(iterations)),
        "ROOT": Path("/unused"), "registered_pair": pair, "save_snapshot": save}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    namespace["cache_rgbd"](node)
    assert [item for item in calls if item[0] == "save"] == [("save", 1), ("save", 2)]
    assert calls[:3] == [("save", 1), ("wait", .10), ("clear",)]


def test_all_camera_inputs_notify_the_shared_cache():
    source = Path(__file__).parents[1] / "src" / "perception.py"
    cls = next(item for item in ast.parse(source.read_text()).body
        if isinstance(item, ast.ClassDef) and item.name == "Perception")
    for name in ("color_cb", "depth_cb", "info_cb"):
        method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == name)
        assert any(isinstance(item, ast.Call) and ast.unparse(item.func) == "self.rgbd_updated.set"
            for item in ast.walk(method)), name
