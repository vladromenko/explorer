"""Verify the adapter preserves original stream parameters without running ROS."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    launch = ModuleType("launch")
    launch.LaunchDescription = lambda actions: SimpleNamespace(entities=actions)
    actions = ModuleType("launch.actions")
    actions.DeclareLaunchArgument = lambda name, default_value: (name, default_value)
    substitutions = ModuleType("launch.substitutions")

    class Configuration:
        def __init__(self, name):
            self.name = name

        def perform(self, context):
            return context[self.name]

    substitutions.LaunchConfiguration = Configuration
    packages = ModuleType("ament_index_python.packages")
    packages.get_package_share_directory = lambda name: None
    for name, value in (("launch", launch), ("launch.actions", actions),
                        ("launch.substitutions", substitutions),
                        ("ament_index_python.packages", packages)):
        monkeypatch.setitem(sys.modules, name, value)
    path = Path(__file__).parents[1] / "bin" / "camera-rgbd.launch.py"
    spec = importlib.util.spec_from_file_location("camera_rgbd_test_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_two_startup_parameters_are_added(adapter):
    stream = {"color_fps": 15, "depth_fps": 15, "depth_registration": True,
              "align_mode": "HW", "color_width": 640, "color_height": 480}
    vendor = SimpleNamespace(load_parameters=lambda context, args: dict(stream),
        generate_launch_description=lambda: SimpleNamespace(entities=["original_action"]))
    description = adapter.extend_vendor_launch(vendor)
    assert description.entities[-1] == "original_action"
    assert description.entities[:2] == [("enable_frame_sync", "false"),
                                        ("frame_aggregate_mode", "ANY")]
    actual = vendor.load_parameters({"enable_frame_sync": "true",
                                     "frame_aggregate_mode": "full_frame"}, [])
    assert actual == {**stream, "enable_frame_sync": True,
                      "frame_aggregate_mode": "full_frame"}


@pytest.mark.parametrize("sync,mode", [("1", "ANY"), ("true", "pretend_sync")])
def test_unknown_startup_values_do_not_silently_change_camera(adapter, sync, mode):
    vendor = SimpleNamespace(load_parameters=lambda context, args: {},
        generate_launch_description=lambda: SimpleNamespace(entities=[]))
    adapter.extend_vendor_launch(vendor)
    with pytest.raises(ValueError):
        vendor.load_parameters({"enable_frame_sync": sync, "frame_aggregate_mode": mode}, [])


def test_unreviewed_driver_version_is_rejected_before_loading_launcher(adapter, monkeypatch, tmp_path):
    (tmp_path / "package.xml").write_text("<package><version>2.10.6</version></package>")
    monkeypatch.setattr(adapter, "get_package_share_directory", lambda name: str(tmp_path))
    with pytest.raises(RuntimeError, match="audited Orbbec driver 1.5.22"):
        adapter.generate_launch_description()
