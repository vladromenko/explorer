"""Add startup RGB-D synchronization controls to the installed DCW2 launcher.

All stream, registration, and calibration parameters remain owned by the
installed launcher. Defaults reproduce its previous synchronization behavior.
"""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def extend_vendor_launch(vendor):
    original_parameters = vendor.load_parameters

    def parameters(context, args):
        result = original_parameters(context, args)
        sync = LaunchConfiguration("enable_frame_sync").perform(context).lower()
        aggregate = LaunchConfiguration("frame_aggregate_mode").perform(context)
        if sync not in ("true", "false"):
            raise ValueError("enable_frame_sync must be true or false")
        if aggregate not in ("ANY", "full_frame", "color_frame", "disable"):
            raise ValueError("Unsupported frame_aggregate_mode")
        result["enable_frame_sync"] = sync == "true"
        result["frame_aggregate_mode"] = aggregate
        return result

    vendor.load_parameters = parameters
    original = vendor.generate_launch_description()
    return LaunchDescription([
        DeclareLaunchArgument("enable_frame_sync", default_value="false"),
        DeclareLaunchArgument("frame_aggregate_mode", default_value="ANY"),
        *original.entities,
    ])


def generate_launch_description():
    share = Path(get_package_share_directory("orbbec_camera"))
    version = ET.parse(share / "package.xml").getroot().findtext("version")
    if version != "1.5.22":
        raise RuntimeError("Camera wrapper requires the audited Orbbec driver 1.5.22")
    filename = share / "launch" / "dabai_dcw2.launch.py"
    spec = importlib.util.spec_from_file_location("explorer_installed_dcw2_launch", filename)
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    return extend_vendor_launch(vendor)
