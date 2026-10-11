"""Select Nav2 goal checkers explicitly without changing its normal recovery tree."""
from pathlib import Path
import xml.etree.ElementTree as ET


def explicit_default_trees(nav2_share, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, name in (("pose", "navigate_to_pose_w_replanning_and_recovery.xml"),
            ("poses", "navigate_through_poses_w_replanning_and_recovery.xml")):
        tree = ET.parse(Path(nav2_share) / "behavior_trees" / name)
        actions = tree.findall(".//FollowPath")
        if not actions:
            raise ValueError("Installed Nav2 default tree lacks FollowPath")
        for action in actions:
            if not action.get("goal_checker_id"):
                action.set("goal_checker_id", "goal_checker")
        path = destination / name
        tree.write(path, encoding="unicode")
        paths[kind] = str(path)
    return paths
