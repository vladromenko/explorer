from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
from navigation_trees import explicit_default_trees


def test_default_recovery_and_controls_survive_explicit_checker_selection(tmp_path):
    share = tmp_path / "nav2"
    (share / "behavior_trees").mkdir(parents=True)
    source = '<root><BehaviorTree><RecoveryNode number_of_retries="6"><FollowPath controller_id="{selected_controller}" path="{path}"/><Spin spin_dist="1.57"/></RecoveryNode></BehaviorTree></root>'
    for kind in ("pose", "poses"):
        (share / "behavior_trees" / ("navigate_to_pose_w_replanning_and_recovery.xml" if kind == "pose" else "navigate_through_poses_w_replanning_and_recovery.xml")).write_text(source)
    paths = explicit_default_trees(share, tmp_path / "runtime")
    for path in paths.values():
        result = ET.parse(path)
        action = result.find(".//FollowPath")
        assert action.attrib == {"controller_id": "{selected_controller}", "path": "{path}", "goal_checker_id": "goal_checker"}
        action.attrib.pop("goal_checker_id")
        assert ET.tostring(result.getroot()) == ET.tostring(ET.fromstring(source))


def test_approach_is_precise_collision_checked_and_has_no_unrequested_recovery_motion():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/navigation.yaml").read_text())["controller_server"]["ros__parameters"]
    tree = ET.parse(root / "config/navigation-approach.xml")
    action = tree.find(".//FollowPath")
    assert action.get("controller_id") == "ApproachPath"
    assert action.get("goal_checker_id") == "approach_goal_checker"
    assert config["ApproachPath"]["use_collision_detection"] is True
    assert config["ApproachPath"]["desired_linear_vel"] <= .10
    assert config["approach_goal_checker"]["xy_goal_tolerance"] == .01
    assert not tree.findall(".//Spin") and not tree.findall(".//BackUp")
    # Ordinary mapping keeps its established omnidirectional controller.
    assert config["FollowPath"]["plugin"] == "dwb_core::DWBLocalPlanner"
    assert config["goal_checker"]["xy_goal_tolerance"] == .10
