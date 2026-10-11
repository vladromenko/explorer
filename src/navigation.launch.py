"""Nav2 execution servers publish only to the Explorer safety arbiter."""
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from navigation_trees import explicit_default_trees

def generate_launch_description():
    config='/home/vlad/Explorer/config/navigation.yaml'
    trees=explicit_default_trees(get_package_share_directory("nav2_bt_navigator"),
        "/home/vlad/Explorer/data/navigation-runtime")
    defaults={"default_nav_to_pose_bt_xml":trees["pose"], "default_nav_through_poses_bt_xml":trees["poses"]}
    servers=[('nav2_controller','controller_server'),('nav2_behaviors','behavior_server'),('nav2_bt_navigator','bt_navigator')]
    nodes=[Node(package=p,executable=n,name=n,parameters=[config]+([defaults] if n=="bt_navigator" else []),
                remappings=[('cmd_vel','/explorer/nav_cmd_vel')],output='screen') for p,n in servers]
    nodes.append(Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='navigation_lifecycle',
                parameters=[{'autostart':True,'node_names':[n for _,n in servers],'bond_timeout':4.0}],output='screen'))
    return LaunchDescription(nodes)
