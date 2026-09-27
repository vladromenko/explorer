"""Nav2 execution servers publish only to the Explorer safety arbiter."""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config='/home/vlad/Explorer/config/navigation.yaml'
    servers=[('nav2_controller','controller_server'),('nav2_behaviors','behavior_server'),('nav2_bt_navigator','bt_navigator')]
    nodes=[Node(package=p,executable=n,name=n,parameters=[config],
                remappings=[('cmd_vel','/explorer/nav_cmd_vel')],output='screen') for p,n in servers]
    nodes.append(Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='navigation_lifecycle',
                parameters=[{'autostart':True,'node_names':[n for _,n in servers],'bond_timeout':4.0}],output='screen'))
    return LaunchDescription(nodes)
