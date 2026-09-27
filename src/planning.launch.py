"""Read-only Nav2 planning bringup: no controller or motor publishers."""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config='/home/vlad/Explorer/config/planning.yaml'
    return LaunchDescription([
        Node(package='nav2_planner',executable='planner_server',name='planner_server',parameters=[config],output='screen'),
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='planning_lifecycle',
             parameters=[{'use_sim_time':False,'autostart':True,'node_names':['planner_server'],'bond_timeout':4.0}],output='screen'),
    ])
