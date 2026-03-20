import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('pickplace_rl_mobile')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_dir, 'launch', 'gazebo.launch.py')
        ])
    )
    
    rl_train_node = Node(
        package='pickplace_rl_mobile',
        executable='train_rl',
        name='rl_env_node',
        output='screen'
    )
    
    # Delay the RL node just slightly so Gazebo can spawn the robot
    delayed_rl_train_node = TimerAction(
        period=5.0,
        actions=[rl_train_node]
    )

    return LaunchDescription([
        gazebo_launch,
        delayed_rl_train_node
    ])
