import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('pickplace_rl_mobile')

    load_model_arg = DeclareLaunchArgument(
        'load_model',
        default_value='',
        description='Path to a saved model to resume training'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_dir, 'launch', 'gazebo.launch.py')
        ])
    )
    
    rl_train_node = Node(
        package='pickplace_rl_mobile',
        executable='train_rl',
        name='rl_env_node',
        output='screen',
        arguments=['--load-model', LaunchConfiguration('load_model')]
    )
    
    # Delay RL node until Gazebo starts + robot spawns (8s) + bridge settles
    delayed_rl_train_node = TimerAction(
        period=20.0,
        actions=[rl_train_node]
    )

    return LaunchDescription([
        load_model_arg,
        gazebo_launch,
        delayed_rl_train_node
    ])
