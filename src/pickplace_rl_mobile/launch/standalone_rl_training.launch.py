import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _make_train_node(context):
    args = []
    load_model = LaunchConfiguration('load_model').perform(context).strip()
    if load_model:
        args.extend(['--load-model', load_model])

    return [
        Node(
            package='pickplace_rl_mobile',
            executable='train_rl',
            name='rl_env_node',
            output='screen',
            arguments=args,
        )
    ]


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
    
    # Delay RL node until Gazebo starts + robot spawns (8s) + bridge settles
    delayed_rl_train_node = TimerAction(
        period=20.0,
        actions=[OpaqueFunction(function=_make_train_node)]
    )

    return LaunchDescription([
        load_model_arg,
        gazebo_launch,
        delayed_rl_train_node
    ])
