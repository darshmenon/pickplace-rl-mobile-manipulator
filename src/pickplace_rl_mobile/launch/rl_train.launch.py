from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    load_model_arg = DeclareLaunchArgument(
        'load_model',
        default_value='',
        description='Path to a saved model to resume training'
    )

    rl_node = Node(
        package='pickplace_rl_mobile',
        executable='train_rl',
        name='rl_env_node',
        output='screen',
        arguments=['--load-model', LaunchConfiguration('load_model')]
    )

    return LaunchDescription([
        load_model_arg,
        rl_node
    ])