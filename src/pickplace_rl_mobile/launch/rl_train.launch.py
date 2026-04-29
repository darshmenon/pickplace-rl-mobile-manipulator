from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def _make_train_node(context):
    args = [
        '--timesteps', LaunchConfiguration('timesteps').perform(context),
        '--save-dir', LaunchConfiguration('save_dir').perform(context),
        '--curriculum-stage', LaunchConfiguration('curriculum_stage').perform(context),
    ]

    load_model = LaunchConfiguration('load_model').perform(context).strip()
    if load_model:
        args.extend(['--load-model', load_model])

    rl_node = Node(
        package='pickplace_rl_mobile',
        executable='train_rl',
        name='rl_env_node',
        output='screen',
        arguments=args,
    )

    return [rl_node]


def generate_launch_description():
    timesteps_arg = DeclareLaunchArgument(
        'timesteps',
        default_value='500000',
        description='Training timesteps for this run'
    )
    save_dir_arg = DeclareLaunchArgument(
        'save_dir',
        default_value='./rl_models',
        description='Directory used for checkpoints and logs'
    )
    curriculum_stage_arg = DeclareLaunchArgument(
        'curriculum_stage',
        default_value='0',
        description='Curriculum stage to train (0=full task)'
    )
    load_model_arg = DeclareLaunchArgument(
        'load_model',
        default_value='',
        description='Path to a saved model to resume training'
    )

    return LaunchDescription([
        timesteps_arg,
        save_dir_arg,
        curriculum_stage_arg,
        load_model_arg,
        OpaqueFunction(function=_make_train_node),
    ])
