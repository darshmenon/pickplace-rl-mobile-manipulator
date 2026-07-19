from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def _make_train_node(context):
    args = [
        '--timesteps', LaunchConfiguration('timesteps').perform(context),
        '--save-dir', LaunchConfiguration('save_dir').perform(context),
        '--curriculum-stage', LaunchConfiguration('curriculum_stage').perform(context),
        '--algo', LaunchConfiguration('algo').perform(context),
        '--policy-arch', LaunchConfiguration('policy_arch').perform(context),
    ]

    load_model = LaunchConfiguration('load_model').perform(context).strip()
    if load_model:
        args.extend(['--load-model', load_model])

    if LaunchConfiguration('adaptive_curriculum').perform(context).lower() in ('true', '1'):
        args.append('--adaptive-curriculum')

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
    algo_arg = DeclareLaunchArgument(
        'algo',
        default_value='tqc',
        description='RL algorithm: tqc, sac, ppo, or ppo_lstm'
    )
    policy_arch_arg = DeclareLaunchArgument(
        'policy_arch',
        default_value='mlp',
        description='Policy feature-extractor architecture: mlp or transformer'
    )
    adaptive_curriculum_arg = DeclareLaunchArgument(
        'adaptive_curriculum',
        default_value='false',
        description='Advance curriculum stages on reward-plateau detection instead of fixed thresholds only'
    )

    return LaunchDescription([
        timesteps_arg,
        save_dir_arg,
        curriculum_stage_arg,
        load_model_arg,
        algo_arg,
        policy_arch_arg,
        adaptive_curriculum_arg,
        OpaqueFunction(function=_make_train_node),
    ])
