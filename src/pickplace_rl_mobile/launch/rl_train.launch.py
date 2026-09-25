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
        '--eval-freq', LaunchConfiguration('eval_freq').perform(context),
        '--n-eval-episodes', LaunchConfiguration('n_eval_episodes').perform(context),
        '--checkpoint-freq', LaunchConfiguration('checkpoint_freq').perform(context),
    ]

    gradient_steps = LaunchConfiguration('gradient_steps').perform(context).strip()
    if gradient_steps:
        args.extend(['--gradient-steps', gradient_steps])

    load_model = LaunchConfiguration('load_model').perform(context).strip()
    if load_model:
        args.extend(['--load-model', load_model])

    if LaunchConfiguration('adaptive_curriculum').perform(context).lower() in ('true', '1'):
        args.append('--adaptive-curriculum')

    if LaunchConfiguration('adaptive_domain_randomization').perform(context).lower() in ('true', '1'):
        args.append('--adaptive-domain-randomization')
        args.extend(['--adr-step', LaunchConfiguration('adr_step').perform(context)])

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
    eval_freq_arg = DeclareLaunchArgument(
        'eval_freq',
        default_value='10000',
        description='Evaluate every N environment steps before n-env scaling'
    )
    n_eval_episodes_arg = DeclareLaunchArgument(
        'n_eval_episodes',
        default_value='10',
        description='Number of episodes per evaluation batch'
    )
    checkpoint_freq_arg = DeclareLaunchArgument(
        'checkpoint_freq',
        default_value='10000',
        description='Checkpoint every N environment steps before n-env scaling'
    )
    gradient_steps_arg = DeclareLaunchArgument(
        'gradient_steps',
        default_value='',
        description='Off-policy gradient updates per environment step'
    )
    adaptive_domain_randomization_arg = DeclareLaunchArgument(
        'adaptive_domain_randomization',
        default_value='false',
        description='Widen/narrow domain randomization (friction, mass, action/perception noise) on eval performance'
    )
    adr_step_arg = DeclareLaunchArgument(
        'adr_step',
        default_value='0.15',
        description='Randomization level step size per ADR adjustment (0-1 range)'
    )

    return LaunchDescription([
        timesteps_arg,
        save_dir_arg,
        curriculum_stage_arg,
        load_model_arg,
        algo_arg,
        policy_arch_arg,
        adaptive_curriculum_arg,
        eval_freq_arg,
        n_eval_episodes_arg,
        checkpoint_freq_arg,
        gradient_steps_arg,
        adaptive_domain_randomization_arg,
        adr_step_arg,
        OpaqueFunction(function=_make_train_node),
    ])
