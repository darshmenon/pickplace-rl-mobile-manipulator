#!/usr/bin/env python3
"""
Full System Launch for Pick-and-Place Mobile Manipulator.

Brings up the complete system:
- Gazebo Harmonic simulation with custom world
- Robot state publisher
- ros_gz_bridge (all sensor + command topics)
- Perception node (camera-based object detection)
- Safety guard node
- Manipulation RL node (optional)
- Nav2 navigation stack (optional)
"""

import os
import re
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument, GroupAction, TimerAction,
    AppendEnvironmentVariable
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def resolve_package_uris(urdf_str):
    """Replace package:// URIs with absolute file:// paths so Gazebo can find meshes."""
    def replace(match):
        pkg = match.group(1)
        rel = match.group(2)
        try:
            share = get_package_share_directory(pkg)
            return f'file://{share}/{rel}'
        except Exception:
            return match.group(0)
    return re.sub(r'package://([^/]+)/([^"\'>\s]+)', replace, urdf_str)


def get_pickplace_share_dir():
    """Resolve active package share dir while supporting symlink-install/source runs."""
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    source_share = os.path.dirname(launch_dir)
    if os.path.exists(os.path.join(source_share, 'urdf', 'mobile_ur3.urdf')):
        return source_share
    return get_package_share_directory('pickplace_rl_mobile')


def generate_launch_description():
    pkg_dir = get_pickplace_share_dir()
    urdf_file = os.path.join(pkg_dir, 'urdf', 'mobile_ur3.urdf')
    world_file = os.path.join(pkg_dir, 'worlds', 'pickplace_world.world')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    ur_description_share = get_package_share_directory('ur_description')
    robotiq_share = get_package_share_directory('robotiq_2f_85_gripper_visualization')

    with open(urdf_file, 'r') as f:
        robot_description_content = resolve_package_uris(f.read())

    set_gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        ':'.join([
            os.path.join(ur_description_share, '..'),
            os.path.join(robotiq_share, '..'),
        ])
    )

    # Launch arguments
    use_nav2_arg = DeclareLaunchArgument(
        'use_nav2', default_value='false',
        description='Whether to launch Nav2 navigation stack')
    use_rl_arg = DeclareLaunchArgument(
        'use_rl', default_value='false',
        description='Whether to launch the RL manipulation node')
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='./rl_models/best_model/best_model.zip',
        description='Path to the trained RL model')
    use_perception_arg = DeclareLaunchArgument(
        'use_perception', default_value='true',
        description='Use camera perception for object detection (false = fallback fixed position)')

    # Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={
            'gz_args': f'-r -v 1 --physics-engine gz-physics-bullet-featherstone-plugin {world_file}'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True
        }]
    )

    # Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'pickplace_world',
            '-name', 'mobile_ur3',
            '-allow_renaming', 'true',
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.08'
        ],
        output='screen'
    )

    # ros_gz_bridge — all topics including new sensors
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Core topics
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            # Camera topics
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/depth@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            # LiDAR
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            # Joint command bridges
            '/shoulder_pan_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/shoulder_lift_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/elbow_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/wrist_1_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/wrist_2_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/wrist_3_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/finger_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/left_inner_knuckle_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/left_inner_finger_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/right_outer_knuckle_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/right_inner_knuckle_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double',
            '/right_inner_finger_joint/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double'
        ],
        output='screen'
    )

    # Perception node
    perception_node = Node(
        package='pickplace_rl_mobile',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Safety guard node
    safety_guard = Node(
        package='pickplace_rl_mobile',
        executable='safety_guard',
        name='safety_guard',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Controller Spawners
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster']
    )
    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller']
    )
    gripper_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller']
    )

    # Manipulation RL node (optional)
    rl_node = GroupAction(
        condition=IfCondition(LaunchConfiguration('use_rl')),
        actions=[
            Node(
                package='pickplace_rl_mobile',
                executable='manip_rl_node',
                name='manip_rl_node',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'model_path': LaunchConfiguration('model_path'),
                    'use_perception': LaunchConfiguration('use_perception')
                }]
            )
        ]
    )

    # Nav2 (optional)
    nav2_group = GroupAction(
        condition=IfCondition(LaunchConfiguration('use_nav2')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(
                        get_package_share_directory('nav2_bringup'),
                        'launch', 'navigation_launch.py')
                ]),
                launch_arguments={
                    'use_sim_time': 'true',
                    'params_file': nav2_params
                }.items()
            )
        ]
    )

    delayed = TimerAction(
        period=8.0,
        actions=[spawn_robot, bridge, perception_node, safety_guard, rl_node, nav2_group, jsb_spawner, arm_spawner, gripper_spawner],
    )

    return LaunchDescription([
        use_nav2_arg,
        use_rl_arg,
        model_path_arg,
        use_perception_arg,
        set_gz_resource_path,
        gazebo,
        robot_state_publisher,
        delayed,
    ])
