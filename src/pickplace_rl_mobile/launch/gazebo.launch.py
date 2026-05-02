#!/usr/bin/env python3

import os
import re
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory


def resolve_package_uris(urdf_str):
    """Resolve package references so Gazebo can load meshes and controller params."""
    def replace_package_uri(match):
        pkg = match.group(1)
        rel = match.group(2)
        try:
            share = get_package_share_directory(pkg)
            return f'file://{share}/{rel}'
        except Exception:
            return match.group(0)

    def replace_find(match):
        pkg = match.group(1)
        try:
            return get_package_share_directory(pkg)
        except Exception:
            return match.group(0)

    urdf_str = re.sub(r'package://([^/]+)/([^"\'>\s]+)', replace_package_uri, urdf_str)
    return re.sub(r'\$\(find\s+([^)]+)\)', replace_find, urdf_str)


def get_pickplace_share_dir():
    """Resolve the active pickplace package share dir without trusting a stale ament entry."""
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    source_share = os.path.dirname(launch_dir)
    if os.path.exists(os.path.join(source_share, 'urdf', 'mobile_ur3.urdf')):
        return source_share
    return get_package_share_directory('pickplace_rl_mobile')


def generate_launch_description():

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo headless (no GUI) for faster training fps'
    )

    pkg_dir = get_pickplace_share_dir()
    world_path = os.path.join(pkg_dir, 'worlds', 'pickplace_world.world')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'mobile_ur3.urdf')
    ur_description_share = get_package_share_directory('ur_description')
    robotiq_share = get_package_share_directory('robotiq_2f_85_gripper_visualization')
    harmonic_gz_control_prefix = '/home/asimov/UR3_ROS2_PICK_AND_PLACE/install/gz_ros2_control'
    harmonic_gz_control_lib = '/home/asimov/UR3_ROS2_PICK_AND_PLACE/install/gz_ros2_control/lib'

    with open(urdf_path, 'r') as f:
        raw_urdf = f.read()

    # Resolve package:// URIs to absolute paths for robot_state_publisher and Gazebo
    robot_description = resolve_package_uris(raw_urdf)

    # Also set GZ_SIM_RESOURCE_PATH so Gazebo can find model:// and package:// assets
    set_gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        ':'.join([
            os.path.join(ur_description_share, '..'),
            os.path.join(robotiq_share, '..'),
        ])
    )
    set_gz_control_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH',
        harmonic_gz_control_lib,
        prepend=True,
    )
    set_gz_control_library_path = AppendEnvironmentVariable(
        'LD_LIBRARY_PATH',
        harmonic_gz_control_lib,
        prepend=True,
    )
    set_gz_control_ament_path = AppendEnvironmentVariable(
        'AMENT_PREFIX_PATH',
        harmonic_gz_control_prefix,
        prepend=True,
    )

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments=[('gz_args', PythonExpression([
            "'-s -r -v 1 --physics-engine gz-physics-bullet-featherstone-plugin " + world_path + "' if '",
            LaunchConfiguration('headless'),
            "' == 'true' else '-r -v 1 --physics-engine gz-physics-bullet-featherstone-plugin " + world_path + "'"
        ]))]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen'
    )

    # Spawn from /robot_description topic.
    # z=0.08 matches wheel collision radius so wheels rest on ground correctly.
    spawn_robot = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-topic', '/robot_description',
                    '-name', 'mobile_ur3',
                    '-allow_renaming', 'true',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.08',
                ],
                output='screen'
            )
        ]
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

    spawners = TimerAction(
        period=8.0,
        actions=[jsb_spawner, arm_spawner, gripper_spawner]
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/shoulder_pan_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
            '/shoulder_lift_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
            '/elbow_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
            '/wrist_1_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
            '/wrist_2_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
            '/wrist_3_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
            '/finger_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
            # Camera/lidar bridges disabled during RL training — not used by policy
            # TF
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            # World dynamic poses — all models including pickup_object
            '/world/pickplace_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen'
    )

    return LaunchDescription([
        headless_arg,
        set_gz_resource_path,
        set_gz_control_plugin_path,
        set_gz_control_library_path,
        set_gz_control_ament_path,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        spawners,
        bridge,
    ])
