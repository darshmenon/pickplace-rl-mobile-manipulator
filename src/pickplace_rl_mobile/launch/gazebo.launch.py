#!/usr/bin/env python3

import os
import re
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
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


def generate_launch_description():

    pkg_dir = get_package_share_directory('pickplace_rl_mobile')
    world_path = os.path.join(pkg_dir, 'worlds', 'pickplace_world.world')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'mobile_ur3.urdf')
    ur_description_share = get_package_share_directory('ur_description')
    robotiq_share = get_package_share_directory('robotiq_2f_85_gripper_visualization')

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

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments=[('gz_args', f'-r -v 4 {world_path}')]
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
            # RGBD camera
            '/camera_head/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_head/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_head/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera_head/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # Lidar
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            # TF
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen'
    )

    return LaunchDescription([
        set_gz_resource_path,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
    ])
