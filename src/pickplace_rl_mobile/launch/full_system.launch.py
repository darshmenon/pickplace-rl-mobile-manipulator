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
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument, GroupAction, TimerAction
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    from launch.substitutions import Command
    from launch_ros.parameter_descriptions import ParameterValue

    pkg_dir = get_package_share_directory('pickplace_rl_mobile')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'mobile_ur3.urdf')
    world_file = os.path.join(pkg_dir, 'worlds', 'pickplace_world.world')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    robot_description_content = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

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
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
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
            '-name', 'pickplace_robot',
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.5', '-z', '0.1',
            '-J', 'shoulder_pan_joint', '0.0',
            '-J', 'shoulder_lift_joint', '-1.57',
            '-J', 'elbow_joint', '1.57',
            '-J', 'wrist_1_joint', '-1.57',
            '-J', 'wrist_2_joint', '-1.57',
            '-J', 'wrist_3_joint', '0.0'
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
        actions=[spawn_robot, bridge, perception_node, safety_guard, rl_node, nav2_group],
    )

    return LaunchDescription([
        use_nav2_arg,
        use_rl_arg,
        model_path_arg,
        use_perception_arg,
        gazebo,
        robot_state_publisher,
        delayed,
    ])
