#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_dir = get_package_share_directory('pickplace_rl_mobile')
    world_path = os.path.join(pkg_dir, 'worlds', 'pickplace_world.world')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'mobile_ur3.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '--verbose', '--render-engine', 'ogre2', world_path],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    # Delay spawn to let Gazebo fully start
    spawn_robot = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-world', 'pickplace_world',
                    '-name', 'mobile_ur3',
                    '-file', urdf_path,
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.05',
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
        ],
        output='screen'
    )

    return LaunchDescription([
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
    ])
