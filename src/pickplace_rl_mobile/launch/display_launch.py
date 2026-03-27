#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Get the package directory
    pkg_dir = get_package_share_directory('pickplace_rl_mobile')
    
    # Get the URDF file path
    urdf_file = os.path.join(pkg_dir, 'urdf', 'mobile_ur3.urdf')
    
    # Use Command to process URDF (in case there are any xacro macros)
    robot_description_content = Command(['cat ', urdf_file])
    robot_description = {'robot_description': robot_description_content}

    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
        arguments=['--ros-args', '--log-level', 'info']
    )

    # Joint state publisher node (publishes zero positions for all joints)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # Joint state publisher GUI node (for manual control with sliders) 
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        parameters=[robot_description]
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['--ros-args', '--log-level', 'warn']
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node, 
        rviz_node,
    ])