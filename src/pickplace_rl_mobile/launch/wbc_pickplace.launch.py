#!/usr/bin/env python3
"""Gazebo robot plus the whole-body controller node (classical baseline).

  ros2 launch pickplace_rl_mobile wbc_pickplace.launch.py

The cube pose comes from the world file (0.6, 0, 0.1325) unless perception is
enabled with use_perception:=true.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('pickplace_rl_mobile')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'gazebo.launch.py')),
        launch_arguments={'headless': LaunchConfiguration('headless')}.items(),
    )
    wbc = Node(
        package='pickplace_rl_mobile',
        executable='wbc_node',
        name='wbc_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,  # /clock is not bridged in gazebo.launch.py
            'use_perception': LaunchConfiguration('use_perception'),
            'fallback_object_pos': [0.6, 0.0, 0.1325],
            'target_pos': [0.6, 0.5, 0.15],
            'gravity_ff': 0.0,
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_perception', default_value='false'),
        gazebo,
        # wait for the robot to spawn and the controllers to come up
        TimerAction(period=20.0, actions=[wbc]),
    ])
