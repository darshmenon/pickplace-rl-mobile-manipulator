#!/usr/bin/env python3
"""Gazebo robot plus the whole-body controller node (classical baseline).

  ros2 launch pickplace_rl_mobile wbc_pickplace.launch.py

Gazebo's wheels have no traction on this robot, so the node reaches with the arm
alone (fixed_base) and the cube is moved from the world's (0.6, 0) to (0.5, 0),
inside the arm's reach. Perception is optional with use_perception:=true.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('pickplace_rl_mobile')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'gazebo.launch.py')),
        launch_arguments={'headless': LaunchConfiguration('headless'), 'spawn_x': '-0.8',
                          'physics': LaunchConfiguration('physics')}.items(),
    )
    wbc = Node(
        package='pickplace_rl_mobile',
        executable='wbc_node',
        name='wbc_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,  # /clock is not bridged in gazebo.launch.py
            'use_perception': LaunchConfiguration('use_perception'),
            'fixed_base': LaunchConfiguration('fixed_base'),
            'base_start': [-0.8, 0.0, 0.0],
            'fallback_object_pos': [0.5, 0.0, 0.1325],
            'target_pos': [0.5, 0.14, 0.14],
            'gravity_ff': 0.0,
            'descend_tol': 0.010,
            'gt_object': True,
        }],
    )
    move_cube = ExecuteProcess(cmd=[
        'gz', 'service', '-s', '/world/pickplace_world/set_pose',
        '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
        '--req', 'name: "pickup_object" position {x: 0.5 y: 0 z: 0.1325}'])
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_perception', default_value='false'),
        DeclareLaunchArgument('fixed_base', default_value='false'),
        DeclareLaunchArgument('physics', default_value='gz-physics-dartsim-plugin'),
        gazebo,
        TimerAction(period=15.0, actions=[move_cube]),
        # wait for the robot to spawn and the controllers to come up
        TimerAction(period=20.0, actions=[wbc]),
    ])
