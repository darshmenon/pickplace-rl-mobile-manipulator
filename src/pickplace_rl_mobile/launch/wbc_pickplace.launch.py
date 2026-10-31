#!/usr/bin/env python3
"""Gazebo robot plus the whole-body controller node (classical baseline).

  ros2 launch pickplace_rl_mobile wbc_pickplace.launch.py

Default is arm-only (fixed_base) under bullet-featherstone, the engine gazebo.launch.py
uses: the wheels spin there without moving the robot, so the cube is moved from the
world's (0.6, 0) to (0.5, 0), inside the arm's reach.

drive:=true runs under DART, where the base does drive. The robot spawns at x=-0.8, the
node follows the cube's Gazebo pose and the base closes the distance. Still experimental,
the hand tends to push the cube while it settles. Perception is optional with
use_perception:=true.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    drive = LaunchConfiguration('drive').perform(context).lower() == 'true'
    spawn_x = -0.8 if drive else 0.0
    pkg = get_package_share_directory('pickplace_rl_mobile')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'spawn_x': str(spawn_x),
            'physics': 'gz-physics-dartsim-plugin' if drive else 'gz-physics-bullet-featherstone-plugin',
        }.items(),
    )
    wbc = Node(
        package='pickplace_rl_mobile',
        executable='wbc_node',
        name='wbc_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,  # /clock is not bridged in gazebo.launch.py
            'use_perception': LaunchConfiguration('use_perception'),
            'fixed_base': not drive,
            'base_start': [spawn_x, 0.0, 0.0],
            'fallback_object_pos': [0.5, 0.0, 0.1325],
            'target_pos': [0.5, 0.14, 0.14],
            'gravity_ff': 0.0,
            'descend_tol': 0.010 if drive else 0.004,
            'gt_object': drive,
        }],
    )
    move_cube = ExecuteProcess(cmd=[
        'gz', 'service', '-s', '/world/pickplace_world/set_pose',
        '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
        '--req', 'name: "pickup_object" position {x: 0.5 y: 0 z: 0.1325}'])
    return [
        gazebo,
        TimerAction(period=15.0, actions=[move_cube]),
        # wait for the robot to spawn and the controllers to come up
        TimerAction(period=20.0, actions=[wbc]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_perception', default_value='false'),
        DeclareLaunchArgument('drive', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
