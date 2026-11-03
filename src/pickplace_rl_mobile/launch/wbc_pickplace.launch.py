#!/usr/bin/env python3
"""Gazebo robot plus the whole-body controller node (classical baseline).

  ros2 launch pickplace_rl_mobile wbc_pickplace.launch.py

Default is arm-only (fixed_base) under bullet-featherstone, the engine gazebo.launch.py
uses: the stock base does not drive there, so the cube is moved from the world's (0.6, 0)
to (0.5, 0), inside the arm's reach.

drive:=true spawns the robot at x=-0.8 with a drive-capable base (mobile_ur3_drive.urdf,
heavier wheels so they grip under Featherstone), drives it up to the cube, parks it and then
runs the same arm-only pick. Perception is optional with use_perception:=true.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    delay = float(os.environ.get('SPAWN_DELAY', '5.0')) - 5.0  # see gazebo.launch.py
    drive = LaunchConfiguration('drive').perform(context).lower() == 'true'
    spawn_x = -0.8 if drive else 0.0
    # drive mode needs wheels that grip under Featherstone (lighter chassis, heavier wheels)
    os.environ['MOBILE_UR3_URDF'] = 'mobile_ur3_drive.urdf' if drive else 'mobile_ur3.urdf'
    pkg = get_package_share_directory('pickplace_rl_mobile')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': LaunchConfiguration('world'),
            'spawn_x': str(spawn_x),
            'physics': 'gz-physics-bullet-featherstone-plugin' if drive else 'gz-physics-bullet-featherstone-plugin',
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
            'fixed_base': True,
            'base_start': [spawn_x, 0.0, 0.0],
            'fallback_object_pos': [0.5, 0.0, 0.1325],
            'target_pos': [0.5, 0.14, 0.14],
            'gravity_ff': 0.0,
            'descend_tol': 0.008 if drive else 0.004,
        }],
    )
    # retry: the call times out when the sim is still busy loading, which leaves the cube out of reach
    move_cube = ExecuteProcess(cmd=['bash', '-c',
        'for i in $(seq 1 10); do gz service -s /world/pickplace_world/set_pose '
        '--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 3000 '
        '--req \'name: "pickup_object" position {x: 0.5 y: 0 z: 0.1325}\' | grep -q "data: true" && exit 0; '
        'sleep 2; done'])
    actions = [gazebo, TimerAction(period=15.0 + delay, actions=[move_cube])]
    if drive:
        # drive the base up to the cube, then run the arm-only pick with the base parked
        drive_to = ExecuteProcess(cmd=['python3', '-m', 'pickplace_rl_mobile.drive_to', '--x', '0.78'],
                                  output='screen')
        actions += [
            TimerAction(period=14.0 + delay, actions=[drive_to]),
            RegisterEventHandler(OnProcessExit(target_action=drive_to,
                                               on_exit=[TimerAction(period=3.0, actions=[wbc])])),
        ]
    else:
        # wait for the robot to spawn and the controllers to come up
        actions.append(TimerAction(period=20.0 + delay, actions=[wbc]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_perception', default_value='false'),
        DeclareLaunchArgument('drive', default_value='false'),
        DeclareLaunchArgument('world', default_value='pickplace_world.world'),
        OpaqueFunction(function=_setup),
    ])
