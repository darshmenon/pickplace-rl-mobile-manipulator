#!/usr/bin/env python3
"""
Multi-robot navigation with shared map.

Spawns two diff-drive robots (robot1, robot2) in Gazebo, each with its own
namespaced Nav2 stack. A single map_server publishes the pre-built map so
both robots share the same global map frame.

Usage:
  ros2 launch diff_drive_robot multi_robot.launch.py

Optional args:
  map:=<path>   Override default map yaml (default: ~/rosnav/my_map.yaml)
  rviz:=False   Disable RViz
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# Robot spawn poses
# ---------------------------------------------------------------------------
ROBOTS = [
    {'name': 'robot1', 'x': '0.0',  'y': '0.0',  'z': '0.3', 'yaw': '0.0'},
    {'name': 'robot2', 'x': '1.5',  'y': '0.0',  'z': '0.3', 'yaw': '0.0'},
]


def generate_launch_description():
    pkg = 'diff_drive_robot'
    pkg_share = get_package_share_directory(pkg)
    home = os.path.expanduser('~')
    nav2_bringup = get_package_share_directory('nav2_bringup')

    map_file    = LaunchConfiguration('map')
    rviz_flag   = LaunchConfiguration('rviz')
    world_lc    = LaunchConfiguration('world')

    declare_map = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(home, 'rosnav', 'my_map.yaml'),
        description='Shared map yaml file')

    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='True', description='Launch RViz')

    declare_world = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg_share, 'worlds', 'obstacles.world'),
        description='Gazebo world file')

    # -----------------------------------------------------------------------
    # Gazebo
    # -----------------------------------------------------------------------
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': ['-r -s -v1 ', world_lc],
            'on_exit_shutdown': 'true',
        }.items())

    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g'}.items())

    # -----------------------------------------------------------------------
    # Shared map server (one instance, no namespace)
    # -----------------------------------------------------------------------
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': map_file,
        }])

    map_server_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server'],
        }])

    # -----------------------------------------------------------------------
    # Per-robot groups
    # -----------------------------------------------------------------------
    robot_groups = []
    for robot in ROBOTS:
        ns = robot['name']
        x, y, z, yaw = robot['x'], robot['y'], robot['z'], robot['yaw']

        # Robot State Publisher (namespaced)
        rsp = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'rsp.launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'urdf': os.path.join(pkg_share, 'urdf', 'robot.urdf.xacro'),
                'frame_prefix': f'{ns}/',
            }.items())

        # Spawn robot in Gazebo
        spawn = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-topic', f'/{ns}/robot_description',
                '-name', ns,
                '-x', x, '-y', y, '-z', z, '-Y', yaw,
            ],
            output='screen')

        # Gazebo <-> ROS bridge (namespaced topics)
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            namespace=ns,
            arguments=[
                f'/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                f'/{ns}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                f'/{ns}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                f'/{ns}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                f'/{ns}/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            ],
            output='screen')

        # Nav2 per-robot (navigation_launch = planner + controller, no map_server)
        # AMCL uses the shared /map topic
        nav2 = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, 'launch', 'navigation_launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'namespace': ns,          # use_namespace removed in Jazzy
                'params_file': os.path.join(
                    pkg_share, 'config',
                    f'nav2_multirobot_params_{ns[-1]}.yaml'),
            }.items())

        group = GroupAction([
            PushRosNamespace(ns),
            rsp,
            spawn,
            bridge,
            TimerAction(period=3.0, actions=[nav2]),   # wait for robot to spawn
        ])
        robot_groups.append(group)

    # -----------------------------------------------------------------------
    # RViz (single instance showing both robots)
    # -----------------------------------------------------------------------
    rviz = GroupAction(
        condition=IfCondition(rviz_flag),
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg_share, 'rviz', 'bot.rviz')],
            output='screen')])

    return LaunchDescription([
        declare_map,
        declare_rviz,
        declare_world,
        gz_server,
        gz_client,
        map_server,
        map_server_lifecycle,
        rviz,
        *robot_groups,
    ])
