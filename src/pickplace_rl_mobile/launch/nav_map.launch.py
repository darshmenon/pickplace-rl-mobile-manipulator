#!/usr/bin/env python3
"""Mapping and navigation for the mobile base, reusing rosnav's slam_toolbox settings.

Run next to wbc_pickplace.launch.py drive:=true (or gazebo.launch.py with mobile_ur3_vel.urdf) so
/scan, /odom, tf and /clock exist. Needs the Gazebo sensors system in the world for lidar scans.

  ros2 launch pickplace_rl_mobile nav_map.launch.py

slam_toolbox builds the map (settings from rosnav_bot/config/mapper_params_online_async.yaml when
rosnav is installed, otherwise slam_toolbox's defaults); Nav2 plans on it with this package's
config/nav2_params.yaml (0.25 m radius, 0.3 m/s).
"""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _slam_params(pkg):
    try:
        rosnav = get_package_share_directory('rosnav_bot')
        path = os.path.join(rosnav, 'config', 'mapper_params_online_async.yaml')
        if os.path.exists(path):
            return path
    except PackageNotFoundError:
        pass
    return os.path.join(get_package_share_directory('slam_toolbox'), 'config', 'mapper_params_online_async.yaml')


def generate_launch_description():
    pkg = get_package_share_directory('pickplace_rl_mobile')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')),
        launch_arguments={'use_sim_time': use_sim_time, 'slam_params_file': _slam_params(pkg)}.items())
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
        launch_arguments={'use_sim_time': use_sim_time,
                          'params_file': os.path.join(pkg, 'config', 'nav2_params.yaml')}.items())
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        slam,
        nav,
    ])
