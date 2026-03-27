#!/usr/bin/env python3
"""
Multi-world parallel RL training launch file.

Spawns N_WORLDS isolated Gazebo instances (each with its own GZ_PARTITION),
bridges each to namespaced ROS topics (/world_0/, /world_1/, ...), and starts
the training node with --n-envs matching the number of worlds.

Each world is isolated via GZ_PARTITION so they don't interfere at the
Gz transport layer. The ros_gz_bridge for each world maps:
  ROS /world_N/<topic>  <->  Gz /<topic>  (within partition world_N)

Usage:
  ros2 launch pickplace_rl_mobile multi_world_training.launch.py
  ros2 launch pickplace_rl_mobile multi_world_training.launch.py n_worlds:=2
"""

import os
import re
import shutil
import yaml
from os import environ, pathsep

from catkin_pkg.package import InvalidPackage, PACKAGE_MANIFEST_FILENAME, parse_package
from ros2pkg.api import get_package_names
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def resolve_package_uris(urdf_str):
    def replace(match):
        pkg = match.group(1)
        rel = match.group(2)
        try:
            share = get_package_share_directory(pkg)
            return f'file://{share}/{rel}'
        except Exception:
            return match.group(0)
    return re.sub(r'package://([^/]+)/([^"\'>\s]+)', replace, urdf_str)


def _gz_paths():
    """Replicate GazeboRosPaths.get_paths() from gz_sim.launch.py."""
    model_paths, plugin_paths = [], []
    for pkg in get_package_names():
        share = get_package_share_directory(pkg)
        manifest = os.path.join(share, PACKAGE_MANIFEST_FILENAME)
        if not os.path.isfile(manifest):
            continue
        try:
            package = parse_package(manifest)
        except InvalidPackage:
            continue
        for export in package.exports:
            if export.tagname == 'gazebo_ros':
                if 'gazebo_model_path' in export.attributes:
                    p = export.attributes['gazebo_model_path'].replace('${prefix}', share)
                    model_paths.append(p)
                if 'plugin_path' in export.attributes:
                    p = export.attributes['plugin_path'].replace('${prefix}', share)
                    plugin_paths.append(p)
    return pathsep.join(model_paths), pathsep.join(plugin_paths)


def _write_bridge_config(ns, config_path):
    """Write a ros_gz_bridge YAML config that maps namespaced ROS topics to bare Gz topics."""
    joint_topics = [
        'shoulder_pan_joint/cmd_vel',
        'shoulder_lift_joint/cmd_vel',
        'elbow_joint/cmd_vel',
        'wrist_1_joint/cmd_vel',
        'wrist_2_joint/cmd_vel',
        'wrist_3_joint/cmd_vel',
        'finger_joint/cmd_vel',
    ]
    entries = [
        {
            'ros_topic_name': f'/{ns}/cmd_vel',
            'gz_topic_name': '/cmd_vel',
            'ros_type_name': 'geometry_msgs/msg/Twist',
            'gz_type_name': 'gz.msgs.Twist',
            'direction': 'ROS_TO_GZ',
        },
        {
            'ros_topic_name': f'/{ns}/odom',
            'gz_topic_name': '/odom',
            'ros_type_name': 'nav_msgs/msg/Odometry',
            'gz_type_name': 'gz.msgs.Odometry',
            'direction': 'GZ_TO_ROS',
        },
        {
            'ros_topic_name': f'/{ns}/joint_states',
            'gz_topic_name': '/joint_states',
            'ros_type_name': 'sensor_msgs/msg/JointState',
            'gz_type_name': 'gz.msgs.Model',
            'direction': 'GZ_TO_ROS',
        },
    ]
    for jt in joint_topics:
        entries.append({
            'ros_topic_name': f'/{ns}/{jt}',
            'gz_topic_name': f'/{jt}',
            'ros_type_name': 'std_msgs/msg/Float64',
            'gz_type_name': 'gz.msgs.Double',
            'direction': 'ROS_TO_GZ',
        })

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(entries, f)


def launch_setup(context, *args, **kwargs):
    n_worlds = int(LaunchConfiguration('n_worlds').perform(context))

    pkg_dir = get_package_share_directory('pickplace_rl_mobile')
    world_path = os.path.join(pkg_dir, 'worlds', 'pickplace_world.world')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'mobile_ur3.urdf')

    ur_description_share = get_package_share_directory('ur_description')
    robotiq_share = get_package_share_directory('robotiq_2f_85_gripper_visualization')

    model_paths, ros_plugin_paths = _gz_paths()

    gz_resource_path = pathsep.join(filter(None, [
        os.path.join(ur_description_share, '..'),
        os.path.join(robotiq_share, '..'),
        environ.get('GZ_SIM_RESOURCE_PATH', ''),
        environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
        model_paths,
    ]))

    # Match exactly what gz_sim.launch.py sets so system plugins load correctly
    gz_plugin_path = pathsep.join(filter(None, [
        environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
        environ.get('LD_LIBRARY_PATH', ''),
        ros_plugin_paths,
    ]))
    ign_plugin_path = pathsep.join(filter(None, [
        environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),
        environ.get('LD_LIBRARY_PATH', ''),
        ros_plugin_paths,
    ]))

    with open(urdf_path, 'r') as f:
        robot_description = resolve_package_uris(f.read())

    # gz executable path (same as gz_sim.launch.py uses ruby gz sim)
    gz_exec = shutil.which('gz') or 'gz'

    actions = []

    for i in range(n_worlds):
        ns = f'world_{i}'
        partition = f'world_{i}'
        gz_env = {
            'GZ_PARTITION': partition,
            'GZ_SIM_RESOURCE_PATH': gz_resource_path,
            'IGN_GAZEBO_RESOURCE_PATH': gz_resource_path,
            'GZ_SIM_SYSTEM_PLUGIN_PATH': gz_plugin_path,
            'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': ign_plugin_path,
        }

        # --- Gazebo server (match gz_sim.launch.py: ruby gz sim ... shell=True) ---
        gz_sim = ExecuteProcess(
            cmd=[f'ruby {gz_exec} sim -r -v 4 {world_path} --force-version 7'],
            additional_env=gz_env,
            output='screen',
            shell=True,
            name=f'gz_sim_{ns}',
        )

        # --- Robot state publisher (publishes /{ns}/robot_description) ---
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=ns,
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen',
        )

        # --- Spawn robot (staggered; bash -c ensures GZ_PARTITION is set) ---
        spawn_cmd = (
            f'GZ_PARTITION={partition} '
            f'ros2 run ros_gz_sim create '
            f'-topic /{ns}/robot_description '
            f'-name mobile_ur3 '
            f'-allow_renaming true '
            f'-x 0.0 -y 0.0 -z 0.08'
        )
        spawn = TimerAction(
            period=10.0 + i * 5.0,
            actions=[
                ExecuteProcess(
                    cmd=['bash', '-c', spawn_cmd],
                    output='screen',
                )
            ],
        )

        # --- Bridge (namespaced ROS <-> bare Gz topics within this partition) ---
        bridge_config = f'/tmp/pickplace_bridge_{ns}.yaml'
        _write_bridge_config(ns, bridge_config)

        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': bridge_config}],
            additional_env={'GZ_PARTITION': partition},
            output='screen',
            name=f'bridge_{ns}',
        )

        actions.extend([gz_sim, rsp, spawn, bridge])

    # --- Training node (wait for all worlds to be ready) ---
    startup_delay = 20.0 + n_worlds * 5.0
    train_node = TimerAction(
        period=startup_delay,
        actions=[
            Node(
                package='pickplace_rl_mobile',
                executable='train_rl',
                name='rl_train_node',
                arguments=['--n-envs', str(n_worlds)],
                output='screen',
            )
        ],
    )
    actions.append(train_node)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'n_worlds',
            default_value='4',
            description='Number of parallel Gazebo worlds for training',
        ),
        OpaqueFunction(function=launch_setup),
    ])
