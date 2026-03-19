#!/usr/bin/env python3
"""
Full VLA Pipeline Launch
Starts all 6 VLA nodes in dependency order:
  1. vla_action_node       — MoveIt2 motion execution
  2. vla_vision_node       — OWLv2 / HSV object detection
  3. vla_language_node     — SmolLM2 / regex instruction parsing
  4. object_memory_node    — persistent object map (NEW)
  5. task_planner_node     — multi-step task decomposition (NEW)
  6. vla_coordinator_node  — orchestrator
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_llm',   default_value='true',
                              description='Load SmolLM2-360M-Instruct for language parsing'),
        DeclareLaunchArgument('use_owlv2', default_value='true',
                              description='Load OWLv2 for open-vocabulary detection'),

        # 1 – Action execution (start first so service is ready)
        Node(
            package='pickplace_rl_mobile',
            executable='vla_action_node',
            name='vla_action_node',
            output='screen',
        ),

        # 2 – Vision / detection (wait for camera topics)
        TimerAction(period=2.0, actions=[
            Node(
                package='pickplace_rl_mobile',
                executable='vla_vision_node',
                name='vla_vision_node',
                output='screen',
                parameters=[{
                    'use_owlv2':           LaunchConfiguration('use_owlv2'),
                    'detection_threshold': 0.1,
                }],
            ),
        ]),

        # 3 – Language parser
        TimerAction(period=2.5, actions=[
            Node(
                package='pickplace_rl_mobile',
                executable='vla_language_node',
                name='vla_language_node',
                output='screen',
                parameters=[{'use_llm': LaunchConfiguration('use_llm')}],
            ),
        ]),

        # 4 – Object memory (start before coordinator)
        TimerAction(period=3.0, actions=[
            Node(
                package='pickplace_rl_mobile',
                executable='object_memory_node',
                name='vla_object_memory',
                output='screen',
                parameters=[{'decay_seconds': 30.0}],
            ),
        ]),

        # 5 – Task planner
        TimerAction(period=3.5, actions=[
            Node(
                package='pickplace_rl_mobile',
                executable='task_planner_node',
                name='vla_task_planner',
                output='screen',
            ),
        ]),

        # 6 – Coordinator (last — needs all others running)
        TimerAction(period=5.0, actions=[
            Node(
                package='pickplace_rl_mobile',
                executable='vla_coordinator_node',
                name='vla_coordinator_node',
                output='screen',
            ),
        ]),
    ])
