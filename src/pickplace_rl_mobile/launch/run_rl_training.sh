#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch pickplace_rl_mobile gazebo.launch.py &
ros2 launch pickplace_rl_mobile rl_train.launch.py &

wait