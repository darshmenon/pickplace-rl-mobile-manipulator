#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source install/setup.bash

MODEL_PATH=${1:-""}

ros2 launch pickplace_rl_mobile gazebo.launch.py &

if [ -n "$MODEL_PATH" ]; then
    ros2 launch pickplace_rl_mobile rl_train.launch.py load_model:="$MODEL_PATH" &
else
    ros2 launch pickplace_rl_mobile rl_train.launch.py &
fi

wait
