#!/usr/bin/env bash
# Usage:
#   ./run_rl_training.sh                          # fresh, GUI
#   ./run_rl_training.sh --headless               # fresh, headless (faster fps)
#   ./run_rl_training.sh ./rl_models/best_model.zip          # resume, GUI
#   ./run_rl_training.sh ./rl_models/best_model.zip --headless  # resume, headless

source /opt/ros/humble/setup.bash
source install/setup.bash

MODEL_PATH=""
HEADLESS=false

for arg in "$@"; do
    if [ "$arg" = "--headless" ]; then
        HEADLESS=true
    else
        MODEL_PATH="$arg"
    fi
done

ros2 launch pickplace_rl_mobile gazebo.launch.py headless:=$HEADLESS &

if [ -n "$MODEL_PATH" ]; then
    ros2 launch pickplace_rl_mobile rl_train.launch.py load_model:="$MODEL_PATH" &
else
    ros2 launch pickplace_rl_mobile rl_train.launch.py &
fi

wait
