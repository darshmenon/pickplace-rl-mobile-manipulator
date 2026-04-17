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

# Fall back to headless mode when no usable GUI display is available.
# This keeps training launchable from remote shells, CI, and sandboxed sessions.
if [ "$HEADLESS" = false ]; then
    if [ -z "${DISPLAY:-}" ]; then
        echo "[run_rl_training] DISPLAY is not set; launching headless instead."
        HEADLESS=true
    elif ! command -v xdpyinfo >/dev/null 2>&1; then
        echo "[run_rl_training] xdpyinfo not found; keeping GUI launch request as-is."
    elif ! xdpyinfo >/dev/null 2>&1; then
        echo "[run_rl_training] DISPLAY '$DISPLAY' is not reachable; launching headless instead."
        HEADLESS=true
    fi
fi

ros2 launch pickplace_rl_mobile gazebo.launch.py headless:=$HEADLESS &

if [ -n "$MODEL_PATH" ]; then
    ros2 launch pickplace_rl_mobile rl_train.launch.py load_model:="$MODEL_PATH" &
else
    ros2 launch pickplace_rl_mobile rl_train.launch.py &
fi

wait
