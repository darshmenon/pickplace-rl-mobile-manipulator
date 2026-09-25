#!/usr/bin/env bash
# Usage:
#   ./run_rl_training.sh                                  # resume best if available, GUI
#   ./run_rl_training.sh --headless                       # resume best if available, headless
#   ./run_rl_training.sh --fresh                          # fresh run, GUI
#   ./run_rl_training.sh --resume-latest --headless       # resume latest numbered checkpoint
#   ./run_rl_training.sh ./rl_models/best_model.zip       # resume explicit checkpoint
#   ./run_rl_training.sh --curriculum-stage 2 --timesteps 200000 --headless
#   ./run_rl_training.sh --algo ppo --policy-arch transformer --headless
#   ./run_rl_training.sh --world pickplace_world_obstacles.world --headless
#   ./run_rl_training.sh --adaptive-curriculum --headless
#   ./run_rl_training.sh --adaptive-domain-randomization --headless
#   ./run_rl_training.sh --fast --headless

set -eo pipefail

export PYTHONUNBUFFERED=1
export ORIGINAL_HOME="${HOME:-/home/asimov}"

source /opt/ros/humble/setup.bash
source install/setup.bash

RUNTIME_ROOT="${PICKPLACE_RUNTIME_ROOT:-/tmp/pickplace_headless_runtime}"
mkdir -p "$RUNTIME_ROOT/home" "$RUNTIME_ROOT/roslogs" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/xdg"

export HOME="$RUNTIME_ROOT/home"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$RUNTIME_ROOT/roslogs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_ROOT/mpl}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$RUNTIME_ROOT/xdg}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
export IGN_IP="${IGN_IP:-127.0.0.1}"

USER_SITE_PACKAGES="$(
python3 - <<'PY'
import os
import sys
major, minor = sys.version_info[:2]
original_home = os.environ.get('ORIGINAL_HOME', '')
if original_home:
    print(os.path.join(original_home, '.local', 'lib', f'python{major}.{minor}', 'site-packages'))
PY
)"
if [ -n "$USER_SITE_PACKAGES" ] && [ -d "$USER_SITE_PACKAGES" ]; then
    export PYTHONPATH="$USER_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi

MODEL_PATH=""
HEADLESS=false
TIMESTEPS=500000
CURRICULUM_STAGE=0
SAVE_DIR="./rl_models"
RESUME_POLICY="best"
ALGO="tqc"
POLICY_ARCH="mlp"
ADAPTIVE_CURRICULUM=false
ADAPTIVE_DOMAIN_RANDOMIZATION=false
ADR_STEP=0.15
EVAL_FREQ=10000
N_EVAL_EPISODES=10
CHECKPOINT_FREQ=10000
GRADIENT_STEPS=""
WORLD="pickplace_world.world"
# Overridable via env so a second concurrent run (e.g. a different --policy-arch
# experiment) can use a disjoint ROS domain / Gazebo transport partition.
TRAIN_DOMAIN="${PICKPLACE_DOMAIN_BASE:-20}"
TRAIN_PARTITION="${PICKPLACE_TRAIN_PARTITION:-sim_0}"
EVAL_DOMAIN=$((TRAIN_DOMAIN + 1))
EVAL_PARTITION="${PICKPLACE_EVAL_PARTITION:-sim_1}"
GAZEBO_PID=""
EVAL_GAZEBO_PID=""
TRAIN_PID=""

cleanup() {
    for pid in "$TRAIN_PID" "$EVAL_GAZEBO_PID" "$GAZEBO_PID"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait "$TRAIN_PID" "$EVAL_GAZEBO_PID" "$GAZEBO_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

latest_numbered_checkpoint() {
    find "$SAVE_DIR" -maxdepth 1 -type f -name 'pickplace_model_*_steps.zip' | sort -V | tail -n 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --headless)
            HEADLESS=true
            ;;
        --fresh)
            RESUME_POLICY="fresh"
            MODEL_PATH=""
            ;;
        --resume-best)
            RESUME_POLICY="best"
            ;;
        --resume-latest)
            RESUME_POLICY="latest"
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift
            ;;
        --curriculum-stage)
            CURRICULUM_STAGE="$2"
            shift
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift
            ;;
        --algo)
            ALGO="$2"
            shift
            ;;
        --policy-arch)
            POLICY_ARCH="$2"
            shift
            ;;
        --adaptive-curriculum)
            ADAPTIVE_CURRICULUM=true
            ;;
        --adaptive-domain-randomization)
            ADAPTIVE_DOMAIN_RANDOMIZATION=true
            ;;
        --adr-step)
            ADR_STEP="$2"
            shift
            ;;
        --fast)
            EVAL_FREQ=50000
            N_EVAL_EPISODES=2
            CHECKPOINT_FREQ=50000
            GRADIENT_STEPS=1
            ;;
        --eval-freq)
            EVAL_FREQ="$2"
            shift
            ;;
        --n-eval-episodes)
            N_EVAL_EPISODES="$2"
            shift
            ;;
        --checkpoint-freq)
            CHECKPOINT_FREQ="$2"
            shift
            ;;
        --gradient-steps)
            GRADIENT_STEPS="$2"
            shift
            ;;
        --world)
            WORLD="$2"
            shift
            ;;
        *)
            MODEL_PATH="$1"
            RESUME_POLICY="explicit"
            ;;
    esac
    shift
done

if [ -z "$MODEL_PATH" ]; then
    if [ "$RESUME_POLICY" = "best" ] && [ -f "$SAVE_DIR/best_model/best_model.zip" ]; then
        MODEL_PATH="$SAVE_DIR/best_model/best_model.zip"
    elif [ "$RESUME_POLICY" = "latest" ]; then
        MODEL_PATH="$(latest_numbered_checkpoint)"
    fi
fi

if [ -n "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "[run_rl_training] Requested checkpoint not found: $MODEL_PATH" >&2
    exit 1
fi

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

if [ -n "$MODEL_PATH" ]; then
    echo "[run_rl_training] Resuming from $MODEL_PATH"
else
    echo "[run_rl_training] Starting fresh training run"
fi
echo "[run_rl_training] timesteps=$TIMESTEPS curriculum_stage=$CURRICULUM_STAGE save_dir=$SAVE_DIR headless=$HEADLESS"
echo "[run_rl_training] algo=$ALGO policy_arch=$POLICY_ARCH adaptive_curriculum=$ADAPTIVE_CURRICULUM world=$WORLD"
echo "[run_rl_training] adaptive_domain_randomization=$ADAPTIVE_DOMAIN_RANDOMIZATION adr_step=$ADR_STEP"
echo "[run_rl_training] eval_freq=$EVAL_FREQ n_eval_episodes=$N_EVAL_EPISODES checkpoint_freq=$CHECKPOINT_FREQ gradient_steps=${GRADIENT_STEPS:-default}"
echo "[run_rl_training] train world: ROS_DOMAIN_ID=$TRAIN_DOMAIN GZ_PARTITION=$TRAIN_PARTITION"
echo "[run_rl_training] eval world:  ROS_DOMAIN_ID=$EVAL_DOMAIN GZ_PARTITION=$EVAL_PARTITION (headless)"
echo "[run_rl_training] runtime root: $RUNTIME_ROOT"

env ROS_DOMAIN_ID=$TRAIN_DOMAIN GZ_PARTITION=$TRAIN_PARTITION \
    ros2 launch pickplace_rl_mobile gazebo.launch.py headless:=$HEADLESS world:="$WORLD" &
GAZEBO_PID=$!

env ROS_DOMAIN_ID=$EVAL_DOMAIN GZ_PARTITION=$EVAL_PARTITION \
    ros2 launch pickplace_rl_mobile gazebo.launch.py headless:=true world:="$WORLD" &
EVAL_GAZEBO_PID=$!

sleep 8

# ros2 launch's CLI parser rejects a value-less 'name:=' argument, so
# gradient_steps (empty by default — rl_train.launch.py's own default
# already means "use algorithm/resume setting") must only be forwarded
# when the user actually set --gradient-steps or --fast.
RL_TRAIN_ARGS=(
    timesteps:="$TIMESTEPS"
    save_dir:="$SAVE_DIR"
    curriculum_stage:="$CURRICULUM_STAGE"
    algo:="$ALGO"
    policy_arch:="$POLICY_ARCH"
    adaptive_curriculum:="$ADAPTIVE_CURRICULUM"
    adaptive_domain_randomization:="$ADAPTIVE_DOMAIN_RANDOMIZATION"
    adr_step:="$ADR_STEP"
    eval_freq:="$EVAL_FREQ"
    n_eval_episodes:="$N_EVAL_EPISODES"
    checkpoint_freq:="$CHECKPOINT_FREQ"
)
if [ -n "$GRADIENT_STEPS" ]; then
    RL_TRAIN_ARGS+=(gradient_steps:="$GRADIENT_STEPS")
fi
if [ -n "$MODEL_PATH" ]; then
    RL_TRAIN_ARGS+=(load_model:="$MODEL_PATH")
fi

env ROS_DOMAIN_ID=$TRAIN_DOMAIN GZ_PARTITION=$TRAIN_PARTITION \
    PICKPLACE_DOMAIN_BASE=$TRAIN_DOMAIN PICKPLACE_TRAIN_PARTITION=$TRAIN_PARTITION \
    PICKPLACE_EVAL_PARTITION=$EVAL_PARTITION \
ros2 launch pickplace_rl_mobile rl_train.launch.py "${RL_TRAIN_ARGS[@]}" &
TRAIN_PID=$!

wait
