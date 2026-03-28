#!/usr/bin/env bash
# Launches multiple independent RL training instances in parallel
# Each instance runs in its own ROS domain and Gazebo partition to avoid topic collisions.

NUM_ENVS=${1:-4} # Default to 4 parallel environments
LOAD_MODEL=${2:-""} # Optional model path to resume training

if [ ! -d "install" ]; then
    echo "Error: Must be run from ros2 workspace root (e.g., ./src/pickplace_rl_mobile/launch/run_parallel_training.sh)"
    exit 1
fi

echo "Starting $NUM_ENVS parallel mobile manipulator training environments..."
if [ -n "$LOAD_MODEL" ]; then
    echo "Resuming training from: $LOAD_MODEL"
fi

for ((i=1; i<=NUM_ENVS; i++)); do
    export ROS_DOMAIN_ID=$i
    export GZ_PARTITION="sim_part_$i"
    
    # We suppress display/RViz for parallel headless training to save GPU
    # and run the environment and RL nodes in headless background processes
    echo "Starting worker $i on ROS_DOMAIN_ID=$i"
    
    if [ -n "$LOAD_MODEL" ]; then
        ros2 launch pickplace_rl_mobile rl_train.launch.py load_model:="$LOAD_MODEL" > /dev/null 2>&1 &
    else
        ros2 launch pickplace_rl_mobile rl_train.launch.py > /dev/null 2>&1 &
    fi
    
    # Stagger launch to prevent massive CPU spikes at exactly the same millisecond
    sleep 5
done

echo "All $NUM_ENVS environments launched in the background!"
echo "Check tensorboard logs with: tensorboard --logdir ./rl_models/tensorboard"
echo "Press Ctrl+C to terminate all parallel environments."

# Trap Ctrl+C to kill all background jobs spawned by this script
trap 'echo "Terminating all environments..."; kill $(jobs -p); exit' SIGINT SIGTERM
wait
