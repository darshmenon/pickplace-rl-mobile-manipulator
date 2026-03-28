#!/usr/bin/env bash
# run_multi_training.sh [N_WORLDS]
# Launches N headless Gazebo worlds (each with its own ROS_DOMAIN_ID + GZ_PARTITION)
# then starts TQC training with SubprocVecEnv across all worlds.
#
# Usage:
#   ./src/pickplace_rl_mobile/launch/run_multi_training.sh        # 2 worlds (default)
#   ./src/pickplace_rl_mobile/launch/run_multi_training.sh 3      # 3 worlds
#
# Each world uses ROS_DOMAIN_ID = 20+i and GZ_PARTITION = sim_i.
# This gives complete ROS2 + Gz transport isolation with no namespace hacks.

set -e

N_WORLDS=${1:-2}
LOAD_MODEL=${2:-""}

if [ ! -d "install" ]; then
    echo "ERROR: run from workspace root"
    exit 1
fi

source /opt/ros/humble/setup.bash
source install/setup.bash

PKG_SHARE=$(ros2 pkg prefix pickplace_rl_mobile)/share/pickplace_rl_mobile
WORLD_FILE="$PKG_SHARE/worlds/pickplace_world.world"
URDF_FILE="$PKG_SHARE/urdf/mobile_ur3.urdf"

# Resolve package:// URIs in URDF once, write to temp file
RESOLVED_URDF=/tmp/mobile_ur3_resolved.urdf
export URDF_FILE RESOLVED_URDF
python3 - <<'PYEOF'
import re, os
from ament_index_python.packages import get_package_share_directory
def repl(m):
    try:
        share = get_package_share_directory(m.group(1))
        return f'file://{share}/{m.group(2)}'
    except Exception:
        return m.group(0)
urdf = open(os.environ['URDF_FILE']).read()
resolved = re.sub(r'package://([^/]+)/([^"\'>\s]+)', repl, urdf)
open(os.environ['RESOLVED_URDF'], 'w').write(resolved)
print(f"Resolved URDF written to {os.environ['RESOLVED_URDF']}")
PYEOF

# Build resource paths
UR_SHARE=$(ros2 pkg prefix ur_description)/share
ROBOTIQ_SHARE=$(ros2 pkg prefix robotiq_2f_85_gripper_visualization)/share
GZ_RESOURCE="$UR_SHARE/..:$ROBOTIQ_SHARE/..:${GZ_SIM_RESOURCE_PATH:-}"

BRIDGE_TOPICS=(
    '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'
    '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry'
    '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model'
    '/shoulder_pan_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
    '/shoulder_lift_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
    '/elbow_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
    '/wrist_1_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
    '/wrist_2_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
    '/wrist_3_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
    '/finger_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
    '/world/pickplace_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
)

echo "Launching $N_WORLDS Gazebo worlds (world 0 with GUI, rest headless)..."

for i in $(seq 0 $((N_WORLDS - 1))); do
    DOMAIN=$((i + 20))
    PARTITION="sim_$i"
    echo "  World $i: ROS_DOMAIN_ID=$DOMAIN  GZ_PARTITION=$PARTITION"

    # World 0 gets GUI so you can watch; rest are headless servers (-s)
    if [ "$i" -eq 0 ]; then
        GZ_FLAGS="-r -v 1"
    else
        GZ_FLAGS="-s -r -v 1"
    fi

    ROS_DOMAIN_ID=$DOMAIN GZ_PARTITION=$PARTITION \
    GZ_SIM_RESOURCE_PATH="$GZ_RESOURCE" IGN_GAZEBO_RESOURCE_PATH="$GZ_RESOURCE" \
        gz sim $GZ_FLAGS "$WORLD_FILE" \
        &> /tmp/gz_world_${i}.log &

    sleep 10  # Wait for physics server to be ready

    # Robot state publisher (publishes /robot_description in this domain)
    ROS_DOMAIN_ID=$DOMAIN GZ_PARTITION=$PARTITION \
        ros2 run robot_state_publisher robot_state_publisher \
        --ros-args \
        -p "robot_description:=$(cat $RESOLVED_URDF)" \
        -p use_sim_time:=true \
        &> /tmp/rsp_${i}.log &

    sleep 3

    # Spawn robot into this world
    ROS_DOMAIN_ID=$DOMAIN GZ_PARTITION=$PARTITION \
        ros2 run ros_gz_sim create \
        -topic /robot_description \
        -name mobile_ur3 \
        -allow_renaming true \
        -x 0.0 -y 0.0 -z 0.08 \
        &> /tmp/spawn_${i}.log &

    sleep 5

    # Bridge: maps Gz topics <-> ROS topics in this domain
    ROS_DOMAIN_ID=$DOMAIN GZ_PARTITION=$PARTITION \
        ros2 run ros_gz_bridge parameter_bridge \
        "${BRIDGE_TOPICS[@]}" \
        &> /tmp/bridge_${i}.log &

    sleep 2
done

echo ""
echo "All $N_WORLDS worlds started. Waiting 15s for bridges to settle..."
sleep 15

echo "Starting TQC training with $N_WORLDS envs..."
ros2 run pickplace_rl_mobile train_rl \
    --ros-args -- --n-envs "$N_WORLDS" --save-dir ./rl_models --load-model "$LOAD_MODEL" &> /tmp/training.log &

wait
