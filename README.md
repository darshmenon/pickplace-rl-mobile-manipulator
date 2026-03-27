# ARES: UR3 Mobile Robot RL Pick and Place

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![SB3-Contrib](https://img.shields.io/badge/SB3--Contrib-TQC-purple)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Platform:** ROS2 Humble · Gazebo Harmonic · Ubuntu 22.04

A mobile manipulator that learns pick-and-place **from scratch via RL** — no hand-coded trajectories, no demonstrations. A differential-drive base carries a 6-DOF UR3 arm with a Robotiq 2F-85 gripper. A scripted P-controller drives the base to the bin; **TQC** (upgraded from SAC) then learns all arm manipulation end-to-end.

This project covers:
- Mobile base + arm RL in a single policy
- Phase-based curriculum with potential-based reward shaping
- Real Gazebo object pose via `ros_gz` dynamic_pose bridge
- Analytical DH forward kinematics (no TF latency)
- Grasp verification over 10 steps to prevent reward hacking

![Robot in Gazebo](./images/gazebo_robot.png)

---

## What it does

The robot learns to:
1. **Approach** — lower the end-effector to object height and align with the object in XY
2. **Grasp** — position the Robotiq 2F-85 gripper around a 4cm-radius cylinder and close
3. **Lift** — raise the grasped object to 25cm
4. **Transport** — drive the base toward the drop zone while holding the object
5. **Place** — lower and release the object at the target location

A scripted P-controller handles base navigation to the bin (~18cm from object). RL takes over from there.

---

## Use Cases

- **Warehouse automation** — learn bin-picking without manually programming approach trajectories
- **Research** — ready-made Gazebo + ROS2 + TQC environment for reward shaping and curriculum experiments
- **Education** — see [CONCEPTS.md](./CONCEPTS.md) for a full breakdown of TQC, curriculum learning, potential-based shaping, FK, and domain randomisation
- **Baseline** — the scripted pre-grasp + RL manipulation split is a reusable template for any mobile manipulation RL project

---

## RL System

### Algorithm: TQC (Truncated Quantile Critics)

Upgraded from SAC after 95k steps (SAC best: −328). TQC models the full return distribution per critic and drops the top quantiles before computing Bellman targets — this pessimistic bias reduces Q-value overestimation in contact-rich manipulation, giving more stable grasping behaviour.

### Observation Space (24-dim)

| Field | Dim | Description |
|-------|-----|-------------|
| `joint_positions` | 6 | UR3 arm joint angles (rad) |
| `joint_velocities` | 6 | UR3 arm joint speeds (rad/s) |
| `finger_position` | 1 | Gripper open/close state |
| `ee_pos` | 3 | End-effector XYZ in world frame (via DH FK) |
| `obj_pos` | 3 | Object XYZ from Gazebo dynamic pose bridge |
| `object_grasped` | 1 | Binary grasp flag |
| `current_phase` | 1 | Task phase (1–5) |
| `base_pose` | 3 | Base x, y, heading θ |

All quantities are in the **world frame** — no mixed frames that break relative distance calculations.

### Action Space (9-dim, continuous [-1, 1])

| Field | Dim | Description |
|-------|-----|-------------|
| `joint_deltas` | 6 | Position delta per arm joint (×0.05 rad/step) |
| `gripper` | 1 | Close (>0) / open (<0) |
| `base_linear` | 1 | Forward speed — locked to 0 during phases 1–3 |
| `base_angular` | 1 | Turn speed — locked to 0 during phases 1–3 |

Position-delta control (not raw velocity) makes learning stable — action 0 = hold still, max action = 2.9°/step.

### 5-Phase Curriculum

| Phase | Goal | Transition |
|-------|------|-----------|
| 1 | Lower EE + approach object XY | `dist_z < 4cm AND dist_xy < 6cm` |
| 2 | Grasp object | Gripper > 0.7 AND EE within 5cm of object |
| 3 | Lift to 25cm | EE height within 5cm of target |
| 4 | Transport to drop zone | EE within 15cm of target XY |
| 5 | Place and release | EE within 8cm of target, gripper open |

### Reward Design

**Phase 1 & 2 — dense shaping:**
```
approach:  Δdist × 100   (phase 1) / × 80   (phase 2)
retreat:   Δdist × 300   (phase 1) / × 320  (phase 2)   ← 3–4× harsher
proximity bonus:  up to +5/step within 15cm  (phase 1)
                  up to +8/step within 10cm  (phase 2)
touch-range bonus: up to +10/step within 5cm of object
gripper-close bonus: +5 × gripper_pos when closing within 7cm
wrong-close penalty: -(0.5 + dist × 5) when closing far away
gripper-open penalty: -2/step if gripper closes in phase 1
```

**Phase transitions:** +100 to +1000 bonuses at each milestone.

**Grasp verification:** Object Z position monitored for 10 steps after claiming grasp — if object hasn't risen, revert to phase 2 (−10 penalty).

**Safety:** Joint velocity > 10 rad/s or EE underground → episode ends with −500.

---

## Architecture

```
Episode reset
    └── Scripted P-controller: drives base to 18cm from object, arm tucked
            ↓
RL policy (TQC, ~40 Hz)
    ├── Reads: joint_states, /odom, /world/.../dynamic_pose/info
    ├── Computes: FK → EE world pos, phase reward, phase transitions
    └── Publishes: joint cmd_vel × 6, finger cmd_vel, cmd_vel (Twist)
```

**End-effector position** is computed analytically via UR3 DH parameters — no TF latency. A 180° yaw correction (`flip x,y`) accounts for the arm's URDF mount orientation.

---

## Training

```bash
# 1. Build (symlink-install so edits to .py take effect without rebuild)
colcon build --packages-select pickplace_rl_mobile --symlink-install
source install/setup.bash

# 2. Launch Gazebo + RL training (no RViz — headless for speed)
bash src/pickplace_rl_mobile/launch/run_rl_training.sh

# 3. Watch live reward in another terminal
grep -a "ep_rew\|fps" /tmp/training.log | tail -6

# 4. TensorBoard
tensorboard --logdir ./rl_models/tensorboard
# then open http://localhost:6006

# 5. Run best model (eval only, no training)
ros2 run pickplace_rl_mobile train_rl --ros-args -p eval_only:=true
```

**Tip — check joint states live:**
```bash
ros2 topic echo /joint_states --once
```

**Tip — check EE reaching the object:**
```bash
ros2 topic echo /odom --once    # base pose
ros2 topic echo /world/pickplace_world/dynamic_pose/info --once  # object pose
```

**Tip — kill everything cleanly:**
```bash
pkill -9 -f "gz|ros2|train_rl|parameter_bridge"
```

Models saved to `./rl_models/`: checkpoints every 10k steps, `best_model.zip` updated on any eval improvement.

### Expected Progress

| Steps | Reward | Behaviour |
|-------|--------|-----------|
| 0–5k | −2000 to −500 | Random policy, random gripper |
| 5k–20k | −500 to −200 | Arm starts approaching, stops random closing |
| 20k–80k | −200 to 0 | Consistent approach, grasp attempts |
| 80k–200k | 0 to +500 | Reliable grasps, lift emerging |
| 200k–500k | +500 to +2000 | Full pick-place cycles |

> Wall time ~4–14 hrs depending on whether CUDA is available (~40 fps GPU vs ~11 fps CPU).

### Current Status
**TQC_30** running at ~38 fps. Key improvements vs SAC baseline:
- Scripted pre-grasp (base stops 18cm from object)
- Phase 1 requires XY + Z approach (not just Z lowering)
- Asymmetric retreat penalty (3–4× harsher than approach reward)
- Touch-range and gripper-closing bonuses added
- Grasp reward raised to +1000

---

## Concepts

See **[CONCEPTS.md](./CONCEPTS.md)** for in-depth explanations of every technique used:
TQC · Phase curriculum · Potential-based shaping · Scripted pre-grasp (hierarchical RL) · Position-delta control · FK pipeline · Grasp verification · Domain randomisation · Replay buffer

---

## Quick Start

```bash
pip install stable-baselines3 sb3-contrib gymnasium tensorboard
colcon build --packages-select pickplace_rl_mobile --symlink-install
source install/setup.bash
bash src/pickplace_rl_mobile/launch/run_rl_training.sh
```

---

## Maintainer

**Darsh Menon** — [darshmenon02@gmail.com](mailto:darshmenon02@gmail.com) · [@darshmenon](https://github.com/darshmenon)
