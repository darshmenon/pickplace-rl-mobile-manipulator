# ARES: Autonomous Robotic Environment System

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mobile manipulator learning pick-and-place via deep RL — no hand-coded trajectories. A differential-drive base carries a UR3 arm; a **TQC policy** controls the arm end-to-end from a scripted pre-grasp position to object lift and placement.

**Training highlights:** phase-curriculum reward shaping · real Gazebo object pose feedback · position-delta arm control · 24-dim obs / 9-dim action · ~35 Hz on a single Gazebo Harmonic world

![Robot Model](./images/gazebo_robot.png)
![New Mobile UR3 Robot](./images/new_mobile_ur3.png)

---



---

## Key Features

| Feature | Implementation |
|---------|----------------|
| **Language understanding** | SmolLM2-360M-Instruct (on-device, ~700 MB) + regex fallback |
| **Open-vocabulary vision** | OWLv2 base-patch16 (~590 MB) + HSV colour fallback |
| **Object memory** | Persistent timestamped map, 30s occlusion tolerance |
| **Task planning** | Sort-all, clear-all, stack, single pick-and-place |
| **RL control** | TQC (sb3-contrib), 24-dim obs, 9-dim continuous action, scripted pre-grasp nav |
| **Motion planning** | MoveIt2 MoveGroup action client + velocity fallback |
| **Navigation** | Nav2 AMCL + DWB, custom obstacle-avoidance FSM |
| **Exploration** | Frontier-based SLAM-aware autonomous exploration |
| **Safety** | Joint limits, workspace bounds, LiDAR e-stop at 20 Hz |
| **Sim-to-real** | Domain randomisation (position, colour, physics noise) |

---

## Reinforcement Learning

The RL system trains a **TQC (Truncated Quantile Critics)** agent via [sb3-contrib](https://sb3-contrib.readthedocs.io/) in a custom Gymnasium environment (`PickPlaceEnv`) that wraps the live ROS 2 + Gazebo simulation. The agent learns a **5-phase curriculum** (lower → grasp → lift → transport → place) — the base approach is handled by a scripted P-controller so RL focuses only on manipulation.

### Observation Space (24-dim)

| Slice | Dimensions | Description |
|-------|-----------|-------------|
| `joint_positions` | 6 | UR3 arm joint angles |
| `joint_velocities` | 6 | UR3 arm joint velocities |
| `finger_position` | 1 | Gripper finger position |
| `end_effector_pos` | 3 | EE position (x, y, z) via FK |
| `object_pos` | 3 | Target object position |
| `object_grasped` | 1 | Binary grasp flag |
| `current_phase` | 1 | Task phase (0-5) |
| `base_pose` | 3 | Mobile base (x, y, θ) |

### Action Space (9-dim, continuous in [-1, 1])

| Slice | Dimensions | Description |
|-------|-----------|-------------|
| `joint_velocities` | 6 | UR3 arm joint velocity commands |
| `gripper_control` | 1 | Open / close gripper |
| `base_linear_vel` | 1 | Forward / backward base speed |
| `base_angular_vel` | 1 | Rotational base speed |

### Training

```bash
# Train SAC with Gazebo visualization (recommended)
./src/pickplace_rl_mobile/launch/run_rl_training.sh

# Monitor training
tensorboard --logdir ./rl_models/tensorboard
```

Checkpoints are saved to `./rl_models/` every 10 000 steps. The best model (by eval reward) is saved as `best_model.zip`.

### Architecture: Scripted Nav + RL Manipulation

Training is split into two stages per episode:

1. **Scripted pre-grasp** (not RL) — a P-controller drives the base to ~25 cm from the bin (just outside the left wall), arm tucked upright. This runs at reset time and takes ~5–10 s.
2. **RL manipulation** — TQC controls the arm and gripper across 5 phases from the pre-positioned base.

This dramatically reduces the RL exploration space — no need to discover driving — and eliminates base-tipping crashes.

### 5-Phase Curriculum Learning

| Phase | Goal | Transition Condition | Bonus |
|-------|------|---------------------|-------|
| **1: Lower** | Descend EE to object height | Vertical distance < 2cm | +100 |
| **2: Grasp** | Close gripper on object | Finger > 0.7 AND EE within 8cm of real object pos | +500 |
| **3: Lift** | Raise object to safe height (z≈0.25m) | Vertical distance < 5cm | +200 |
| **4: Transport** | Drive EE + base toward drop target | EE within 15cm of target XY | +100 |
| **5: Place** | Lower and release | Distance < 8cm, gripper open | +1000 |

**Grasp verification:** After claiming a grasp (phase 2→3), the real Gazebo object position (bridged via `/world/pickplace_world/dynamic_pose/info`) is monitored for 10 steps. If the object hasn't risen off the floor, the grasp is cancelled and phase resets to 2 with −50 penalty — no more false grasps.

**Safety penalties:** Ground collision (EE z < 3cm) and joint velocity spike > 10 rad/s both terminate with −500.

### Why TQC? (Upgraded from SAC)

The project initially trained with **SAC** for ~95,000 timesteps (best eval reward −328). After switching to a scripted pre-grasp approach, the algorithm was upgraded to **TQC** which extends SAC with distributional critics that model the full return distribution rather than just the mean. For robotic manipulation:
- All SAC benefits (continuous actions, max entropy, off-policy, auto temperature)
- **Reduced overestimation bias** — drops the top quantiles per critic network, leading to more stable Q-value estimates
- **Better on manipulation benchmarks** — shown to outperform SAC on robotics tasks in the original paper (Kuznetsov et al., 2020)

SAC was switched out because after 95k steps the reward plateaued at −328. The combination of scripted pre-grasp (eliminating base navigation from the RL problem) and TQC's distributional critics gives the agent a much cleaner learning signal from the pre-positioned start.

### Reward Shaping: Potential-Based Differences

Instead of sparse rewards (only +1000 at the very end), we use **potential-based shaping**:

```
reward = (previous_distance - current_distance) × scale_factor
```

This gives the agent a dense gradient (warm/cold signal) at every step. Moving closer → positive reward, moving away → negative. The key insight is that difference-based shaping is **theoretically optimal** — it doesn't change the optimal policy (see Ng et al., 1999), but dramatically accelerates learning.

### Expected Training Results

With scripted pre-grasp + TQC, the robot starts every episode already positioned at the bin. Typical progression on a CUDA GPU:

| Timesteps | Wall Time | Mean Reward | Episode Length | Behavior |
|-----------|-----------|------------|----------------|----------|
| 0–10k | ~15 min | −700 to −500 | 400–800 | Random arm exploration from pre-grasp pose |
| 10k–30k | ~45 min | −400 to −600 | 600–800 | Arm begins lowering toward object |
| 30k–80k | ~2 hrs | −200 to −400 | 700–800 | Consistent arm approach, grasp attempts |
| 80k–150k | ~4 hrs | −100 to +200 | 800 | Grasps emerging, occasional lifts |
| 150k–300k | ~8 hrs | +200 to +800 | 800 | Reliable grasp + lift, transport learning |
| 300k–500k | ~14 hrs | +800 to +1500 | 800 | Full pick-place cycles |

> **Note:** Wall time assumes ~40 fps simulation. Each episode reset includes ~5–10 s of scripted navigation, so effective step rate is lower than pure-sim benchmarks.

### Where to Find the Trained Policy

```
./rl_models/
├── best_model.zip              # Best model by eval reward
├── pickplace_model_10000_steps.zip  # Checkpoint at 10k
├── pickplace_model_20000_steps.zip  # Checkpoint at 20k
├── ...
├── evaluations.npz             # Numpy array of eval metrics
└── tensorboard/
    └── SAC_XX/                 # TensorBoard event files
```

### Inference

The trained policy runs inside `manip_rl_node` at **20 Hz**:

```bash
ros2 run pickplace_rl_mobile manip_rl_node --ros-args -p model_path:=./rl_models/best_model.zip
```

### Training

```bash
# Single robot training (stable, recommended)
ros2 launch pickplace_rl_mobile standalone_rl_training.launch.py
```

**Multi-world parallel training** (`multi_world_training.launch.py`) is implemented but WIP — Gz plugin isolation via `GZ_PARTITION` requires further debugging before it is stable for production use.

*Note: The environment is mathematically tuned so the autonomous chassis must approach within **0.4m** of the **20cm x 20cm** object bin. This guarantees the pickup target is exactly **0.25m** from the arm base—perfectly avoiding chassis collisions while keeping the grasp safely within the UR3's 0.5m envelope.*

### Current Training Status (Active)
TQC run **TQC_3** is live. After switching from SAC (95k steps, best −328) to scripted pre-grasp + TQC, episode length is now a stable **800 steps** (no more base crashes). ~7k steps completed, reward trending from −527 toward improvement. Object is a 500g puck (r=4cm, h=8cm) — fits within the Robotiq 2F-85 gripper's 8.5cm max opening.

---

## Architecture

### Mobile Base Design

The differential drive chassis features:
- **4-point contact**: Two driven wheels (radius 8cm) + two passive zero-friction caster spheres (radius 4cm)
- **10kg chassis mass**: Low center of gravity prevents tipping under UR3 arm movement
- **Static world camera**: RGBD sensor fixed at (0.6, 0, 0.8) for jitter-free visual observations

### Control Architecture

```
SAC Policy (20 Hz)
    ├── Joint velocities → /shoulder_pan_joint/cmd_vel (Float64)
    ├── Joint velocities → /shoulder_lift_joint/cmd_vel (Float64)
    ├── Joint velocities → /elbow_joint/cmd_vel (Float64)
    ├── Joint velocities → /wrist_1_joint/cmd_vel (Float64)
    ├── Joint velocities → /wrist_2_joint/cmd_vel (Float64)
    ├── Joint velocities → /wrist_3_joint/cmd_vel (Float64)
    ├── Gripper command  → /finger_joint/cmd_vel (Float64)
    └── Base velocity    → /cmd_vel (Twist)
```

---

## Quick Start

```bash
# Install dependencies
pip install stable-baselines3 sb3-contrib gymnasium tensorboard

# Build
colcon build --packages-up-to pickplace_rl_mobile
source install/setup.bash

# Train (Gazebo + scripted pre-grasp + TQC)
ros2 launch pickplace_rl_mobile standalone_rl_training.launch.py

# Monitor training
tensorboard --logdir ./rl_models/tensorboard

# Run trained policy
ros2 run pickplace_rl_mobile manip_rl_node --ros-args -p model_path:=./rl_models/best_model.zip
```

---

## Future Plans

### VLA (Vision-Language-Action) Pipeline

A planned language-conditioned manipulation system that extends the RL policy with:

```
User text  →  SmolLM2 (language)  →  Task Planner  →  Coordinator
                                                           ↑
Camera     →  OWLv2 (vision)      →  Object Memory  ──────┘
                                                           ↓
                                                    MoveIt2 Action Node
```

- **Understand natural language** — `"sort all objects by colour"` or `"pick the blue cube and place in the tray"`
- **See any object** — open-vocabulary detection via OWLv2 (`"red coffee mug"`, `"yellow banana"`)
- **Remember where things are** — persistent object map with 30s occlusion decay
- **Plan multi-step tasks** — decompose sort/stack/clear goals into sequential pick-and-place actions
- **Navigate autonomously** — Nav2 + SLAM + frontier-based exploration
- **Stay safe** — real-time joint limit, workspace, and obstacle monitoring with e-stop
- **Domain randomization** — Randomize object pose, mass, friction for sim-to-real transfer
- **HER (Hindsight Experience Replay)** — Relabel failed episodes for faster learning
- **Image observations** — CNN encoder for the RGBD camera feed

---

## Documentation

- **[docs/vla_concepts.md](./docs/vla_concepts.md)** — VLA architecture, node reference, launch guide
- **[docs/system_architecture_and_improvements.md](./docs/system_architecture_and_improvements.md)** — Full system architecture, topic reference
- **[docs/robot_architecture.md](./docs/robot_architecture.md)** — URDF, hardware specs, sensor configuration

---

## Maintainer

**Darsh Menon** — [darshmenon02@gmail.com](mailto:darshmenon02@gmail.com) · [@darshmenon](https://github.com/darshmenon)
