# ARES: Autonomous Robotic Environment System

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **autonomous mobile manipulator** combining a differential-drive base with a 6-DOF UR3-based arm for open-vocabulary pick-and-place. The system has two complementary control paths:

- **Reinforcement Learning (SAC)** — a trained policy for direct low-level arm and base control, running at 20 Hz
- **VLA (Vision-Language-Action) pipeline** — language understanding + open-vocabulary vision + task planning, driving MoveIt2 for high-level task execution

Both run on top of **Nav2 + SLAM** for autonomous navigation. The mobile base relies on a specialized 4-point differential drive configuration featuring zero-friction caster wheels to support the mass of the 6-DOF UR3 manipulator. Visual observations are captured securely through a static environment camera to avoid jitter.

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
| **RL control** | SAC (Stable-Baselines3), 16-dim obs, 8-dim continuous action |
| **Motion planning** | MoveIt2 MoveGroup action client + velocity fallback |
| **Navigation** | Nav2 AMCL + DWB, custom obstacle-avoidance FSM |
| **Exploration** | Frontier-based SLAM-aware autonomous exploration |
| **Safety** | Joint limits, workspace bounds, LiDAR e-stop at 20 Hz |
| **Sim-to-real** | Domain randomisation (position, colour, physics noise) |

---

## Reinforcement Learning

The RL system trains a **SAC (Soft Actor-Critic)** agent via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) in a custom Gymnasium environment (`PickPlaceEnv`) that wraps the live ROS 2 + Gazebo simulation. The agent learns a **6-phase curriculum** (approach → lower → grasp → lift → transport → place).

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

### 6-Phase Curriculum Learning

The agent learns through **potential-based reward shaping** across 6 sequential phases. Each phase uses distance-difference rewards (`Δd × scale`) that create a dense gradient, preventing the agent from exploiting back-and-forth movement.

| Phase | Goal | Transition Condition | Bonus |
|-------|------|---------------------|-------|
| **0: Approach** | Drive base + extend arm toward object | EE within 10cm XY, base aligned | +100 |
| **1: Lower** | Descend EE to grasp height (z≈0.07m) | Vertical distance < 2cm | +100 |
| **2: Grasp** | Close gripper fingers | Finger position > 0.7 | +500 |
| **3: Lift** | Raise grasped object to safe height (z≈0.25m) | Vertical distance < 5cm | +200 |
| **4: Transport** | Drive to final placement location | EE within 15cm of target | +100 |
| **5: Place** | Lower and release object | Distance < 8cm, gripper open | +1000 |

**Anti-regression penalties:** In Phases 0 and 1, the agent receives a `-10.0` penalty for moving away from the target and `-1.0` for keeping joints stationary. This prevents the common RL failure mode of oscillating or freezing.

**Safety penalties:** Ground collision (EE z < 3cm) and base tipping (joint velocity spike > 10 rad/s) both terminate the episode with `-500`.

### Why SAC?

**Soft Actor-Critic** is ideal for robotic manipulation because:
- **Continuous action space** — SAC handles the 9-dim continuous output (joint velocities + gripper + base) natively, unlike DQN which requires discretization
- **Maximum entropy** — SAC maximizes both reward AND entropy, encouraging exploration. This is critical for multi-phase tasks where the agent must discover the grasp phase to unlock later rewards
- **Off-policy** — SAC reuses past experience (replay buffer), making it 10-100× more sample-efficient than PPO for robotics
- **Automatic temperature tuning** — The entropy coefficient (`ent_coef`) auto-tunes, so the agent explores aggressively early on and exploits learned behavior later

### Reward Shaping: Potential-Based Differences

Instead of sparse rewards (only +1000 at the very end), we use **potential-based shaping**:

```
reward = (previous_distance - current_distance) × scale_factor
```

This gives the agent a dense gradient (warm/cold signal) at every step. Moving closer → positive reward, moving away → negative. The key insight is that difference-based shaping is **theoretically optimal** — it doesn't change the optimal policy (see Ng et al., 1999), but dramatically accelerates learning.

### Expected Training Results

| Timesteps | Mean Reward | Episode Length | Behavior |
|-----------|------------|----------------|----------|
| 0-10k | -600 to -1000 | 50-100 | Random exploration, frequent collisions |
| 10k-50k | -400 to -600 | 200-400 | Base begins aligning toward object |
| 50k-150k | -100 to -400 | 400-600 | Consistent approach + arm lowering |
| 150k-300k | 0 to +200 | 600-800 | Grasping attempts, occasional lifts |
| 300k-500k | +500 to +1500 | 800 | Full pick-place with transport |

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
The SAC policy has completed **~95,000 timesteps** across 32 training runs. Best eval reward so far: **−328** (phase 0→1 approach behaviour is emerging). The robot consistently drives toward the object and begins arm extension. Cylinder physics have been hardened (500g, r=3.5cm, damping added) to prevent tipping on contact.

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
pip install stable-baselines3 gymnasium tensorboard

# Build
colcon build --packages-up-to pickplace_rl_mobile
source install/setup.bash

# Train with Gazebo visualization
./src/pickplace_rl_mobile/launch/run_rl_training.sh

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
