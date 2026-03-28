# ARES: Autonomous Robotic End-to-End System — UR3 Mobile Pick & Place via RL

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![SB3-Contrib TQC](https://img.shields.io/badge/SB3--Contrib-TQC-purple)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Platform:** ROS2 Humble · Gazebo Harmonic · Ubuntu 22.04 · CUDA

A mobile manipulator that learns pick-and-place **entirely from scratch via reinforcement learning** — no hand-coded trajectories, no demonstrations, no motion planning. A differential-drive base carries a 6-DOF UR3 arm with a Robotiq 2F-85 gripper. A lightweight scripted controller handles base approach and arm pre-positioning; **TQC** (Truncated Quantile Critics) learns all arm manipulation end-to-end from dense reward shaping.

![Robot in Gazebo](./images/gazebo_robot.png)

---

## What it does

| Step | Description |
|------|-------------|
| **1. Pre-grasp** | Scripted P-controller faces robot toward object, keeps caster clear of bin wall, extends arm (shoulder=-1.7, elbow=2.0, wrist=-1.0) |
| **2. Approach** | RL lowers EE to object height and closes XY distance simultaneously |
| **3. Grasp** | RL positions gripper around the 4cm-radius cylinder and closes fingers |
| **4. Lift** | RL raises grasped object to 25cm clearance height |
| **5. Transport** | RL drives base toward drop zone while holding object |
| **6. Place** | RL lowers and releases object at target location |

---

## Highlights

- **No motion planning** — single TQC policy controls 6 arm joints + gripper + base simultaneously
- **Phase-based curriculum** — 5 phases with milestone bonuses (+100 to +1000) and asymmetric retreat penalties (3–4× harsher than approach reward)
- **Analytical FK** — UR3 DH parameters compute EE world position with zero TF latency
- **Real Gazebo poses** — `ros_gz` dynamic_pose bridge gives ground-truth object position, no fake randomisation
- **Grasp verification** — object Z monitored for 10 steps after grasp claim; reverts to phase 2 if object hasn't risen
- **VecNormalize** — online normalisation of all 24 obs dimensions + reward, critical for mixed-scale inputs
- **Caster-aware pregrasp** — front caster (r=6cm at x=+30cm from chassis) kept clear of bin wall; arm reaches over wall from spawn

---

## RL System

### Algorithm: TQC (Truncated Quantile Critics)

Upgraded from SAC (SAC best reward: −328). TQC distributes return estimates across multiple quantile networks and truncates the top quantiles before Bellman updates — the pessimistic bias suppresses Q-value overestimation in contact-rich manipulation, producing more stable and consistent grasping behaviour.

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Policy network | `[512, 512, 512]` | Deeper than default [256,256] — maps complex 24-dim obs to 9-dim action |
| `gradient_steps` | 4 | 4 updates per env step — higher sample efficiency |
| `buffer_size` | 500 000 | Replay buffer for off-policy learning |
| `batch_size` | 512 | Large batch for stable gradients |
| `gamma` | 0.99 | Long-horizon discounting for multi-phase task |
| `learning_starts` | 1 000 | Random exploration before first update |
| `VecNormalize` | `clip_obs=10` | Normalises obs online; eval env uses frozen stats |
| `top_quantiles_to_drop` | 2 | Conservative Q-targets for manipulation stability |

---

### Observation Space — 24 dimensions

| Field | Dim | Notes |
|-------|-----|-------|
| `joint_positions` | 6 | UR3 arm angles (rad) |
| `joint_velocities` | 6 | UR3 arm speeds (rad/s) |
| `finger_position` | 1 | 0 = open, ~0.8 = fully closed |
| `ee_pos` | 3 | EE XYZ in **world frame** via DH FK |
| `obj_pos` | 3 | Object XYZ from Gazebo dynamic_pose bridge |
| `object_grasped` | 1 | Binary — updated by grasp verification |
| `current_phase` | 1 | Integer phase (1–5) |
| `base_pose` | 3 | Base x, y, heading θ from odometry |

All quantities share the **world frame** — no mixed-frame distance bugs.

---

### Action Space — 9 dimensions (continuous, clipped to [−1, 1])

| Field | Dim | Notes |
|-------|-----|-------|
| `joint_deltas` | 6 | Position delta per arm joint; max ±0.25 rad/step, P-controlled to velocity |
| `gripper` | 1 | >0 → close at 0.5 rad/s, <0 → open |
| `base_linear` | 1 | Forward speed (×0.5 m/s); **zeroed in phases 1–3** |
| `base_angular` | 1 | Turn speed (×1.0 rad/s); **zeroed in phases 1–3** |

Position-delta control (not raw velocity) gives a stable zero-action baseline — the arm holds still when the policy outputs 0.

---

### 5-Phase Curriculum

| Phase | Goal | Reward Signal | Transition Condition |
|-------|------|---------------|---------------------|
| **1** | Lower EE to grasp height + close XY to object | `Δdist × 100` approach / `× 300` retreat | `dist_z < 4 cm AND dist_xy < 6 cm` |
| **2** | Reach object and close gripper | `Δdist × 80 / × 320` + proximity/touch bonuses | Gripper > 0.7 AND dist < 4 cm |
| **3** | Lift object to 25 cm | `Δheight × 100 / × 200` | EE height within 5 cm of 25 cm |
| **4** | Transport to drop zone | `Δdist × 50` base + arm | EE within 15 cm of target XY |
| **5** | Lower and release | `Δdist × 50` | EE within 8 cm, gripper open |

**Milestone bonuses:** +100 (phases 1, 3, 4, 5), +1000 (grasp success at phase 2).

---

### Reward Design — Full Detail

```
Phase 1  (approach)
  Δ(dist_z + dist_xy) × 100   approach  |  × 300   retreat   ← 3× harsher
  proximity bonus:  +5 × (1 − dist_xy/0.15)  when dist_xy < 15 cm
  z-align bonus:    +4 × (1 − dist_z/0.05)   when dist_z  <  5 cm
  dual-close bonus: +8 flat                  when both xy < 8 cm AND z < 5 cm
  gripper-close penalty: −2/step             if gripper closed during approach

Phase 2  (grasp)
  Δdist_to_grasp_target × 80  approach  |  × 320   retreat   ← 4× harsher
  proximity bonus:   +8 × (1 − dist/0.10)   when dist < 10 cm
  touch-range bonus: +15 × (1 − dist/0.04)  when dist <  4 cm  (surface contact)
  very-close bonus:  +10 flat               when dist <  2 cm
  gripper-close bonus: +8 × gripper_pos     when closing within 7 cm
  wrong-close penalty: −(0.5 + dist × 5)    when closing far away
  wrist orientation:  −|wrist_2_angle| × 0.3  (keep gripper horizontal)

Safety / global
  out-of-bounds:      −500 + terminate   if base > 1.5 m from object
  EE underground:     −500 + terminate   (phase ∉ 1,2,5)
  high joint vel:     −500 + terminate   if any joint vel > 10 rad/s
  misalignment penalty: −|angle_err| × 0.2  if base not facing object in phases 1–3
  action smoothness:  −0.01 × Σ|joint_vels|  every step
```

---

## Architecture

```
Episode reset()
    ├── Randomise object XY ±3 cm (domain randomisation)
    ├── Scripted pre-grasp (P-controller, up to 300 steps):
    │       · Turn base to face object
    │       · Drive forward only while chassis_x ≤ 0.04 m
    │         (keeps 6 cm front caster clear of bin back wall at x≈0.40 m)
    │       · Extend arm: pan=0, shoulder=-1.7, elbow=2.0, wrist_1=-1.0
    │       · Break when EE within 30 cm XY of object
    └── Hand off to RL at phase 1

RL step()  (~40 Hz)
    ├── Spin ROS node (joint_states, odom, dynamic_pose)
    ├── Compute DH FK → EE world position
    ├── Execute action (position-delta arm + gripper + base)
    ├── Compute phase reward + check transitions
    └── Return (obs, reward, terminated, truncated)
```

**FK pipeline:** UR3 DH parameters compute EE analytically. A 180° yaw on `base_link_inertia` means FK output is flipped (x→−x, y→−y) before adding the arm mount offset, giving EE in the chassis frame and then world frame via odometry.

---

## Quick Start

```bash
# Dependencies
pip install stable-baselines3 sb3-contrib gymnasium tensorboard

# Build
colcon build --packages-select pickplace_rl_mobile --symlink-install
source install/setup.bash

# Launch with Gazebo GUI (watch the robot)
bash src/pickplace_rl_mobile/launch/run_rl_training.sh

# Launch headless — no GUI window, ~3-4× faster fps
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --headless

# Resume from checkpoint, GUI
bash src/pickplace_rl_mobile/launch/run_rl_training.sh ./rl_models/best_model.zip

# Resume from checkpoint, headless (recommended for long runs)
bash src/pickplace_rl_mobile/launch/run_rl_training.sh ./rl_models/best_model.zip --headless
```

---

## Training

```bash
# Watch live rewards (log written by ros2 launch)
grep -a "ep_rew_mean\|fps" /tmp/training.log | tail -10

# TensorBoard
tensorboard --logdir ./rl_models/tensorboard
# open http://localhost:6006

# Kill everything cleanly
pkill -9 -f "gz|ros2|train_rl|parameter_bridge"

# Check EE and object pose live
ros2 topic echo /joint_states --once
ros2 topic echo /odom --once
ros2 topic echo /world/pickplace_world/dynamic_pose/info --once
```

Checkpoints saved every 10 k steps to `./rl_models/`. `best_model.zip` updated automatically on eval improvement. VecNormalize stats saved to `./rl_models/vecnormalize.pkl` — loaded automatically on resume.

---

### Expected Training Progress

| Steps | Mean Reward | Behaviour |
|-------|-------------|-----------|
| 0–5 k | −2000 to −500 | Random policy, random gripper flapping |
| 5–20 k | −500 to −200 | Arm starts moving toward object, gripper noise drops |
| 20–80 k | −200 to 0 | Consistent approach, first grasp attempts |
| 80–200 k | 0 to +500 | Reliable grasps, lift phase emerging |
| 200–500 k | +500 to +2000 | Full pick-place cycles completing |

> Wall time: ~4–8 hrs on GPU (~38 fps), ~14+ hrs on CPU (~11 fps).

---

### Current Status

**TQC_38** — CUDA, ~38 fps. Cumulative improvements over SAC baseline:

| Change | Impact |
|--------|--------|
| SAC → TQC | Eliminated Q-overestimation; SAC best was −328 |
| Scripted pre-grasp | Deterministic base approach frees RL to focus on manipulation |
| Caster-aware driving | Stops chassis at x ≤ 4 cm to avoid bin wall collision |
| Arm pre-extension | Pregrasp sets shoulder/elbow/wrist to face object; EE ≤ 30 cm from target |
| Asymmetric penalties | 3–4× harsher retreat vs approach; prevents oscillating policy |
| Equal XY+Z weight (phase 1) | Was 0.5× XY; now 1.0× so agent approaches horizontally and vertically together |
| VecNormalize | Normalises mixed-scale 24-dim obs; critical for stable TQC training |
| Network [512,512,512] | Larger than default [256,256]; better function approximation |
| `gradient_steps=4` | 2× more updates per env step; faster convergence |
| Grasp reward +1000 | Strong signal to commit to grasping |
| Grasp verification | 10-step object-Z check prevents reward hacking |

---

## Concepts

See **[CONCEPTS.md](./CONCEPTS.md)** for deep-dives on every technique:
TQC · Phase curriculum · Potential-based reward shaping · Hierarchical control (scripted + RL) · Position-delta control · DH forward kinematics · Grasp verification · Domain randomisation · VecNormalize · Replay buffer

---

## Maintainer

**Darsh Menon** — [darshmenon02@gmail.com](mailto:darshmenon02@gmail.com) · [@darshmenon](https://github.com/darshmenon)
