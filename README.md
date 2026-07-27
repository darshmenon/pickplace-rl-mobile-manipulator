# ARES: Autonomous Robotic End-to-End System — UR3 Mobile Pick & Place via RL

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![SB3-Contrib TQC](https://img.shields.io/badge/SB3--Contrib-TQC-purple)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Platform:** ROS2 Humble · Gazebo Harmonic · Ubuntu 22.04 · CUDA

A mobile manipulator that learns pick-and-place **from reinforcement learning plus a scripted pre-grasp routine** — no demonstrations and no motion planning in the RL loop. A differential-drive base carries a 6-DOF UR3 arm with a Robotiq 2F-85 gripper. A lightweight controller handles base approach and arm pre-positioning; **TQC** (Truncated Quantile Critics) then learns the manipulation stages from dense reward shaping.

![Robot in Gazebo](./images/gazebo_robot.png)

---

## What it does

| Step | Description |
|------|-------------|
| **1. Pre-grasp** | Scripted P-controller faces robot toward object, keeps caster clear of bin wall, extends arm (shoulder=-1.7, elbow=2.0, wrist=-1.0) |
| **2. Approach** | RL lowers EE to object height and closes XY distance simultaneously |
| **3. Grasp** | RL positions the gripper around the pickup cube and closes fingers |
| **4. Lift** | RL raises grasped object to 25cm clearance height |
| **5. Transport** | RL drives base toward drop zone while holding object |
| **6. Place** | RL lowers and releases object at target location |

---

## Highlights

- **No motion planning** — single TQC policy controls 6 arm joints + gripper + base simultaneously
- **Phase-based curriculum** — 5 phases with milestone bonuses (+100 to +1000) and asymmetric retreat penalties (3–4× harsher than approach reward)
- **Analytical FK** — UR3 DH parameters compute EE world position with zero TF latency
- **Real Gazebo poses** — `ros_gz` dynamic_pose bridge gives ground-truth object position, no fake randomisation
- **Grasp verification** — object lift is checked against the real Gazebo pose for up to 30 steps before confirming success
- **VecNormalize** — online normalisation of all 46 obs dimensions + reward, critical for mixed-scale inputs
- **Caster-aware pregrasp** — front caster (r=6cm at x=+30cm from chassis) kept clear of bin wall; arm reaches over wall from spawn

---

## RL System

### Algorithm: TQC (Truncated Quantile Critics)

Upgraded from SAC (SAC best reward: −328). TQC distributes return estimates across multiple quantile networks and truncates the top quantiles before Bellman updates — the pessimistic bias suppresses Q-value overestimation in contact-rich manipulation, producing more stable and consistent grasping behaviour.

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Policy network | `[512, 512, 512]` | Deeper than default [256,256] — maps complex 46-dim obs to 9-dim action |
| `gradient_steps` | 4 | 4 updates per env step — higher sample efficiency |
| `buffer_size` | 500 000 | Replay buffer for off-policy learning |
| `batch_size` | 512 | Large batch for stable gradients |
| `gamma` | 0.99 | Long-horizon discounting for multi-phase task |
| `learning_starts` | 1 000 | Random exploration before first update |
| `VecNormalize` | `clip_obs=10` | Normalises obs online; eval env uses frozen stats |
| `top_quantiles_to_drop` | 2 | Conservative Q-targets for manipulation stability |

---

### Observation Space — 46 dimensions

| Field | Dim | Notes |
|-------|-----|-------|
| `joint_positions` | 6 | UR3 arm angles (rad) |
| `joint_velocities` | 6 | UR3 arm speeds (rad/s) |
| `finger_position` | 1 | 0 = open, ~0.8 = fully closed |
| `ee_pos` | 3 | EE XYZ in **world frame** via DH FK |
| `obj_pos` | 3 | Object XYZ from Gazebo dynamic_pose bridge |
| `ee_to_obj` | 3 | Direct tracking vector from EE to object |
| `ee_to_target` | 3 | Vector from EE to placement target |
| `obj_to_target` | 3 | Vector from object to placement target |
| `obj_in_base` | 3 | Object position expressed in base frame |
| `gripper_error` | 1 | Error to desired open/closed gripper state |
| `object_grasped` | 1 | Binary — updated by grasp verification |
| `current_phase` | 1 | Integer phase (1–5) |
| `base_pose` | 3 | Base x, y, heading θ from odometry |
| `prev_action` | 9 | Previous action for temporal smoothing/context |

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
  touch-range bonus: +15 × (1 − dist/0.04)  when dist <  4 cm
  very-close bonus:  +10 flat               when dist <  3 cm
  XY-align bonus:    +10 × (1 − xy/0.05)    when xy < 5 cm
  Z-align bonus:     +8 × (1 − z/0.04)      when z  < 4 cm
  dual-align bonus:  +12 flat               when xy < 4 cm AND z < 3 cm
  gripper-close bonus: +8 × gripper_pos     when closing in true grasp range
  open-near-object penalty: −5              when gripper opens inside 5 cm
  wrong-close penalty: −(0.5 + dist × 5)    when closing far away
  high-close penalty: −10 × z_dist          when closing while vertically misaligned
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
    ├── Randomise object XY ±4 cm (domain randomisation)
    ├── Scripted pre-grasp (P-controller, up to 300 steps):
    │       · Turn base to face object
    │       · Drive forward only while chassis_x ≤ 0.16 m
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

## Setup

```bash
# Clone
git clone https://github.com/darshmenon/pickplace-rl-mobile.git
cd pickplace-rl-mobile

# ROS
source /opt/ros/humble/setup.bash

# Python deps
pip install stable-baselines3 sb3-contrib gymnasium tensorboard

# Build the workspace
colcon build --packages-select pickplace_rl_mobile --symlink-install
source install/setup.bash
```

If you open a new terminal later, run:

```bash
cd /path/to/pickplace-rl-mobile
source /opt/ros/humble/setup.bash
source install/setup.bash
```

---

## Quick Start

Build and source first:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select pickplace_rl_mobile --symlink-install
source install/setup.bash
```

Run the best saved policy in Gazebo:

```bash
ros2 launch pickplace_rl_mobile full_system.launch.py \
  use_rl:=true \
  use_perception:=false \
  model_path:=./rl_models/best_model/best_model.zip
```

`use_perception:=false` uses the world-file fallback object pose `[0.6, 0.0, 0.1325]`, which matches the red pickup box in `pickplace_world.world`.

Train or resume training:

```bash
# Resume best checkpoint, with Gazebo GUI if DISPLAY works
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --resume-best

# Resume best checkpoint, headless
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --resume-best --headless

# Resume latest numbered checkpoint
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --resume-latest --headless

# Start fresh
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --fresh --headless

# Train a specific curriculum stage
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --curriculum-stage 2 --timesteps 200000 --headless
```

This is the **recommended launch path** for RL work because it starts Gazebo and the trainer with the wiring used by the current checkpoints.

---

## Launch Guide

### RL training

```bash
# Resume best checkpoint
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --resume-best --headless

# Start a fresh run
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --fresh --headless

# Resume from the latest numbered checkpoint
bash src/pickplace_rl_mobile/launch/run_rl_training.sh --resume-latest --headless

# Resume an explicit checkpoint
bash src/pickplace_rl_mobile/launch/run_rl_training.sh ./rl_models/pickplace_model_690000_steps.zip --headless
```

### Gazebo only

```bash
ros2 launch pickplace_rl_mobile gazebo.launch.py

# Headless Gazebo only
ros2 launch pickplace_rl_mobile gazebo.launch.py headless:=true
```

### Trainer only

Use this only if Gazebo is already running in another terminal.

```bash
ros2 launch pickplace_rl_mobile rl_train.launch.py

# Resume from a saved checkpoint
ros2 launch pickplace_rl_mobile rl_train.launch.py load_model:=./rl_models/best_model/best_model.zip
```

### Full system launch

This path runs the broader stack for inference/demo. Use the training script above for checkpointed learning.

```bash
ros2 launch pickplace_rl_mobile full_system.launch.py

# Full system with RL inference node
ros2 launch pickplace_rl_mobile full_system.launch.py \
  use_rl:=true \
  use_perception:=false \
  model_path:=./rl_models/best_model/best_model.zip

# Full system with Nav2
ros2 launch pickplace_rl_mobile full_system.launch.py use_nav2:=true
```

### VLA pipeline

```bash
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py

# Lighter fallback mode without the LLM or OWLv2
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py use_llm:=false use_owlv2:=false
```

### RViz and URDF view

```bash
ros2 launch pickplace_rl_mobile display_launch.py
```

---

## Training

```bash
# TensorBoard
tensorboard --logdir ./rl_models/tensorboard
# open http://localhost:6006

# Check live state
ros2 topic echo /joint_states --once
ros2 topic echo /odom --once

# Check whether the policy is publishing arm commands
ros2 topic hz /shoulder_pan_joint/cmd_vel --window 5

# Stop Gazebo/ROS if needed
pkill -f "gz sim|ros2 launch|parameter_bridge|train_rl"
```

Checkpoints save every 10 k steps to `./rl_models/`. The best eval checkpoint is `./rl_models/best_model/best_model.zip`, with normalization stats at `./rl_models/best_model/best_vecnormalize.pkl`. Latest-run VecNormalize stats save to `./rl_models/vecnormalize.pkl` and replay data to `./rl_models/replay_buffer.pkl`; both are reused automatically on resume when compatible.

---

### Latest Local Checkpoints

| Artifact | Status |
|----------|--------|
| Latest numbered checkpoint | `rl_models/pickplace_model_785000_steps.zip` |
| Best eval checkpoint | `rl_models/best_model/best_model.zip` |
| Latest eval file | `rl_models/evaluations.npz` |
| Last recorded eval step | `775000` |
| Last recorded mean eval reward | **-302.44** |

Eval reward and the training-reward scale were realigned (see "Align TQC training and evaluation behavior"), so this number isn't directly comparable to older reward figures — treat it as a fresh baseline. Grasping and lifting (phases 2-3) are consistently reached in training rollouts; full end-to-end place success (phase 5) under the harder curriculum-stage-0 (full task) setting is the current focus.

---

### Current Trainer Behavior

The current trainer resumes from checkpoints, restores VecNormalize stats, reloads the replay buffer when possible, and auto-detects legacy 27-dim checkpoints versus the current 46-dim observation mode. Key improvements over the older SAC setup:

| Change | Impact |
|--------|--------|
| SAC → TQC | Replaced the older SAC baseline with a more stable critic ensemble |
| Scripted pre-grasp | Deterministic base approach frees RL to focus on manipulation |
| Caster-aware driving | Stops chassis at x ≤ 16 cm to avoid bin wall collision |
| Arm pre-extension | Pregrasp sets shoulder/elbow/wrist to face object; EE ≤ 30 cm from target |
| Asymmetric penalties | 3–4× harsher retreat vs approach; prevents oscillating policy |
| Equal XY+Z weight (phase 1) | Was 0.5× XY; now 1.0× so agent approaches horizontally and vertically together |
| VecNormalize | Normalises mixed-scale 46-dim obs; critical for stable TQC training |
| Network [512,512,512] | Larger than default [256,256]; better function approximation |
| `gradient_steps=4` | 2× more updates per env step; faster convergence |
| Verified grasp reward +1000 | Only awarded once the real Gazebo object actually lifts |
| Grasp verification | Real-object lift verification over a 30-step window prevents reward hacking |
| Action/perception latency randomization | Episode-sampled delay on executed actions and on the perceived object position (`domain_randomizer.py`), so the policy tolerates real actuator/control-loop lag and camera-pipeline latency instead of only ever seeing instant, ground-truth state |
| Perception noise separated from reward | The observation sees a noisy estimate of object position; reward/grasp-verification logic still uses Gazebo ground truth, so training signal quality doesn't degrade along with simulated perception |
| Transformer policy option | `--policy-arch transformer` tokenizes the observation into named groups (joints, EE pos, object pos, ...) and self-attends over them instead of a flat MLP; requires a fresh model, not resumable from an MLP checkpoint |
| Atomic checkpoint saves | Numbered checkpoints save to a temp file and rename into place, so a killed process can no longer leave a truncated, unloadable `*_steps.zip` |

Training rollouts use annealed approach/transport assist, but evaluation environments disable assist by default so best-model selection reflects the learned policy.

Concurrent experiments (e.g. comparing `--policy-arch mlp` vs `transformer`) can share the machine without colliding: set `PICKPLACE_DOMAIN_BASE`, `PICKPLACE_TRAIN_PARTITION`, `PICKPLACE_EVAL_PARTITION`, and `PICKPLACE_RUNTIME_ROOT` to disjoint values per run, and give each a separate `--save-dir`.

---

## Roadmap

| Feature | Status | Description |
|---------|--------|-------------|
| Safety guard joint names | Done | Fixed to match UR3 (`shoulder_pan_joint`, etc.) with real DH FK |
| Eval reward plot script | Done | `python3 plot_training.py` — reward curves + episode length |
| Place phase reward shaping | Done | Phase 5 now has proximity bonuses, alignment bonuses, early-drop penalty |
| Approach assist annealing | Done | Assist blend decays 1.0 → 0 over 1M steps |
| Base navigation assist (phase 4) | Done | Gentle base nudge toward drop zone, also annealed |
| Running success rate metric | Done | Rolling 100-episode success rate logged to `monitor.csv` and `info` |
| Domain randomization → Gazebo | Done | Object respawned each episode with randomized color, mass, size, friction; gravity perturbed via `set_physics` |
| Multi-object generalization | Done | Box, cylinder, and sphere spawned per episode with per-shape inertia; curriculum thresholds adjusted for added difficulty |
| Success rate plot | Done | `plot_training.py` now shows rolling success rate from monitor CSVs as a third panel |

---

## Concepts

See **[CONCEPTS.md](./CONCEPTS.md)** for deep-dives on every technique:
TQC · Phase curriculum · Potential-based reward shaping · Hierarchical control (scripted + RL) · Position-delta control · DH forward kinematics · Grasp verification · Domain randomisation · VecNormalize · Replay buffer

---

## Maintainer

**Darsh Menon** — [darshmenon02@gmail.com](mailto:darshmenon02@gmail.com) · GitHub: [@darshmenon](https://github.com/darshmenon)
