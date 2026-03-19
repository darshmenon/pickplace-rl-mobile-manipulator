# ARES: Autonomous Robotic Environment System

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **autonomous mobile manipulator** combining a differential-drive base with a 6-DOF UR3-based arm for open-vocabulary pick-and-place. The system uses **Reinforcement Learning (SAC)** for end-to-end manipulation, a **VLA (Vision-Language-Action) pipeline** powered by small on-device ML models, and **Nav2** for autonomous navigation.

![Robot Model](./images/gazebo_robot.png)
![New Mobile UR3 Robot](./images/new_mobile_ur3.png)

---

## What This Robot Can Do

1. **Understand natural language** — `"sort all objects by colour"` or `"pick the blue cube and place in the tray"`
2. **See any object** — open-vocabulary detection via OWLv2 (`"red coffee mug"`, `"yellow banana"`)
3. **Remember where things are** — persistent object map with 30s occlusion decay
4. **Plan multi-step tasks** — decompose sort/stack/clear goals into sequential pick-and-place actions
5. **Navigate autonomously** — Nav2 + SLAM + frontier-based exploration
6. **Stay safe** — real-time joint limit, workspace, and obstacle monitoring with e-stop

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

## Quick Start

```bash
# Install ML dependencies (optional — graceful fallback without them)
pip install transformers torch pillow

# Build
colcon build --packages-select pickplace_rl_mobile
source install/setup.bash

# Launch VLA pipeline (all 6 nodes)
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py

# Launch without ML models (zero extra dependencies)
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py use_llm:=false use_owlv2:=false

# Send a command
ros2 topic pub /vla_instruction std_msgs/String "data: 'pick the blue cube and place in tray'"

# Multi-step task
ros2 topic pub /vla_instruction std_msgs/String "data: 'sort all objects by colour'"

# Full system (Gazebo + perception + safety + Nav2)
ros2 launch pickplace_rl_mobile full_system.launch.py use_nav2:=true

# RL training
ros2 launch pickplace_rl_mobile rl_train.launch.py
tensorboard --logdir ./rl_models/tensorboard
```

---

## VLA Pipeline

```
User text  →  SmolLM2 (language)  →  Task Planner  →  Coordinator
                                                           ↑
Camera     →  OWLv2 (vision)      →  Object Memory  ──────┘
                                                           ↓
                                                    MoveIt2 Action Node
```

Six nodes, each independently testable:

| Node | Purpose |
|------|---------|
| `vla_language_node` | Text → structured JSON (SmolLM2 / regex) |
| `vla_vision_node` | Camera → object poses (OWLv2 / HSV) |
| `object_memory_node` | Persistent object map with decay |
| `task_planner_node` | Multi-step task decomposition |
| `vla_coordinator_node` | Resolves poses and drives execution |
| `vla_action_node` | MoveIt2 motion planning + gripper |

---

## Documentation

- **[docs/vla_concepts.md](./docs/vla_concepts.md)** — VLA architecture, node reference, launch guide, ML setup
- **[docs/system_architecture_and_improvements.md](./docs/system_architecture_and_improvements.md)** — Full system architecture, topic reference, roadmap
- **[docs/robot_architecture.md](./docs/robot_architecture.md)** — URDF, hardware specs, sensor configuration

---

## Maintainer

**Darsh Menon** — [darshmenon02@gmail.com](mailto:darshmenon02@gmail.com) · [@darshmenon](https://github.com/darshmenon)
