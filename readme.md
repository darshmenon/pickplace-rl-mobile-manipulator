# ARES: UR3 Mobile Robot RL Pick and Place

## Overview
The **ARES UR3 Mobile Robot RL Pick and Place** is a ROS 2 Jazzy-based project that integrates a mobile robotic base and a 6-DOF UR3-based robotic arm. The system uses **Reinforcement Learning (RL)** with the SAC (Soft Actor-Critic) algorithm to autonomously pick objects from a bin and place them at a target location.

This project combines multiple robotics concepts:
- Mobile navigation with differential drive
- 6-DOF UR3-based robotic arm manipulation
- Parallel gripper for grasping
- RGB and Depth cameras for RGB-D perception with HSV color segmentation
- 2D LiDAR sensor for obstacle detection and Nav2 navigation
- Wheel encoders for odometry and Joint State sensors for proprioception
- RL-based policy learning with Stable-Baselines3
- Real-time safety monitoring with emergency stop
- Domain randomization for sim-to-real transfer
- ROS 2-based modular architecture
- Gazebo Harmonic simulation with physics

The goal is to create an end-to-end autonomous mobile manipulator capable of performing pick-and-place tasks in simulated environments, with a clear path to real-world deployment.

### Why this project?
Most reinforcement learning robotics projects focus on either navigation (mobile robots) or manipulation (fixed-base arms). Combining both into a **mobile manipulator** (like ARES) offers a significantly more complex but capable system. By utilizing ROS 2, Gazebo Harmonic, and Stable-Baselines3, this project functions as a comprehensive boilerplate and educational resource for mastering the intersection of modern simulation, continuous control RL, and advanced sensor processing.

---

## System Architecture Deep-Dive

This section is heavily detailed in our supplementary documentation. 
Please refer to:
1. **[Robot Architecture Document](./docs/robot_architecture.md)** for details on the URDF, mobile base, 6-DOF UR3 arm, and sensors (RGB-D, LiDAR).
2. **[System Architecture & Future Improvements](./docs/system_architecture_and_improvements.md)** for details on the software stack (Perception, Safety Guard, RL Env, Domain Randomization, Nav2) and the future roadmap.

This section explains how every component of the system works together, from sensors to decision-making. The architecture is designed to be highly modular, explicitly separating perception, safety, and control so that real-world sensors (like a physical RealSense camera) can eventually substitute the simulated ones without modifying the RL policy.

### Overview

The system follows a **sense-plan-act** architecture. Sensors mounted on the robot (cameras, LiDAR, wheel encoders) feed data through ROS 2 topics to perception and safety nodes. The RL policy node consumes processed perception data and joint state feedback to produce motor commands for both the mobile base and the arm.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gazebo Harmonic                          │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Diff Drive    │  │ RGB Cam  │  │ Depth Cam│  │ 2D LiDAR  │  │
│  │ /cmd_vel      │  │ /camera/ │  │ /camera/ │  │ /scan     │  │
│  │ /odom         │  │ image_raw│  │ depth    │  │           │  │
│  └──────┬───────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│         │               │             │               │        │
│  ┌──────┴───────────────┴─────────────┴───────────────┴─────┐  │
│  │                    ros_gz_bridge                          │  │
│  └──────┬───────────────┬─────────────┬───────────────┬─────┘  │
└─────────┼───────────────┼─────────────┼───────────────┼────────┘
          │               │             │               │
    ┌─────┴─────┐  ┌──────┴─────────────┴───────┐ ┌────┴────────┐
    │  ManipRL  │  │     Perception Node        │ │ Safety Guard│
    │  Node     │←─│  HSV Segmentation          │ │ Joint Limits│
    │  (SAC)    │  │  Depth 3D Projection       │ │ Obstacle    │
    │           │  │  /perception/detected_obj  │ │ E-Stop      │
    └─────┬─────┘  └────────────────────────────┘ └─────────────┘
          │
    ┌─────┴─────┐
    │   Nav2    │
    │  (opt.)   │
    │  AMCL +   │
    │  DWB      │
    └───────────┘
```

### 1. The Robot (URDF)

The robot is described in a single URDF file (`pickplace_mobile_arm.urdf`) and consists of:

**Mobile Base:**
- A rectangular chassis (0.45m x 0.35m) with two driven wheels (differential drive) and aesthetic details like hubcaps and a top panel
- Controlled via `/cmd_vel` (Twist messages) through the Gazebo `DiffDrive` plugin
- Publishes odometry on `/odom` at 50Hz

**6-DOF UR3-based Robotic Arm:**
- **Shoulder pan joint** (revolute, Z-axis rotation) — pans the arm left/right
- **Shoulder lift joint** (revolute, Y-axis) — lifts the arm up/down
- **Elbow joint** (revolute, Y-axis) — bends the forearm
- **Wrist 1 joint** (revolute, Y-axis)
- **Wrist 2 joint** (revolute, Z-axis)
- **Wrist 3 joint** (continuous, Y-axis) — rotates the gripper
- Each joint has its own Gazebo `JointController` plugin accepting velocity commands

**Parallel Gripper:**
- Two prismatic finger joints that slide in/out along the Y-axis
- Rubber-pad aesthetics on the finger tips
- Controlled by velocity commands for open/close

**Sensors:**
- **RGB Camera** (640x480 @ 30Hz) — mounted on the front of the chassis, angled slightly downward (0.3 rad pitch). Publishes to `/camera/image_raw`
- **Depth Camera** (640x480 @ 15Hz) — co-located with the RGB camera. Publishes to `/camera/depth`. Used for back-projecting 2D detections into 3D world coordinates
- **2D LiDAR** (360 degrees, 640 samples @ 10Hz, 12m range) — mounted on top of the chassis. Publishes to `/scan`. Used for Nav2 costmaps and obstacle avoidance



---

## Previews

### Simulation Environment
![Gazebo Simulation](./images/gazebo_simulation.png)

### Robot Close-up
![Robot Model](./images/gazebo_robot.png)

---

## Features

### 1. Mobile Manipulator
- Differential drive mobile base with wheel encoders
- 4-DOF robotic arm (shoulder pan, shoulder pitch, elbow, wrist pitch)
- Parallel gripper with prismatic fingers for grasping
- RGB-D camera and 2D LiDAR for perception

### 2. VLA Pipeline (Vision-Language-Action)
- **Language**: SmolLM2-360M-Instruct (on-device, ~700 MB) parses natural language into structured JSON — regex fallback requires zero extra dependencies
- **Vision**: OWLv2 base-patch16 (~590 MB) for open-vocabulary object detection; HSV colour segmentation fallback
- **Object Memory**: Persistent timestamped object map with 30 s occlusion decay
- **Task Planning**: Decomposes `sort-all`, `clear-all`, `stack`, and single pick-and-place goals automatically
- **MoveIt2 Action Node**: Cartesian/joint-space motion planning with velocity fallback
- Six independently testable nodes — run the full pipeline or just language/vision during development

### 3. Reinforcement Learning
- **Algorithm**: SAC (Soft Actor-Critic) for continuous control
- **Environment**: Custom Gymnasium environment with ROS 2 integration
- **Observation Space**: Joint positions, end-effector position, object position, grasp state, and current state-machine phase (0-5)
- **Action Space**: Joint velocities + gripper control + base motion
- **State Machine Training**: The agent follows a precise sequence to avoid random thrashing:
   - `APPROACH` -> `LOWER` -> `GRASP` -> `LIFT` -> `MOVE_TO_TARGET` -> `RELEASE`
- **Reward Shaping**:
  - Substantial bonus rewards for transitioning between the specific state phases
  - Dense rewards for minimizing distance to the current phase goal
  - Heavy collision penalty: end episodes immediately with massive penalty if reaching too low
  - Smoothness penalty: encourages smooth motions

### 4. Autonomous Navigation
- Nav2 AMCL + DWB controller for map-based localisation and path planning
- Frontier-based SLAM-aware exploration
- `navigate_and_pick_node` integrates Nav2 goals with VLA pick-and-place execution

### 5. Simulation
- **Gazebo Harmonic**: Full physics simulation
- **World**: Custom environment with object bin and target zone
- **Plugins**: Differential drive controller, camera sensor, depth camera, LiDAR, joint state publisher
- **Domain Randomisation**: Object position, colour, and physics noise for sim-to-real transfer
- **Pickable Objects**: Dynamic cubes for pick-and-place tasks

---

## Directory Structure

```
pickplace-rl-mobile-manipulator/
├── docs/                            # Architecture and concept docs
├── images/                          # Screenshots and media
├── rl_models/                       # Saved RL model checkpoints
├── rviz/                            # RViz configuration files
├── src/pickplace_rl_mobile/
│   ├── config/
│   │   ├── training_config.yaml     # RL hyperparameters
│   │   └── nav2_params.yaml         # Nav2 navigation config
│   ├── launch/
│   │   ├── full_system.launch.py    # Full system (Gazebo + all nodes)
│   │   ├── vla_full_pipeline.launch.py  # All 6 VLA nodes
│   │   ├── vla_phase1.launch.py     # Vision + language only
│   │   ├── gazebo_launch.py         # Gazebo-only simulation
│   │   ├── gazebo.launch.py         # Minimal Gazebo
│   │   ├── display_launch.py        # RViz visualization
│   │   ├── rl_train.launch.py       # RL training
│   │   └── standalone_rl_training.launch.py
│   ├── pickplace_rl_mobile/
│   │   ├── vla_language_node.py     # Text → JSON (SmolLM2 / regex)
│   │   ├── vla_vision_node.py       # Camera → object poses (OWLv2 / HSV)
│   │   ├── vla_coordinator_node.py  # Resolves poses, drives execution
│   │   ├── vla_action_node.py       # MoveIt2 motion planning + gripper
│   │   ├── object_memory_node.py    # Persistent object map with decay
│   │   ├── task_planner_node.py     # Multi-step task decomposition
│   │   ├── navigate_and_pick_node.py # Nav2-integrated pick-and-place
│   │   ├── perception_node.py       # RGB-D object detection (legacy)
│   │   ├── safety_guard.py          # Safety monitoring + e-stop
│   │   ├── manip_rl_node.py         # RL policy inference node
│   │   ├── domain_randomizer.py     # Sim-to-real randomization
│   │   ├── pickplace_env.py         # Gymnasium RL environment
│   │   ├── train_rl.py              # RL training script
│   │   ├── test_policy.py           # Policy evaluation
│   │   ├── smart_pick_place.py      # Scripted IK pick-and-place
│   │   └── demo_pick_place.py       # Simple demo
│   ├── urdf/
│   │   └── pickplace_mobile_arm.urdf
│   ├── worlds/
│   │   └── pickplace_world.world
│   ├── setup.py
│   └── package.xml
└── readme.md
```

---

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 or later
- **ROS 2**: Jazzy Jalisco
- **Gazebo**: Gazebo Harmonic (Sim)
- **Python**: 3.10+

### ROS 2 Dependencies
```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-gazebo-ros-pkgs \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-rviz2 \
  ros-jazzy-xacro \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup
```

### Python Dependencies
```bash
pip install gymnasium stable-baselines3 torch numpy
```

---

## Installation and Build

### 1. Clone the Repository
```bash
cd ~/
git clone https://github.com/darshmenon/pickplace-rl-mobile-manipulator.git
cd pickplace-rl-mobile-manipulator
```

### 2. Build the Workspace
```bash
colcon build --packages-select pickplace_rl_mobile
source install/setup.bash
```

### 3. Verify Installation
```bash
ros2 pkg list | grep pickplace_rl_mobile
```

---

## Usage

### Full System Launch
Launch everything (Gazebo + perception + safety guard):
```bash
source install/setup.bash
ros2 launch pickplace_rl_mobile full_system.launch.py
```

### VLA Pipeline
```bash
# All 6 VLA nodes
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py

# Without ML models (zero extra dependencies)
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py use_llm:=false use_owlv2:=false

# Send a natural language command
ros2 topic pub /vla_instruction std_msgs/String "data: 'pick the blue cube and place in tray'"

# Multi-step task
ros2 topic pub /vla_instruction std_msgs/String "data: 'sort all objects by colour'"
```

### With Nav2 Navigation
```bash
ros2 launch pickplace_rl_mobile full_system.launch.py use_nav2:=true
```

### With Trained RL Policy
```bash
ros2 launch pickplace_rl_mobile full_system.launch.py use_rl:=true model_path:=./rl_models/pickplace_final_model.zip
```

### Gazebo Only (No Perception)
```bash
ros2 launch pickplace_rl_mobile gazebo_launch.py
```

### Visualization in RViz
View the robot model interactively:
```bash
source install/setup.bash
ros2 launch pickplace_rl_mobile display_launch.py
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

| Node | Topic/Service | Purpose |
|------|---------------|---------|
| `vla_language_node` | `/vla_instruction` → `/vla/structured_command` | Text → structured JSON (SmolLM2 / regex) |
| `vla_vision_node` | `/camera/image_raw` → `/perception/detected_object` | Camera → object poses (OWLv2 / HSV) |
| `object_memory_node` | `/perception/detected_object` → `/object_memory/query` | Persistent object map with 30 s decay |
| `task_planner_node` | `/vla/structured_command` → `/task_planner/next_action` | Multi-step task decomposition |
| `vla_coordinator_node` | Orchestrates language + memory + planning | Resolves poses and drives execution |
| `vla_action_node` | MoveIt2 `MoveGroup` action + `/gripper/cmd` | Cartesian motion planning + gripper |

---

## Training the RL Agent

### Quick Start Training
Train for 100k timesteps:
```bash
source install/setup.bash

# In terminal 1: Launch Gazebo
ros2 launch pickplace_rl_mobile gazebo_launch.py

# In terminal 2: Start training
ros2 run pickplace_rl_mobile train_rl --timesteps 100000 --save-dir ./rl_models
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 100,000 | Total training steps |
| `--save-dir` | `./rl_models` | Model checkpoint directory |

### Monitoring Training
```bash
tensorboard --logdir ./rl_models/tensorboard
```

Checkpoints saved every 10,000 steps. Evaluation runs every 5,000 steps.

---

## Testing the Trained Policy

After training, test the learned policy:
```bash
source install/setup.bash

# Launch Gazebo
ros2 launch pickplace_rl_mobile gazebo_launch.py

# In another terminal, test the policy
ros2 run pickplace_rl_mobile test_policy --model ./rl_models/pickplace_final_model.zip --episodes 5
```

The test script records camera snapshots and prints success rate + average reward.

---

## Checking Topics

After launching the full system, verify sensor data:
```bash
ros2 topic list | grep -E "camera|scan|perception|safety"
```

### Expected Topics

| Topic | Type | Source |
|-------|------|--------|
| `/camera/image_raw` | `sensor_msgs/Image` | RGB camera |
| `/camera/depth` | `sensor_msgs/Image` | Depth camera |
| `/scan` | `sensor_msgs/LaserScan` | 2D LiDAR |
| `/perception/detected_object` | `geometry_msgs/PoseStamped` | Perception node |
| `/perception/debug_image` | `sensor_msgs/Image` | Perception node |
| `/perception/markers` | `visualization_msgs/Marker` | Perception node |
| `/safety/status` | `std_msgs/String` | Safety guard |
| `/odom` | `nav_msgs/Odometry` | Diff drive |
| `/joint_states` | `sensor_msgs/JointState` | Joint state publisher |
| `/cmd_vel` | `geometry_msgs/Twist` | Base velocity |

---

## Configuration

### Training Hyperparameters
Edit `config/training_config.yaml`:
- Learning rate, buffer size, batch size
- Reward weights and episode length
- Algorithm parameters

### Nav2 Navigation
Edit `config/nav2_params.yaml`:
- AMCL localization parameters
- DWB controller speeds and acceleration limits
- Costmap resolution and obstacle detection settings

### Robot Model
Modify `urdf/pickplace_mobile_arm.urdf`:
- Link dimensions and joint limits
- Sensor parameters (camera resolution, LiDAR range)
- Inertial properties

### Gazebo World
Edit `worlds/pickplace_world.world`:
- Object bin position and size
- Target zone location
- Lighting and physics settings

---

## Results

### Training Performance
- **Algorithm**: SAC (Soft Actor-Critic)
- **Training Duration**: ~100k timesteps
- **Training Time**: ~2-4 hours (GPU recommended)
- **Success Rate**: 60-80% after full training

---

## Troubleshooting

### Build Issues
| Error | Solution |
|-------|----------|
| `Package not found` | Run `source install/setup.bash` |
| `CMake Error: ament_cmake` | `sudo apt install ros-jazzy-ament-cmake` |

### Gazebo Issues
| Error | Solution |
|-------|----------|
| Gazebo doesn't start | Check `gz sim --version` and ROS bridge |
| Robot falls through ground | Increase physics step size or check collision geometry |
| No camera/lidar data | Verify bridge topics with `ros2 topic list` |

### Training Issues
| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: gymnasium` | `pip install gymnasium stable-baselines3` |
| Training crashes | Reduce learning rate, increase buffer size |
| NaN rewards | Check physics settings and reward function |

---

## Roadmap

- [x] Mobile base with differential drive
- [x] 4-DOF UR3-based robotic arm with parallel gripper
- [x] SAC-based RL training pipeline
- [x] RGB-D camera perception pipeline
- [x] 2D LiDAR for obstacle detection
- [x] Safety guard with emergency stop
- [x] Nav2 navigation stack integration
- [x] Domain randomization for sim-to-real
- [x] VLA pipeline — open-vocabulary pick and place (SmolLM2 + OWLv2)
- [x] Natural language commanding (`/vla_instruction` topic)
- [x] Multi-step task planning (sort-all, clear-all, stack)
- [x] MoveIt2 integration for motion planning
- [ ] Real robot deployment (Jetson + RealSense + ARES)
- [ ] Dynamic obstacle avoidance using RL
- [ ] Sim-to-real transfer validation

---

## License
MIT License

## Maintainer
**Darsh Menon**
- Email: darshmenon02@gmail.com
- GitHub: [@darshmenon](https://github.com/darshmenon)

---

## Acknowledgments
- [ROS 2](https://docs.ros.org/) community for excellent documentation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL implementations
- [Gazebo](https://gazebosim.org/) simulation framework
- [Nav2](https://docs.nav2.org/) for navigation stack
- [OpenAI Gymnasium](https://gymnasium.farama.org/) for environment interface

---

## Citation
```bibtex
@software{pickplace_rl_mobile,
  author = {Menon, Darsh},
  title = {Pick-and-Place RL Mobile Manipulator},
  year = {2025},
  url = {https://github.com/darshmenon/pickplace-rl-mobile-manipulator}
}
```
