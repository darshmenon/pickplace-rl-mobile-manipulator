# System Architecture & Future Improvements

## 1. System Architecture

The software stack follows a modular **Sense-Plan-Act** pipeline on ROS 2 Jazzy + Python 3.10+.

```
┌──────────────────────────── Gazebo Harmonic ───────────────────────────────┐
│  Diff Drive  │  RGB Cam        │  Depth Cam      │  2D LiDAR  │  Encoders  │
│  /cmd_vel    │  /camera/image  │  /camera/depth  │  /scan     │  /odom     │
│  /odom       │  _raw           │                 │            │            │
└──────┬───────┴────────┬────────┴────────┬─────────┴─────┬──────┴────────────┘
       │                │                 │               │
       ▼                ▼                 ▼               ▼
  ManipRL Node    vla_vision_node    vla_vision_node  Safety Guard
  (SAC policy)    OWLv2 (primary)    HSV (fallback)   joint limits
  20 Hz           open-vocabulary    colour masks     obstacle e-stop
       │                │
       │          object_memory_node (30s decay)
       │                │
       │          task_planner_node
       │          (sort-all, clear, stack, single)
       │                │
       │          vla_coordinator_node
       │          (resolves pick/place poses)
       │                │
       └────────→ vla_action_node
                  MoveIt2 MoveGroup
                  gripper velocity fallback
```

---

## 2. Node Descriptions

### 2.1 Perception (`perception_node.py`)
Standalone RGB-D object detector used by the RL pipeline.
- Converts RGB→HSV (pure NumPy, no OpenCV dependency)
- Samples 10×10 depth neighbourhood for robust depth estimate
- Back-projects to 3D camera frame via pinhole model
- Publishes `/perception/detected_object` (PoseStamped) + RViz markers

### 2.2 VLA Vision (`vla_vision_node.py`)
Used by the VLA pipeline. Dual-mode:
- **Primary:** `google/owlv2-base-patch16-finetuned` — open-vocabulary queries like `"red coffee mug"` (requires `transformers torch pillow`)
- **Fallback:** HSV segmentation for red/blue/green/yellow/orange/purple
- Accepts arbitrary text queries via `/vla_track_object` when OWLv2 is active
- Publishes per-object poses and a JSON world state

### 2.3 Language Parser (`vla_language_node.py`)
- **Primary:** `HuggingFaceTB/SmolLM2-360M-Instruct` (360M params, CPU-only)
- **Fallback:** Regex + keyword extraction
- Outputs structured JSON: `{action, color, object, destination, confidence, parser}`
- Disable with `use_llm:=false` for zero-dependency offline mode

### 2.4 Object Memory (`object_memory_node.py`)
- Subscribes to `/vla/world_state` and maintains timestamped positions
- Keeps last-known location for up to 30s after an object disappears from view
- Service `/vla/query_object_map` returns full JSON map
- Prevents task failures from momentary occlusion during arm motion

### 2.5 Task Planner (`task_planner_node.py`)
Decomposes high-level goals into ordered atomic task queues:

| Instruction | Behaviour |
|-------------|-----------|
| `"pick blue cube and place in tray"` | Single pick-and-place |
| `"sort all objects by colour"` | One task per detected colour → matching bin |
| `"clear all objects"` | Move all objects to default tray |
| `"stack all blocks"` | Stack at centre with increasing Z |

Advances queue only after coordinator confirms task completion via `/vla/task_feedback`.

### 2.6 Coordinator (`vla_coordinator_node.py`)
- Receives current task from task planner
- Resolves pick pose: object memory → partial label match → live tracked pose
- Resolves place pose from task's `place_xyz` field
- Publishes poses to action node, waits for result, reports feedback

### 2.7 Action Node (`vla_action_node.py`)
- **Primary:** MoveIt2 `MoveGroup` action client — joint-space planning to home/ready configs, gripper control
- **Fallback:** Direct velocity commands to arm joints
- Sequence: ready → open gripper → pre-pick → grasp → lift → carry → lower → release → home

### 2.8 Safety Guard (`safety_guard.py`)
- 20 Hz monitoring of joint limits, workspace bounds, and LiDAR obstacles
- Issues emergency `/cmd_vel` stop if obstacle <0.25 m or EE <0.02 m from ground
- Publishes JSON status: `{severity, violations[], ee_position, min_obstacle_dist, e_stop}`

### 2.9 RL Node (`manip_rl_node.py`)
- Loads trained SAC model from `./rl_models/pickplace_final_model.zip`
- 16-dim observation (joints, EE, object pose, phase, base pose)
- 8-dim action (joint velocities, gripper, base linear/angular)
- Runs inference at 20 Hz

---

## 3. Navigation Stack

### 3.1 Obstacle-Avoidance Navigator (`navigation.py`)
Custom FSM-based navigator without Nav2 dependency:
- States: GOAL_SEEK → FIND_CLEAR → MOVE_CLEAR → REALIGN
- LiDAR front-sector obstacle detection, 5-direction escape scanning
- 20 Hz control loop

### 3.2 Frontier Explorer (`frontier_explorer.py`)
- Reads `/map` occupancy grid from SLAM toolbox
- Labels frontier cells (free bordering unknown) using scipy `binary_dilation`
- Sends Nav2 `navigate_to_pose` goals to nearest unvisited frontier
- Repeats until map is fully explored

### 3.3 Waypoint Navigator (`waypoint_nav.py`)
- Loads ordered waypoints from YAML
- Uses Nav2 action client to execute each in sequence

---

## 4. RL Training Pipeline

```
PickPlaceEnv (Gymnasium)
    ↕ 16-dim obs / 8-dim action
SAC Agent (Stable-Baselines3)
    ↓ saves checkpoints every 10k steps
./rl_models/pickplace_final_model.zip
    ↓ loaded by manip_rl_node
Runtime inference at 20 Hz
```

Training: `ros2 launch pickplace_rl_mobile rl_train.launch.py`
Monitor:  `tensorboard --logdir ./rl_models/tensorboard`

Domain randomisation (per episode): object position, target position, colour HSV, mass, friction, gravity noise.

---

## 5. Topic Reference

| Topic | Type | Direction | Hz | Purpose |
|-------|------|-----------|-----|---------|
| `/vla_instruction` | String | in | — | Raw text command |
| `/vla/structured_command` | String (JSON) | internal | event | Parsed intent |
| `/vla/current_task` | String (JSON) | internal | 2 | Active atomic task |
| `/vla/object_map` | String (JSON) | internal | 1 | Persistent object positions |
| `/vla/world_state` | String (JSON) | internal | 2 | Live detected objects |
| `/vla/detected_object_pose` | PoseStamped | internal | 30 | Live tracked pose |
| `/vla/action_target` | String (JSON) | internal | event | Pick+place poses |
| `/vla/task_feedback` | String (JSON) | internal | event | Task completion status |
| `/vla/planner_status` | String (JSON) | out | event | Queue status |
| `/safety/status` | String (JSON) | out | 20 | Safety state |
| `/perception/detected_object` | PoseStamped | out | 10 | RL perception |
| `/cmd_vel` | Twist | out | 20 | Base velocity |

---

## 6. Roadmap

### Done
- [x] Mobile base + 6-DOF arm + parallel gripper URDF
- [x] RGB-D perception (HSV) + RViz debug overlay
- [x] SAC RL environment + domain randomisation
- [x] Real-time safety guard with e-stop
- [x] Nav2 navigation + frontier-based exploration
- [x] VLA pipeline skeleton (Phases 1-4)
- [x] SmolLM2-360M-Instruct language parser
- [x] OWLv2 open-vocabulary vision node
- [x] Task planner (sort/clear/stack/single)
- [x] Object memory with occlusion decay

### Next
- [ ] `navigate_and_pick` integration — navigate to detected object, pick, navigate to drop zone
- [ ] Voice input via OpenAI Whisper
- [ ] Trained RL model included in repo (or download script)
- [ ] Real robot deployment: Jetson Orin + RealSense D435
- [ ] Benchmark evaluation node (success rate, task time)
- [ ] Multi-robot coordination via `/vla_robot_N/` namespacing
