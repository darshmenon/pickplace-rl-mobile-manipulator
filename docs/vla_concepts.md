# Vision-Language-Action (VLA) Architecture

## 1. Core Principles

A VLA system maps `(Image, Instruction) → Robot Action`. It connects three domains:
- **Vision** — perceiving the world from camera/depth data
- **Language** — understanding natural language commands
- **Action** — planning and executing robot motion

---

## 2. Pipeline Overview

```
User: "pick the blue cube and place it in the tray"
          │
          ▼
┌──────────────────────┐
│  vla_language_node   │  SmolLM2-360M-Instruct (+ regex fallback)
│  /vla_instruction    │  Parses intent → structured JSON
└──────────┬───────────┘
           │ /vla/structured_command
           ▼
┌──────────────────────┐
│  task_planner_node   │  Decomposes into atomic task queue
│                      │  Supports: single, sort-all, clear, stack
└──────────┬───────────┘
           │ /vla/current_task
           ▼
┌──────────────────────┐     ┌───────────────────────┐
│  vla_coordinator     │◄────│  object_memory_node   │
│                      │     │  Persistent object map │
│  Resolves pick/place │     │  30s occlusion decay  │
│  poses & triggers    │     └───────────────────────┘
│  action execution    │             ▲
└──────────┬───────────┘             │ /vla/world_state
           │ /vla/action_target      │
           │ /execute_vla_sequence   │
           ▼                         │
┌──────────────────────┐     ┌───────┴───────────────┐
│  vla_action_node     │     │  vla_vision_node       │
│  MoveIt2 MoveGroup   │     │  OWLv2 (open-vocab)   │
│  joint planning &    │     │  + HSV fallback        │
│  gripper control     │     │  /vla/detected_object_ │
└──────────────────────┘     │  pose + /vla/world_    │
                             │  state                  │
                             └───────────────────────-─┘
```

---

## 3. Node Reference

### `vla_language_node.py`
- **Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` (360M params, CPU-runnable)
- **Fallback:** Regex + keyword matching
- **Input:** `/vla_instruction` (raw text string)
- **Output:** `/vla/structured_command` (JSON: action, color, object, destination, confidence, parser)
- **Param:** `use_llm: true/false`

Example output:
```json
{"action": "pick_and_place", "color": "blue", "object": "cube",
 "destination": "tray", "confidence": 0.92, "parser": "smollm2"}
```

---

### `vla_vision_node.py`
- **Model:** `google/owlv2-base-patch16-finetuned` (open-vocabulary)
- **Fallback:** HSV colour segmentation (red, blue, green, yellow, orange, purple)
- **Input:** `/camera/image_raw`, `/camera/depth`, `/vla_track_object` (text query)
- **Output:** `/vla/detected_object_pose` (PoseStamped), `/vla/world_state` (JSON)
- **Param:** `use_owlv2: true/false`, `detection_threshold: 0.1`

With OWLv2, the `/vla_track_object` topic accepts full text queries like `"blue coffee mug"` instead of just colour names.

---

### `task_planner_node.py` *(new)*
- **Input:** `/vla/structured_command`, `/vla/world_state`, `/vla/task_feedback`
- **Output:** `/vla/current_task` (dispatched at 2Hz), `/vla/task_queue`, `/vla/planner_status`

Supported decomposition patterns:

| Instruction pattern | Behaviour |
|---------------------|-----------|
| `"pick X and place in Y"` | Single task |
| `"sort all objects"` | One task per detected colour → colour-matched bin |
| `"clear all objects"` | Move all detected objects to default tray |
| `"stack all blocks"` | Stack them at centre with increasing Z |

---

### `object_memory_node.py` *(new)*
- **Input:** `/vla/world_state`
- **Output:** `/vla/object_map` (full timestamped map), `/vla/object_summary`
- **Service:** `/vla/query_object_map` (Trigger → returns JSON map)
- **Param:** `decay_seconds: 30.0` — objects not seen for 30s are evicted

Keeps last-known positions for objects temporarily occluded by the robot arm.

---

### `vla_coordinator_node.py`
- **Input:** `/vla/current_task`, `/vla/object_map`, `/vla/detected_object_pose`
- **Output:** `/vla/action_target` (pick+place poses JSON), `/vla/task_feedback`
- **Service client:** `/execute_vla_sequence`

Resolution priority for pick pose:
1. Object memory (exact colour match)
2. Object memory (partial label match)
3. Live tracked pose from vision node

---

### `vla_action_node.py`
- **MoveIt2:** `moveit_msgs/action/MoveGroup` action client
- **Fallback:** Gripper velocity commands via `/left_finger_joint/cmd_vel` etc.
- **Input:** `/vla/action_target` (JSON), `/joint_states`
- **Service:** `/execute_vla_sequence` (Trigger)

Execution sequence: ready → open gripper → pre-pick → grasp → lift → carry → lower → release → home

---

## 4. Launch

```bash
# Full pipeline (with ML models)
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py

# Disable ML models (regex + HSV only, no GPU/download needed)
ros2 launch pickplace_rl_mobile vla_full_pipeline.launch.py use_llm:=false use_owlv2:=false

# Send a command
ros2 topic pub /vla_instruction std_msgs/String "data: 'pick the blue cube and place in tray'"

# Multi-step: sort all
ros2 topic pub /vla_instruction std_msgs/String "data: 'sort all objects by colour'"

# Monitor pipeline
ros2 topic echo /vla/planner_status
ros2 topic echo /vla/object_map
```

---

## 5. ML Model Setup

```bash
pip install transformers torch pillow

# Pre-download models (optional, avoids first-run delay)
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct')
AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct')
"

python3 -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
Owlv2Processor.from_pretrained('google/owlv2-base-patch16-finetuned')
Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-finetuned')
"
```

Model sizes:
- SmolLM2-360M-Instruct: ~700 MB
- OWLv2-base-patch16: ~590 MB

---

## 6. Why Modular Over End-to-End?

End-to-end models (RT-2, OpenVLA 7B) require massive GPU, datasets, and are hard to debug.
A modular pipeline gives:

- Individual subsystem testing without moving hardware
- Swappable backends (regex → SmolLM2, HSV → OWLv2 → YOLOv8)
- Deterministic safety bounds via `safety_guard`
- CPU-runnable on Jetson or embedded hardware
- Interpretable intermediate representations for debugging
