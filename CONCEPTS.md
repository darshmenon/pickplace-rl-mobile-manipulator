# RL Concepts Used in ARES

## 1. Reinforcement Learning Basics

**The core loop**
At every timestep the robot (agent) receives an observation from Gazebo (environment), chooses an action via its policy, and receives a scalar reward. The cycle repeats: `obs → action → reward → next_obs → …`. The objective is to find the policy that maximises the sum of discounted future rewards:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …    (γ = 0.99 here)
```

No demonstrations are used. The agent discovers useful behaviour purely from the reward signal.

**Policy and Value functions**
- **Policy π(s)** — maps observation → action (our MLP network)
- **Q-function Q(s,a)** — estimated cumulative reward if you take action `a` in state `s` and follow `π` thereafter. The critic learns this; the actor is updated to maximise it.
- **Advantage** — `A(s,a) = Q(s,a) - V(s)`: how much better action `a` is vs the average. Used in actor gradient updates.

**Observation Space (24-dim)**
```
joint_positions[6]   # arm joint angles (rad)
joint_velocities[6]  # arm joint speeds (rad/s)
finger_joint[1]      # gripper open/close position
ee_pos[3]            # end-effector XYZ in world frame
obj_pos[3]           # pickup object XYZ (real Gazebo pose via gz bridge)
grasped[1]           # binary: object currently grasped?
phase[1]             # current curriculum phase (1–5)
base_pose[3]         # base x, y, heading θ (rad)
```

All positional quantities are in the **world frame** (consistent reference), not mixed local/global frames which would confuse the network.

**Action Space (9-dim, continuous [-1, 1])**
```
joint_deltas[6]      # arm position delta per joint (×0.05 rad = ±2.9°/step)
gripper[1]           # >0 = close, <0 = open
base_linear[1]       # forward speed (phases 0, 4, 5 only; locked during grasping)
base_angular[1]      # turn speed   (phases 0, 4, 5 only)
```

---

## 2. SAC → TQC: Why We Upgraded

We started with **SAC (Soft Actor-Critic)** and switched to **TQC (Truncated Quantile Critics)**.

**SAC**
SAC adds an entropy term to the objective (`α·H(π)`), encouraging exploration by rewarding the policy for being stochastic. It uses two Q-networks and takes the minimum to reduce overestimation. Very sample-efficient on locomotion and simple manipulation.

**Why SAC struggled here**
Pick-and-place is contact-rich. Small positional errors (finger position vs cylinder) change Q-values drastically. SAC's min-of-two critics still overestimates in these discontinuous, high-variance reward regions, causing the policy to commit to phantom high-reward actions that fail at execution time.

**TQC (Kuznetsov et al. 2020)**
Instead of a point estimate of Q(s,a), TQC models the full *return distribution* as a mixture of quantile atoms. During Bellman target computation it **drops the top N quantiles** (pessimistic truncation), which biases estimates downward without throwing away useful signal:

```
Q_target = mean of bottom (num_atoms - top_quantiles_to_drop) quantiles
```

Effect: the policy is punished for overconfident Q-estimates in risky contact situations. In practice, TQC outperforms SAC on the majority of manipulation benchmarks (Gymnasium Robotics, dm_control).

**Key hyperparameters**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Standard Adam; lower → more stable but slower |
| `buffer_size` | 500 000 | Large off-policy replay; covers many episodes |
| `batch_size` | 512 | Larger = smoother gradient, fits in GPU memory |
| `tau` | 0.005 | Polyak averaging for target networks; small = stable |
| `gamma` | 0.99 | Discounts rewards 100 steps ahead by ~37% |
| `gradient_steps` | 2 | Two critic/actor updates per env step → faster learning |
| `top_quantiles_to_drop` | 2 | Pessimistic bias; higher = more conservative |
| `learning_starts` | 1000 | Fill replay with random actions first |

---

## 3. Phase-Based Curriculum Learning

Training end-to-end (random → place object) from scratch fails because:
1. The reward is extremely sparse — grasp success happens maybe 1 in 10 000 random rollouts
2. The exploration space is huge (9D continuous actions × 800 steps)

**Solution: curriculum over 5 phases**

Each phase has its own dense reward that only activates when the previous phase is complete:

```
Phase 1: Lower EE to grasp height AND approach object XY
         → transition when dist_z < 5cm AND dist_xy < 12cm

Phase 2: Bring EE to grasp target (2cm above object center)
         → transition when gripper closed AND EE within 8cm of object

Phase 3: Verify grasp (object must rise), lift EE to 25cm height
         → transition when EE at 25cm; abort → phase 2 if object doesn't rise

Phase 4: Navigate base + EE toward placement zone (x=0.6, y=0.5)
         → transition when EE within 15cm of target AND base aligned

Phase 5: Position EE over target, open gripper
         → episode success (+1000) when EE < 8cm and gripper open
```

The scripted pre-grasp handles base approach before phase 1; the RL policy controls everything from phase 1 onward.

**Why curriculum works**
Each phase is a strictly easier problem than the full task. The agent learns to solve phase 1 in hundreds of episodes; phase 2 in thousands; etc. Without curriculum the agent would need millions of steps just to accidentally grasp an object once.

---

## 4. Potential-Based Reward Shaping

**The sparse-reward problem**
Phase transitions give large bonuses (+100 to +500), but those only happen occasionally. Between transitions, the agent gets no signal about whether it is improving.

**Potential-based shaping (Ng et al. 1999)**
For any distance metric `d(s)` toward the sub-goal, define:

```
F(s, s') = Φ(s') - Φ(s)    where Φ(s) = -d(s)
         = d(s) - d(s')      (reward = reduction in distance)
```

In code:
```python
reward += (prev_distance - current_distance) × scale
```

This is guaranteed **policy-invariant**: adding shaping does not change which policy is optimal, it only makes the gradient denser. Every step now carries signal.

**Asymmetric shaping (new)**
Moving away from the object is penalised more harshly than approaching is rewarded:
```python
delta = prev_dist - current_dist
reward += delta * 50   if delta > 0   # approaching: normal reward
reward += delta * 150  if delta < 0   # retreating: 3× harsher penalty
```
This breaks the symmetry so the agent strongly prefers staying near the object even when uncertain.

**Scales by phase**
| Phase | Metric | Approach | Retreat |
|-------|--------|----------|---------|
| 1 | `dist_z + 0.5×dist_xy` | ×50 | ×150 |
| 2 | 3D to grasp target | ×30 | ×120 |
| 3 | Z to 25cm lift | ×50 | ×50 |
| 4 | XY + base + heading err | ×50 | ×50 |
| 5 | 3D to place pos | ×50 | ×50 |

---

## 5. Scripted Pre-Grasp (Hierarchical RL / Options)

**Why not learn navigation too?**
A differential-drive base needs to: (1) turn to face the bin, (2) drive forward to ~25cm, (3) stop precisely. This navigation sub-task has nothing to do with arm control and takes ~300 steps. Having the RL policy also learn navigation multiplies sample requirements by ~10×.

**Solution: scripted primitive + RL handoff**
A P-controller executes the approach every reset:
```python
angle_err = atan2(obj_y - base_y, obj_x - base_x) - base_theta
angular_vel = clip(angle_err × 2.0, -1, 1)
linear_vel  = clip((dist - 0.25) × 1.5, 0, 0.4) if |angle_err| < 0.4 else 0
# arm tucked: shoulder_lift → -1.2 rad, elbow → +1.2 rad
```
Runs for up to 300 steps, exits when `dist < 27cm`. Then RL takes over.

This is equivalent to the **options framework** in hierarchical RL: a high-level option (navigate) terminates and passes control to a low-level option (manipulate). The RL policy never needs to explore navigation — it always starts positioned correctly.

**Base locked during manipulation**
During phases 1–3, base linear/angular velocity actions are zeroed regardless of what the policy outputs. This prevents the agent from accidentally driving away from the object mid-grasp.

---

## 6. Position-Delta Control

**Raw velocity control** requires the agent to discover that holding small consistent velocity over many steps moves the arm smoothly. High velocities crash the arm; zero velocity does nothing. The agent has to learn PID-like behaviour from scratch.

**Position-delta control** reframes actions as incremental position targets:
```python
target_pos  = current_joint_pos + action × 0.05   # ±0.05 rad max per step
vel_command = clip((target_pos - current_pos) × 10, -0.5, 0.5)
```

Benefits:
- Action 0 → arm stays still (stable default)
- Action ±1 → arm moves at most 0.05 rad (≈ 2.9°) per step — physically interpretable
- No velocity explosion possible (hard clamp at ±0.5 rad/s)
- Natural impedance: the P-gain of 10 gives stiff position tracking

This is mathematically equivalent to an inner-loop PD controller driven by the RL policy's desired position increments.

---

## 7. How EE Position Is Computed

Getting a reliable end-effector (EE) position every step is a 4-stage pipeline.

**Stage 1 — Read joint angles from ROS2**

The `/joint_states` topic is published by Gazebo via the ros_gz_bridge at ~1 kHz. The callback captures:
```python
self.joint_positions = np.array(msg.position[:9])
# 0=shoulder_pan  1=shoulder_lift  2=elbow
# 3=wrist_1       4=wrist_2        5=wrist_3   6=finger
```

**Stage 2 — Forward Kinematics (DH chain)**

We analytically compute EE position from joint angles using UR3 Denavit-Hartenberg parameters — no TF lookup, no latency.

Each joint `i` contributes a 4×4 homogeneous transform:
```
T_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)
```

Chain all 6: `T = T_1 · T_2 · T_3 · T_4 · T_5 · T_6`

The EE position in the arm's own base frame is the last column: `T[:3, 3] = [x, y, z]`

UR3 DH parameters used:
```python
_UR3_DH = [
    (0.0,      0.1519,  π/2),   # shoulder_pan
    (-0.24365, 0.0,     0.0),   # shoulder_lift
    (-0.21325, 0.0,     0.0),   # elbow
    (0.0,      0.11235, π/2),   # wrist_1
    (0.0,      0.08535, -π/2),  # wrist_2
    (0.0,      0.0819,  0.0),   # wrist_3
]
```

**Stage 3 — Fix the 180° yaw mount**

The UR3 URDF mounts `base_link_inertia` with `rpy="0 0 π"` so the arm physically faces the robot's +x (forward). The DH chain assumes the arm faces its own -x. Without correction, all EE positions would be mirrored.

Fix — negate x and y after FK:
```python
ee_in_arm_base = ur3_fk(joint_positions[:6])
ee_local = np.array([-ee_in_arm_base[0],   # flip x
                     -ee_in_arm_base[1],   # flip y
                      ee_in_arm_base[2]])  # z unchanged
ee_local += [0, 0, 0.1]   # arm mount is 10cm above chassis origin
```

**Stage 4 — Transform to world frame**

The chassis pose `(base_x, base_y, θ)` comes from odometry. Apply a 2D rotation to bring the local EE offset into world coordinates:
```python
gx = base_x + ee_local_x·cos(θ) - ee_local_y·sin(θ)
gy = base_y + ee_local_x·sin(θ) + ee_local_y·cos(θ)
gz = 0.08 + ee_local_z      # 0.08m = chassis spawn height
```

This world-frame EE position is used directly in both the observation vector and the reward distance calculations.

**Why not use ROS2 TF?**
The TF tree involves a bridge round-trip (Gazebo → gz_bridge → ROS2 TF buffer → lookup). That adds variable latency and can return stale transforms mid-step. Direct FK from the same joint_states message is synchronous and deterministic.

**Frame consistency requirement**
All positional quantities in the observation must be in the same frame:
- `ee_pos` → world frame (via stages 1–4 above)
- `obj_pos` → world frame (from Gazebo dynamic_pose bridge)
- `base_pose` → world frame (from odometry)

If `ee_pos` were in chassis-local frame while `obj_pos` is in world frame, the network would see the relative error `ee_pos - obj_pos` rotate with the base heading — the same physical approach would look different depending on which direction the robot faces. Keeping everything in world frame eliminates this ambiguity.

---

## 8. Grasp Verification (Sim Feedback Loop)

After the gripper closes, the policy sets `object_grasped = True` and transitions to phase 3 (lift). But the gripper may have closed on air or grazed the object without a real grip.

**Verification via real Gazebo pose**
```python
# In phase 3, after 10 steps:
if real_object_pos[2] < 0.06:   # object z hasn't risen above 6cm
    object_grasped = False
    current_phase = 2            # back to grasping
    reward -= 10
```

This uses the `/world/pickplace_world/dynamic_pose/info` Gazebo bridge topic to get ground-truth object position — not an estimate. The penalty is mild (-10) to discourage fake grasps without catastrophically punishing exploration near the grasp zone.

---

## 9. Domain Randomisation

**Why it matters**
Without randomisation the policy overfits to a single object position. At test time, even a 1cm shift causes failure because the policy learned an open-loop motion rather than closed-loop error correction.

**What we randomise**
```python
ox = 0.6 + rng.uniform(-0.03, 0.03)   # ±3cm XY
oy = 0.0 + rng.uniform(-0.03, 0.03)
```
Object is respawned via Gazebo's `/world/pickplace_world/set_pose` service each reset. The policy must observe `ee_pos - obj_pos` and correct for the offset — it cannot memorise a fixed trajectory.

**Future: increase randomisation range** to ±5cm XY and add slight height variation as training matures.

---

## 10. Replay Buffer and Off-Policy Learning

TQC (like SAC) is **off-policy**: it can learn from experience collected by older versions of the policy. This is stored in a **replay buffer** (500 000 transitions here).

Each training step:
1. Collect 1 env step with current policy → store `(s, a, r, s', done)` in buffer
2. Sample random mini-batch of 512 transitions from buffer
3. Run 2 gradient updates (critic loss = Bellman TD error; actor loss = -Q)

**Why off-policy is critical for manipulation**
Manipulation is rare-event-dominated: most steps are near-grasp exploration that never quite succeeds. The replay buffer lets the agent reuse near-success transitions many times, amplifying the learning signal from each rare positive experience.

**Buffer fill strategy**
`learning_starts=1000`: the first 1000 steps use a random policy to populate the buffer before any gradient steps. This prevents early Q-network collapse from training on highly correlated sequential data.
