# RL Concepts Used in ARES

## 1. Reinforcement Learning Basics

**Agent, Environment, Reward**
The robot (agent) takes actions in Gazebo (environment) and receives a scalar reward signal each step. The goal is to learn a policy that maximises cumulative reward. No labelled demonstrations — the agent discovers behaviour entirely from trial and error.

**Policy**
A function mapping observations → actions. Here it is a multi-layer perceptron (MLP) taking 24 numbers (joint angles, EE position, object position, phase…) and outputting 9 continuous values (arm joint deltas, gripper, base velocity).

**Observation Space (24-dim)**
```
joint_positions[6]   # arm joint angles (rad)
joint_velocities[6]  # arm joint speeds (rad/s)
finger_joint[1]      # gripper open/close position
ee_pos[3]            # end-effector XYZ in world frame
obj_pos[3]           # pickup object XYZ (real Gazebo pose)
grasped[1]           # binary: is object grasped?
phase[1]             # current curriculum phase (1-5)
base_pose[3]         # base x, y, heading (rad)
```

**Action Space (9-dim, continuous [-1, 1])**
```
joint_deltas[6]      # position delta per arm joint (×0.05 rad)
gripper[1]           # >0 = close, <0 = open
base_linear[1]       # forward speed (phases 0, 4, 5 only)
base_angular[1]      # turn speed  (phases 0, 4, 5 only)
```

---

## 2. TQC — Truncated Quantile Critics

We use **TQC** (Kuznetsov et al. 2020) from `sb3-contrib` instead of vanilla SAC.

**Why not SAC?**
SAC uses a single Q-value estimate per network. In manipulation, Q-values tend to be overestimated because of function approximation error, leading the policy to choose actions that look good on paper but fail in practice.

**What TQC adds**
TQC models the *distribution* of returns using quantile regression (multiple "atoms" per critic). It then **drops the top quantiles** (controlled by `top_quantiles_to_drop_per_net=2`) before computing the Bellman target. This pessimistically biases Q-estimates downward, reducing overestimation and producing more conservative, reliable policies — important for contact-rich manipulation.

**Key hyperparameters**
| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | 3e-4 | Standard Adam LR |
| `buffer_size` | 500 000 | Large replay for off-policy stability |
| `batch_size` | 512 | Larger batch = smoother gradient |
| `tau` | 0.005 | Soft target update (stable) |
| `gamma` | 0.99 | Long-horizon discount |
| `gradient_steps` | 2 | Two gradient updates per env step |
| `top_quantiles_to_drop` | 2 | Pessimistic bias against overestimation |

---

## 3. Phase-Based Curriculum Learning

Training a policy to do everything at once (drive → approach → grasp → lift → transport → place) from scratch is too hard — the reward is too sparse and the exploration space too large.

**Solution: phase curriculum**
Break the task into 5 sequential sub-tasks. Each phase has its own dense reward. The agent only moves to the next phase when it achieves the sub-goal.

```
Phase 1 → Lower EE to grasp height + approach object in XY
Phase 2 → Bring EE to grasp target (2cm above object) + close gripper
Phase 3 → Verify grasp (object rises), lift EE to 25cm
Phase 4 → Transport base + EE toward placement target
Phase 5 → Position EE over target, open gripper
```

The scripted pre-grasp navigator handles base approach before phase 1 hands off to the RL policy.

---

## 4. Potential-Based Reward Shaping

Each phase uses a **potential-based** dense reward:

```
reward += (prev_distance - current_distance) × scale
```

This gives a positive reward when getting closer to the sub-goal and negative when moving away. It is **policy-invariant** (Ng et al. 1999) — it does not change the optimal policy, only the gradient signal density.

Without shaping, rewards are sparse (only at phase transitions). With shaping, every step gives meaningful gradient.

**Scales used**
| Phase | Shaping metric | Scale |
|-------|---------------|-------|
| 1 | `dist_z + 0.5×dist_xy` to grasp height | ×50 |
| 2 | 3D distance to grasp target | ×30 |
| 3 | Z distance to lift height (25cm) | ×50 |
| 4 | Combined XY + base distance to placement | ×50 |
| 5 | 3D distance to place position | ×50 |

---

## 5. Scripted Pre-Grasp (Hierarchical / Options)

Driving a differential-drive base from an arbitrary position to a precise grasp stance purely by RL is extremely sample-inefficient. Instead:

- A **P-controller** (scripted, not learned) drives the base to ~25cm from the bin while keeping the arm tucked
- RL takes over **after** the base is positioned

This is a form of **hierarchical RL** (options framework): one high-level primitive (navigate) + one learned policy (manipulate). It drastically reduces the effective state space for the RL policy.

---

## 6. Position-Delta Control

The arm joints use **position-delta** control:

```python
target = current_joint_pos + action * 0.05   # max 0.05 rad per step
vel_cmd = clip((target - current) * 10, -0.5, 0.5)
```

This is easier to learn than raw velocity control because:
- Actions have physical units the agent can reason about (small = fine motion)
- Accidental large velocities can't happen (clamped to ±0.5 rad/s)
- Stops naturally when action → 0 (stable at any pose)

---

## 7. Grasp Verification

After the gripper closes (phase 2 → 3), the policy cannot trust its own grasped flag — the object may have slipped. We verify using **real Gazebo object Z position**:

```python
if grasp_verify_steps > 10 and real_object_pos[2] < 0.06:
    # Object didn't rise → false grasp
    revert to phase 2, penalise -10
```

This closes the feedback loop with ground-truth simulation state, preventing the agent from learning to "fake" a grasp.

---

## 8. Domain Randomisation

Object XY position is randomised ±3cm each episode:

```python
ox = 0.6 + rng.uniform(-0.03, 0.03)
oy = 0.0 + rng.uniform(-0.03, 0.03)
```

Without this the policy memorises a single location and fails to generalise even within simulation. Randomisation forces the policy to learn a reactive EE-to-object error correction rather than an open-loop motion primitive.
