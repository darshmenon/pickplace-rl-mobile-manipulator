# Resume - ARES (Autonomous Robotic End-to-End System)
**GitHub:** [https://github.com/darshmenon](https://github.com/darshmenon)

- Developed ARES, an autonomous mobile manipulator combining a differential-drive base with a 6-DOF UR3 arm and Robotiq 2F-85 gripper for pick-and-place operations learned entirely from reinforcement learning — no demonstrations and no motion planning in the RL loop.
- Trained a TQC (Truncated Quantile Critics) agent via SB3-Contrib within a custom ROS 2-integrated Gymnasium environment at 20 Hz, achieving multi-phase manipulation across a 46-dimensional observation space and 9-dimensional continuous action space.
- Designed a 5-phase curriculum (reach → grasp → lift → transport → place) with milestone bonuses (+100 to +1000), asymmetric retreat penalties, and automated stage advancement/reversion driven by evaluation reward thresholds.
- Implemented analytical UR3 DH forward kinematics for zero-latency end-effector world-frame positioning, eliminating TF bridge latency from the reward computation loop.
- Built a grasp verification system that cross-checks the Gazebo ground-truth object z-pose over up to 30 steps before confirming a lift — preventing false-positive grasp signals from contact noise.
- Applied domain randomization encompassing object position, size, color, mass, friction, and gravity perturbation, with per-stage widening of positional ranges and object shape generalization across box, cylinder, and sphere geometries.
- Deployed VecNormalize for online normalization of all 46 observation dimensions and reward, critical for stable learning across mixed-scale inputs (joint angles, metric positions, velocities).
- Engineered a hierarchical control architecture: a scripted P-controller handles base approach and arm pre-positioning (caster-aware clearance, bin-wall avoidance); TQC then takes over for contact-rich manipulation stages.
- Instrumented training with rolling 100-episode success rate logging, per-episode phase diagnostics, and curriculum-stage tracking persisted to monitor CSVs for offline analysis.
