# Resume - ARES (Autonomous Robotic Environment System)
**GitHub:** [https://github.com/darshmenon](https://github.com/darshmenon)

- Developed an autonomous mobile manipulator combining a differential-drive base with a 6-DOF UR3-based arm for open-vocabulary pick-and-place operations.
- Trained a Soft Actor-Critic (SAC) reinforcement learning agent via Stable-Baselines3 within a custom ROS 2-integrated Gymnasium (`PickPlaceEnv`) environment for 20 Hz low-level continuous robotic control.
- Designed a custom 16-dimensional observation space and 8-dimensional continuous action space mapping joint kinematics, end-effector tracking, and base odometry.
- Engineered a Vision-Language-Action (VLA) pipeline bridging a local SmolLM2 model for language understanding with OWLv2 for open-vocabulary semantic vision.
- Deployed MoveIt 2 alongside continuous velocity override control for robust manipulation and executed motion plans.
- Constructed a persistent object memory node with 30-second occlusion decay, enabling complex multi-step task decomposition (e.g., sorting, stacking).
- Integrated Nav2 and SLAM with a custom obstacle-avoidance FSM and frontier-based autonomous exploration.
- Enforced system safety with real-time joint limit constraints, strict workspace boundaries, and a 20 Hz LiDAR-triggered E-stop mechanism.
- Facilitated sim-to-real transfer by implementing domain randomization encompassing position, color, and simulated physics noise.
