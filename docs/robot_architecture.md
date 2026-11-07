# Robot Architecture Document

This document explains the hardware configuration, URDF details, sensors, and mobile robot architecture of the ARES Pick-and-Place RL Mobile Robot.

## 1. URDF (Unified Robot Description Format)
The robot is defined in a single URDF file (`pickplace_mobile_arm.urdf`). 
The URDF contains the visual, collision, and inertial properties of the robot, combining a custom differential drive mobile base with a standard 6-DOF UR3 robotic arm and a parallel gripper.

### 1.1 Mobile Base Architecture
The base of the robot is a differential-drive chassis:
- **Chassis**: A rectangular box (`0.45m x 0.35m x 0.08m`) that acts as the main body.
- **Wheels**: Two driven wheels (left and right) are attached to the chassis via continuous joints.
- **Odometry & Control**: The base is controlled using the Gazebo `DiffDrive` plugin, which subscribes to `/cmd_vel` (Twist messages) and publishes odometry data on the `/odom` topic at 50Hz.
- **Safety Base Workspace**: The RL environment enforces that the mobile base must stay within a configurable 3m radius from the world origin.

### 1.2 Robotic Arm (6-DOF UR3-based)
Mounted on top of the chassis is a 6-DOF UR3-based robotic arm. The joints consist of:
1. **Shoulder Pan Joint**: Revolute (Z-axis) - pans left/right.
2. **Shoulder Lift Joint**: Revolute (Y-axis) - lifts up/down.
3. **Elbow Joint**: Revolute (Y-axis) - bends the forearm.
4. **Wrist 1 Joint**: Revolute (Y-axis) - pitches the wrist.
5. **Wrist 2 Joint**: Revolute (Z-axis) - yaws the wrist.
6. **Wrist 3 Joint**: Continuous (Y-axis) - rolls the gripper.

The URDF defines strict joint limits (both position and velocity) for safety. Each joint has a Gazebo `JointController` plugin that accepts velocity commands from the RL policy.

### 1.3 End-Effector (Parallel Gripper)
The end-effector is a parallel jaw gripper with two prismatic finger joints moving along the Y-axis. The gripper accepts velocity commands to open or close during the picking and placing phases of the task.

## 2. Sensors

For the RL policy and the safety guard system to work, the robot is equipped with several simulated sensors defined in the URDF:

### 2.1 RGB-D Camera
- **RGB Camera**: Mounted at the front center of the robot, angled slightly downward (pitch of 0.3 rad). It publishes 640x480 images at 30Hz to the `/camera/image_raw` topic.
- **Depth Camera**: Co-located with the RGB camera, publishing depth data (R_FLOAT32) at 15Hz to the `/camera/depth` topic.
- **Purpose**: Feeds the perception pipeline to perform HSV color segmentation (detecting the red object) and depth back-projection to calculate the 3D position of the object in the camera frame.

### 2.2 2D LiDAR
- **Specification**: A 360-degree, 640-sample 2D LiDAR sensor mounted on top of the chassis.
- **Topic**: Publishes `sensor_msgs/LaserScan` messages at 10Hz to the `/scan` topic, with a range of 0.1m to 12.0m.
- **Purpose**: Used for obstacle detection by the `safety_guard.py` node (triggering an emergency stop if an obstacle is within 25cm) and for the Nav2 AMCL localization and costmaps.

### 2.3 Wheel Encoders & Joint States
- **Odometry**: The `DiffDrive` plugin simulates wheel encoders to produce odometry (`/odom`). With `drive:=true` the base uses `mobile_ur3_vel.urdf` instead, where Gazebo's velocity-control plugin moves the model and an odometry publisher reports its world pose (see [gazebo_drive_traction.md](gazebo_drive_traction.md)).
- **Proprioception**: The `JointStatePublisher` system plugin publishes current positions and velocities of all arm and wheel joints to the `/joint_states` topic, forming the core of the RL agent's observation space.
