#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
import time

# UR3 DH parameters: (a_m, d_m, alpha_rad) per joint
# Source: ur_description/config/ur3/default_kinematics.yaml
_UR3_DH = [
    (0.0,      0.1519,  np.pi / 2),   # shoulder_pan
    (-0.24365, 0.0,     0.0),          # shoulder_lift
    (-0.21325, 0.0,     0.0),          # elbow
    (0.0,      0.11235, np.pi / 2),   # wrist_1
    (0.0,      0.08535, -np.pi / 2),  # wrist_2
    (0.0,      0.0819,  0.0),          # wrist_3
]

# Arm base_link offset from chassis_link (from URDF chassis_to_arm_base joint)
_ARM_MOUNT_XYZ = np.array([0.0, 0.0, 0.1])

# Robot spawn height (chassis z at spawn)
_BASE_SPAWN_Z = 0.08


def _dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,      ca,      d],
        [0.0,     0.0,     0.0,    1.0],
    ])


def ur3_fk(joint_angles: np.ndarray) -> np.ndarray:
    """Return EE position (x, y, z) in the arm base_link frame using UR3 DH params."""
    T = np.eye(4)
    for i, (a, d, alpha) in enumerate(_UR3_DH):
        T = T @ _dh_transform(joint_angles[i], d, a, alpha)
    return T[:3, 3]


class PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for pick-and-place RL training.

    Observation (27): [joint_pos(6), joint_vel(6), finger_pos(1), ee_pos(3), obj_pos(3),
                       ee_to_obj(3), grasped(1), phase(1), base_pose(3)]
    Action (9):       [joint_vels(6), gripper(1), base_linear(1), base_angular(1)]

    ee_to_obj = obj_pos - ee_pos is the direct tracking vector: if the object moves,
    this immediately reflects the new direction/distance the arm needs to travel.
    """

    def __init__(self, namespace='', ros_domain_id=None, gz_partition=None):
        super().__init__()

        self.gz_partition = gz_partition  # stored for subprocess calls (e.g. _spawn_object)

        # Must be set before rclpy.init() so the node joins the right domain
        if ros_domain_id is not None:
            os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)
        if gz_partition is not None:
            os.environ['GZ_PARTITION'] = gz_partition

        if not rclpy.ok():
            rclpy.init()

        # namespace is used for multi-world parallel training (e.g. 'world_0', 'world_1')
        node_name = f'pickplace_env_{namespace}' if namespace else 'pickplace_env_node'
        self.node = Node(node_name, namespace=namespace)

        # Action space: 6 arm joints + 1 gripper + 2 base (linear, angular)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        # Observation space: 6 joint pos + 6 joint vel + 1 finger pos + 3 ee + 3 obj
        #                  + 3 ee_to_obj + 1 grasped + 1 phase + 3 base pose  = 27
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),
            dtype=np.float32
        )

        # Publishers — relative topic names are prefixed by namespace automatically
        self.cmd_vel_pub = self.node.create_publisher(Twist, 'cmd_vel', 10)

        # Joint velocity publishers — topic names match URDF JointController plugin
        self.shoulder_pub       = self.node.create_publisher(Float64, 'shoulder_pan_joint/cmd_vel', 10)
        self.shoulder_pitch_pub = self.node.create_publisher(Float64, 'shoulder_lift_joint/cmd_vel', 10)
        self.elbow_pub          = self.node.create_publisher(Float64, 'elbow_joint/cmd_vel', 10)
        self.wrist_1_pub        = self.node.create_publisher(Float64, 'wrist_1_joint/cmd_vel', 10)
        self.wrist_2_pub        = self.node.create_publisher(Float64, 'wrist_2_joint/cmd_vel', 10)
        self.wrist_3_pub        = self.node.create_publisher(Float64, 'wrist_3_joint/cmd_vel', 10)
        self.finger_pub         = self.node.create_publisher(Float64, 'finger_joint/cmd_vel', 10)

        # Subscribers
        self.joint_state_sub = self.node.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.world_pose_sub = self.node.create_subscription(
            TFMessage, '/world/pickplace_world/dynamic_pose/info',
            self.world_pose_callback, 10)

        # State variables
        # JointStatePublisher order: shoulder_pan[0], shoulder_lift[1], elbow[2],
        #   wrist_1[3], wrist_2[4], wrist_3[5], finger_joint[6], left_wheel[7], right_wheel[8]
        self.joint_positions = np.zeros(9)
        self.joint_velocities = np.zeros(9)
        self._joint_states_received = False
        self.base_pose = np.zeros(3)  # x, y, theta
        self.episode_steps = 0
        self.max_episode_steps = 800

        # Targets
        self.object_start_pos = np.array([0.6, 0.0, 0.055])  # world frame
        self.target_pos = np.array([0.6, 0.5, 0.1])
        self.object_pos = self.object_start_pos.copy()
        self.object_grasped = False
        self.real_object_pos = None  # updated from Gazebo world pose
        self.grasp_verify_steps = 0

        self.current_phase = 0
        self.prev_distance = None

    def joint_state_callback(self, msg):
        n = len(msg.position)
        if n >= 7:
            self.joint_positions = np.array(msg.position[:9] if n >= 9 else list(msg.position) + [0.0] * (9 - n))
            self.joint_velocities = np.array(msg.velocity[:9] if len(msg.velocity) >= 9 else
                                             list(msg.velocity) + [0.0] * (9 - len(msg.velocity))) \
                if len(msg.velocity) >= 7 else np.zeros(9)
            self._joint_states_received = True

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.base_pose = np.array([x, y, np.arctan2(siny_cosp, cosy_cosp)])

    def world_pose_callback(self, msg):
        for t in msg.transforms:
            # frame_id is "pickup_object::link" in Gazebo Harmonic Pose_V bridge
            if 'pickup_object' in t.child_frame_id:
                p = t.transform.translation
                self.real_object_pos = np.array([p.x, p.y, p.z])
                break

    def get_end_effector_pos(self) -> np.ndarray:
        """EE position in robot chassis frame using UR3 DH FK.
        The URDF has a 180° yaw on base_link_inertia so the arm faces +x of chassis.
        FK gives EE in arm-base frame (arm faces -x before rotation), so flip x,y.
        """
        ee_in_arm_base = ur3_fk(self.joint_positions[:6])
        # Account for 180° yaw: x→-x, y→-y (arm faces forward after flip)
        ee_flipped = np.array([-ee_in_arm_base[0], -ee_in_arm_base[1], ee_in_arm_base[2]])
        return _ARM_MOUNT_XYZ + ee_flipped

    def get_global_ee_pos(self) -> np.ndarray:
        """Transform local EE pos to world frame using odometry."""
        local_ee = self.get_end_effector_pos()
        bx, by, btheta = self.base_pose
        gx = bx + local_ee[0] * np.cos(btheta) - local_ee[1] * np.sin(btheta)
        gy = by + local_ee[0] * np.sin(btheta) + local_ee[1] * np.cos(btheta)
        gz = _BASE_SPAWN_Z + local_ee[2]
        return np.array([gx, gy, gz])

    def get_observation(self) -> np.ndarray:
        # Spin once here to get the freshest object pose before building the obs
        rclpy.spin_once(self.node, timeout_sec=0.001)
        ee_pos = self.get_global_ee_pos()
        obj_pos = self.real_object_pos if self.real_object_pos is not None else self.object_pos
        # ee_to_obj: direct tracking vector — if the object moves, this updates instantly
        ee_to_obj = obj_pos - ee_pos
        obs = np.concatenate([
            self.joint_positions[:6],    # arm joint positions
            self.joint_velocities[:6],   # arm joint velocities
            [self.joint_positions[6]],   # finger_joint position
            ee_pos,                      # EE in world frame
            obj_pos,                     # object in world frame (live from Gazebo bridge)
            ee_to_obj,                   # vector from EE to object — arm tracks this to zero
            [float(self.object_grasped)],
            [float(self.current_phase)],
            self.base_pose,
        ])
        return obs.astype(np.float32)

    def compute_reward(self):
        reward = 0.0
        terminated = False

        gripper_pos = self.joint_positions[6]  # finger_joint: 0=open, ~0.8=closed
        ee_global = self.get_global_ee_pos()

        # Base tipping penalty: if robot falls over, base height deviates significantly
        # Normal base z is ~0.08 (spawn height). If it tilts, z changes dramatically.
        base_z_deviation = abs(ee_global[2])  # check if EE goes underground
        # Use odom-based check: if base is no longer upright
        if len(self.joint_positions) > 0:
            # If the arm joints show extreme values, the robot likely tipped
            max_joint_vel = np.max(np.abs(self.joint_velocities[:6])) if len(self.joint_velocities) >= 6 else 0
            if max_joint_vel > 10.0:  # abnormally high velocity = robot tumbling
                return -500.0, True

        # Collision penalty: arm crashes into ground (very low)
        if ee_global[2] < 0.03 and self.current_phase not in [1, 2, 5]:
            return -500.0, True

        # Use real Gazebo object position when available (fixes fake-randomisation bug)
        obj_pos = self.real_object_pos if self.real_object_pos is not None else self.object_pos

        # Out-of-bounds penalty: robot wandered too far from target (avoid local optimum)
        if np.linalg.norm(obj_pos[:2] - self.base_pose[:2]) > 1.5:
            return -500.0, True

        if self.current_phase == 0:
            target_xy = obj_pos[:2]
            ee_xy = ee_global[:2]
            base_xy = self.base_pose[:2]
            base_theta = self.base_pose[2]

            base_dist_xy = np.linalg.norm(target_xy - base_xy)
            arm_dist_xy = np.linalg.norm(target_xy - ee_xy)

            angle_to_target = np.arctan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])
            angle_diff = angle_to_target - base_theta
            while angle_diff > np.pi:  angle_diff -= 2 * np.pi
            while angle_diff < -np.pi: angle_diff += 2 * np.pi

            dist_xy = base_dist_xy + arm_dist_xy + abs(angle_diff) * 0.5

            if self.prev_distance is not None:
                step_reward = (self.prev_distance - dist_xy) * 100.0
                reward += step_reward

            self.prev_distance = dist_xy

            # Penalize the standing still arm - force it to move toward the goal!
            arm_speed = np.linalg.norm(self.joint_velocities[:6])
            if arm_speed < 0.05:
                reward -= 1.0

            # Keep arm raised and close to body while driving to avoid tipping the base.
            local_ee = self.get_end_effector_pos()
            if ee_global[2] < 0.15:
                reward -= 2.0  # too low during approach
            elif ee_global[2] > 0.20:
                reward += 0.5
            # Penalize arm extending far forward while still driving (base_dist > 0.30)
            if base_dist_xy > 0.30 and local_ee[0] > 0.25:
                reward -= 3.0 * (local_ee[0] - 0.25)  # progressively stiffer

            # Phase 0 -> Phase 1 Transition:
            # Bin left-wall outer face is at x=0.405 (object_x - 0.195).
            # Robot should stop 2-5 cm from that wall → base ~0.22-0.25m from object.
            # arm_dist_xy < 0.08 ensures EE is already close toward target.
            if arm_dist_xy < 0.08 and base_dist_xy < 0.25 and abs(angle_diff) < 0.5 and ee_global[2] > 0.15:
                self.current_phase = 1
                self.prev_distance = None
                reward += 100.0

        elif self.current_phase == 1:
            # Phase 1: Lower arm to grasp height AND approach object XY
            grasp_z = obj_pos[2] if obj_pos[2] > 0.03 else 0.065
            dist_z = abs(ee_global[2] - grasp_z)
            dist_xy = np.linalg.norm(ee_global[:2] - obj_pos[:2])
            # Combined: reach grasp height + move EE toward object horizontally
            dist_combined = dist_z + dist_xy * 1.0
            if self.prev_distance is not None:
                delta = self.prev_distance - dist_combined
                reward += delta * 100.0 if delta > 0 else delta * 300.0  # 3× harsher when retreating
            self.prev_distance = dist_combined

            # Proximity bonus: reward staying near the object (not just approaching)
            if dist_xy < 0.15:
                reward += 5.0 * (1.0 - dist_xy / 0.15)

            # Z-alignment bonus: reward being at the correct grasp height
            if dist_z < 0.05:
                reward += 4.0 * (1.0 - dist_z / 0.05)

            # Extra bonus when BOTH xy and z are close simultaneously
            if dist_xy < 0.08 and dist_z < 0.05:
                reward += 8.0

            # Penalise closing gripper during approach — it serves no purpose in phase 1
            if gripper_pos > 0.5:
                reward -= 2.0

            # Transition when EE is at right height and very close in XY
            if dist_z < 0.04 and dist_xy < 0.06:
                self.current_phase = 2
                self.prev_distance = None
                reward += 100.0

        elif self.current_phase == 2:
            # Phase 2: Approach object and grasp
            ref_obj = self.real_object_pos if self.real_object_pos is not None else self.object_pos

            # Approach target: 2cm above object center so fingers wrap around
            # cylinder sides rather than crashing into the object top
            grasp_target = ref_obj.copy()
            grasp_target[2] += 0.02

            dist_to_obj = np.linalg.norm(ee_global - ref_obj)
            dist_to_target = np.linalg.norm(ee_global - grasp_target)

            # Dense reward: guide EE toward the grasp target (not the object center)
            if self.prev_distance is not None:
                delta = self.prev_distance - dist_to_target
                reward += delta * 80.0 if delta > 0 else delta * 320.0  # 4× harsher when retreating
            self.prev_distance = dist_to_target

            # Proximity bonus: reward staying near grasp target (prevents drifting away)
            if dist_to_target < 0.10:
                reward += 8.0 * (1.0 - dist_to_target / 0.10)

            # Touch-range bonus: EE is basically at the object surface — reward heavily
            if dist_to_obj < 0.04:
                reward += 15.0 * (1.0 - dist_to_obj / 0.04)

            # Very close bonus: within 3cm is pre-grasp territory
            if dist_to_obj < 0.02:
                reward += 10.0

            # Reward gripper closing when already near object (actively encourage grasping)
            if gripper_pos > 0.4 and dist_to_obj < 0.07:
                reward += 8.0 * gripper_pos  # more reward the more closed the gripper

            # Penalty for opening gripper when very close (don't retreat from grasp)
            if gripper_pos < 0.2 and dist_to_obj < 0.05:
                reward -= 5.0

            # Grasp fires when EE is within 5cm of object AND gripper closed
            if gripper_pos > 0.7 and dist_to_obj < 0.04:
                self.object_grasped = True
                self.current_phase = 3
                self.grasp_verify_steps = 0
                self.prev_distance = None
                reward += 1000.0
            elif gripper_pos > 0.7 and dist_to_obj >= 0.08:
                # Penalty scales with distance: closing far away = much worse than closing nearby
                reward -= 0.5 + dist_to_obj * 5.0

            # Wrist orientation reward: nudge gripper to horizontal (wrist_2 ≈ 0)
            # so fingers are parallel to ground for side-grasp of cylinder
            wrist_orient_err = abs(self.joint_positions[4])  # wrist_2_joint
            reward -= wrist_orient_err * 0.3

        elif self.current_phase == 3:
            # Verify grasp: real object should rise with EE; if it stays on the floor, abort.
            if self.real_object_pos is not None:
                self.grasp_verify_steps += 1
                if self.grasp_verify_steps > 10 and self.real_object_pos[2] < 0.06:
                    # Object didn't lift — grasp failed, go back to phase 2
                    self.object_grasped = False
                    self.current_phase = 2
                    reward -= 10.0
            dist_z = abs(ee_global[2] - 0.25)
            if self.prev_distance is not None:
                delta = self.prev_distance - dist_z
                reward += delta * 100.0 if delta > 0 else delta * 200.0
            self.prev_distance = dist_z
            # Bonus for being near lift height
            if dist_z < 0.08:
                reward += 5.0 * (1.0 - dist_z / 0.08)
            if dist_z < 0.05:
                self.current_phase = 4
                self.prev_distance = None
                reward += 200.0

        elif self.current_phase == 4:
            target_xy = self.target_pos[:2]
            ee_xy = ee_global[:2]
            base_xy = self.base_pose[:2]
            # Include angle-to-target penalty so base faces goal before placing
            angle_to_target = np.arctan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])
            angle_diff = angle_to_target - self.base_pose[2]
            while angle_diff > np.pi:  angle_diff -= 2 * np.pi
            while angle_diff < -np.pi: angle_diff += 2 * np.pi
            dist_xy = np.linalg.norm(target_xy - ee_xy) + np.linalg.norm(target_xy - base_xy) + abs(angle_diff) * 0.3
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist_xy) * 50.0
            self.prev_distance = dist_xy
            if np.linalg.norm(target_xy - ee_xy) < 0.15 and abs(angle_diff) < 0.6:
                self.current_phase = 5
                self.prev_distance = None
                reward += 100.0

        elif self.current_phase == 5:
            dist = np.linalg.norm(ee_global - self.target_pos)
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist) * 50.0
            self.prev_distance = dist
            if dist < 0.08 and gripper_pos < 0.1:
                self.object_grasped = False
                reward += 1000.0
                terminated = True

        reward -= 0.01 * np.sum(np.abs(self.joint_velocities[:6]))

        # During grasp phases (1-3): penalise base rotating away from object
        # so the arm stays aligned with the bin after scripted pre-grasp
        if self.current_phase in [1, 2, 3]:
            ref_pos = self.real_object_pos if self.real_object_pos is not None else self.object_pos
            bx, by, btheta = self.base_pose
            desired_angle = np.arctan2(ref_pos[1] - by, ref_pos[0] - bx)
            angle_err = desired_angle - btheta
            while angle_err > np.pi:  angle_err -= 2 * np.pi
            while angle_err < -np.pi: angle_err += 2 * np.pi
            if abs(angle_err) > 0.5:
                reward -= abs(angle_err) * 0.2

        return reward, terminated

    def step(self, action):
        rclpy.spin_once(self.node, timeout_sec=0.005)

        # Position delta control: target = current + delta, then P-drive toward target.
        # Max delta per step = 0.25 rad → faster reach toward object.
        delta = action[:6] * 0.25
        target_joints = self.joint_positions[:6] + delta
        # P-controller: velocity = (target - current) * gain, clamped to ±0.5 rad/s
        joint_vels = np.clip((target_joints - self.joint_positions[:6]) * 10.0, -0.5, 0.5)

        gripper_command = action[6]

        # Zero base during manipulation phases (1-3) to preserve scripted approach pose
        if self.current_phase in [1, 2, 3]:
            base_linear_vel = 0.0
            base_angular_vel = 0.0
        else:
            base_linear_vel = float(action[7]) * 0.5
            base_angular_vel = float(action[8]) * 1.0

        self.shoulder_pub.publish(Float64(data=float(joint_vels[0])))
        self.shoulder_pitch_pub.publish(Float64(data=float(joint_vels[1])))
        self.elbow_pub.publish(Float64(data=float(joint_vels[2])))
        self.wrist_1_pub.publish(Float64(data=float(joint_vels[3])))
        self.wrist_2_pub.publish(Float64(data=float(joint_vels[4])))
        self.wrist_3_pub.publish(Float64(data=float(joint_vels[5])))

        gripper_vel = 0.5 if gripper_command > 0 else -0.5
        self.finger_pub.publish(Float64(data=gripper_vel))

        twist_msg = Twist()
        twist_msg.linear.x = base_linear_vel
        twist_msg.angular.z = base_angular_vel
        self.cmd_vel_pub.publish(twist_msg)

        if self.object_grasped:
            ee_global = self.get_global_ee_pos()
            self.object_pos = ee_global.copy()
            self.object_pos[2] -= 0.05

        time.sleep(0.002)

        obs = self.get_observation()
        reward, terminated = self.compute_reward()

        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def _scripted_pregrasp(self):
        """
        Face the object, drive forward only until the front caster (at chassis_x + 0.36m)
        would reach the bin back-wall (outer face ≈ 0.40m), then extend the arm.
        Runs for up to 300 steps before handing off to RL.
        """
        # Bin back-wall outer face ≈ 0.405m; caster front = chassis_x + 0.36m.
        # Keep chassis_x ≤ 0.04m so the caster stays clear of the wall.
        SAFE_CHASSIS_X = 0.04

        obj_pos = self.real_object_pos if self.real_object_pos is not None else self.object_pos
        for _ in range(300):
            rclpy.spin_once(self.node, timeout_sec=0.005)
            obj_pos = self.real_object_pos if self.real_object_pos is not None else self.object_pos

            bx, by, btheta = self.base_pose
            dx = obj_pos[0] - bx
            dy = obj_pos[1] - by

            # P-controller: turn to face object
            angle_to = np.arctan2(dy, dx)
            angle_err = angle_to - btheta
            while angle_err > np.pi:  angle_err -= 2 * np.pi
            while angle_err < -np.pi: angle_err += 2 * np.pi

            tw = Twist()
            tw.angular.z = float(np.clip(angle_err * 2.0, -1.0, 1.0))

            # Only drive forward if chassis x stays within the safe limit
            if bx < SAFE_CHASSIS_X and abs(angle_err) < 0.4:
                tw.linear.x = float(np.clip((SAFE_CHASSIS_X - bx) * 3.0, 0.0, 0.2))
            else:
                tw.linear.x = 0.0
            self.cmd_vel_pub.publish(tw)

            # Break once arm EE is within 0.30m XY of the object and robot is facing it
            ee_global = self.get_global_ee_pos()
            ee_dist_xy = np.linalg.norm(ee_global[:2] - obj_pos[:2])
            if ee_dist_xy < 0.30 and abs(angle_err) < 0.4:
                self.cmd_vel_pub.publish(Twist())
                break

            # Pre-grasp arm pose: extend arm forward toward the object.
            # shoulder_pan=0 faces arm +x (chassis forward after 180° yaw flip).
            # shoulder_lift=-1.7, elbow=2.0 extend the arm forward/down.
            # wrist_1=-1.0 keeps gripper roughly horizontal for side-grasp.
            pan_err = 0.0   - self.joint_positions[0]
            s_err   = -1.7  - self.joint_positions[1]
            e_err   =  2.0  - self.joint_positions[2]
            w1_err  = -1.0  - self.joint_positions[3]

            pan_vel = np.clip(pan_err * 2.0, -0.4, 0.4) if abs(pan_err) > 0.05 else 0.0
            s_vel   = np.clip(s_err   * 2.0, -0.4, 0.4) if abs(s_err)   > 0.05 else 0.0
            e_vel   = np.clip(e_err   * 2.0, -0.4, 0.4) if abs(e_err)   > 0.05 else 0.0
            w1_vel  = np.clip(w1_err  * 2.0, -0.4, 0.4) if abs(w1_err)  > 0.05 else 0.0

            self.shoulder_pub.publish(Float64(data=float(pan_vel)))
            self.shoulder_pitch_pub.publish(Float64(data=float(s_vel)))
            self.elbow_pub.publish(Float64(data=float(e_vel)))
            self.wrist_1_pub.publish(Float64(data=float(w1_vel)))
            time.sleep(0.01)

        self.cmd_vel_pub.publish(Twist())

    def _spawn_object(self, x, y, z):
        """Move the Gazebo pickup_object to (x, y, z) via gz service."""
        req = f'name: "pickup_object" position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}} orientation: {{w: 1}}'
        env = os.environ.copy()
        if self.gz_partition:
            env['GZ_PARTITION'] = self.gz_partition
        subprocess.run(
            ['gz', 'service', '-s', '/world/pickplace_world/set_pose',
             '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '2000', '--req', req],
            env=env, capture_output=True
        )

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.episode_steps = 0
        self.object_grasped = False
        self.current_phase = 1  # start directly at lowering phase (base already positioned)
        self.prev_distance = None
        self.grasp_verify_steps = 0

        # Randomize object XY ±3cm so the policy generalises, not memorises one spot
        rng = np.random.default_rng(seed)
        ox = 0.6 + rng.uniform(-0.03, 0.03)
        oy = 0.0 + rng.uniform(-0.03, 0.03)
        oz = 0.065
        self.object_start_pos = np.array([ox, oy, oz])
        self.object_pos = self.object_start_pos.copy()
        self._spawn_object(ox, oy, oz)

        self.cmd_vel_pub.publish(Twist())

        if not self._joint_states_received:
            self.node.get_logger().info('Waiting for /joint_states...')
            while not self._joint_states_received:
                rclpy.spin_once(self.node, timeout_sec=0.1)

        # Scripted pre-grasp: drive base to ~25cm from object, arm tucked
        self._scripted_pregrasp()

        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.005)
            time.sleep(0.01)

        return self.get_observation(), {}

    def close(self):
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
