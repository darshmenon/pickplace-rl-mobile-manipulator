#!/usr/bin/env python3

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
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

    Observation (24): [joint_pos(6), joint_vel(6), finger_pos(1), ee_pos(3), obj_pos(3), grasped(1), phase(1), base_pose(3)]
    Action (9):       [joint_vels(6), gripper(1), base_linear(1), base_angular(1)]
    """

    def __init__(self):
        super().__init__()

        if not rclpy.ok():
            rclpy.init()

        self.node = Node('pickplace_env_node')

        # Action space: 6 arm joints + 1 gripper + 2 base (linear, angular)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        # Observation space: 6 joint pos + 6 joint vel + 1 finger pos + 3 ee + 3 obj + 1 grasped + 1 phase + 3 base pose
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32
        )

        # Publishers
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Joint velocity publishers — topic names match URDF JointController plugin
        self.shoulder_pub       = self.node.create_publisher(Float64, '/shoulder_pan_joint/cmd_vel', 10)
        self.shoulder_pitch_pub = self.node.create_publisher(Float64, '/shoulder_lift_joint/cmd_vel', 10)
        self.elbow_pub          = self.node.create_publisher(Float64, '/elbow_joint/cmd_vel', 10)
        self.wrist_1_pub        = self.node.create_publisher(Float64, '/wrist_1_joint/cmd_vel', 10)
        self.wrist_2_pub        = self.node.create_publisher(Float64, '/wrist_2_joint/cmd_vel', 10)
        self.wrist_3_pub        = self.node.create_publisher(Float64, '/wrist_3_joint/cmd_vel', 10)
        self.finger_pub         = self.node.create_publisher(Float64, '/finger_joint/cmd_vel', 10)

        # Subscribers
        self.joint_state_sub = self.node.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

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

    def get_end_effector_pos(self) -> np.ndarray:
        """EE position in robot chassis frame using proper UR3 DH FK."""
        ee_in_arm_base = ur3_fk(self.joint_positions[:6])
        return _ARM_MOUNT_XYZ + ee_in_arm_base

    def get_global_ee_pos(self) -> np.ndarray:
        """Transform local EE pos to world frame using odometry."""
        local_ee = self.get_end_effector_pos()
        bx, by, btheta = self.base_pose
        gx = bx + local_ee[0] * np.cos(btheta) - local_ee[1] * np.sin(btheta)
        gy = by + local_ee[0] * np.sin(btheta) + local_ee[1] * np.cos(btheta)
        gz = _BASE_SPAWN_Z + local_ee[2]
        return np.array([gx, gy, gz])

    def get_observation(self) -> np.ndarray:
        ee_pos = self.get_end_effector_pos()
        obs = np.concatenate([
            self.joint_positions[:6],    # arm joint positions
            self.joint_velocities[:6],   # arm joint velocities
            [self.joint_positions[6]],   # finger_joint position
            ee_pos,
            self.object_pos,
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

        # Out-of-bounds penalty: robot wandered too far from target (avoid local optimum)
        if np.linalg.norm(self.object_pos[:2] - self.base_pose[:2]) > 1.5:
            return -500.0, True

        if self.current_phase == 0:
            target_xy = self.object_pos[:2]
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
                
                # Heavily penalize moving AWAY from the target!
                if dist_xy > self.prev_distance:
                    reward -= 10.0
                    
            self.prev_distance = dist_xy

            # Penalize the standing still arm - force it to move toward the goal!
            arm_speed = np.linalg.norm(self.joint_velocities[:6])
            if arm_speed < 0.05:
                reward -= 1.0

            # Encourage keeping the arm raised during approach (stay above 0.15m)
            if ee_global[2] < 0.15:
                reward -= 2.0  # gentle penalty for being too low during approach
            elif ee_global[2] > 0.20:
                reward += 0.5  # small bonus for good height

            # Phase 0 -> Phase 1 Transition: Base arrived, Arm hovering over target
            if arm_dist_xy < 0.10 and base_dist_xy < 0.8 and abs(angle_diff) < 0.5 and ee_global[2] > 0.15:
                self.current_phase = 1
                self.prev_distance = None
                reward += 100.0

        elif self.current_phase == 1:
            # Phase 1: Try to PICK! Moving straight down to the cube.
            dist_z = abs(ee_global[2] - 0.07)
            if self.prev_distance is not None:
                step_reward = (self.prev_distance - dist_z) * 50.0
                reward += step_reward
                
                # Punish moving away (back upwards)!
                if dist_z > self.prev_distance:
                    reward -= 5.0
            self.prev_distance = dist_z

            if dist_z < 0.02:
                self.current_phase = 2
                self.prev_distance = None
                reward += 100.0

        elif self.current_phase == 2:
            # Phase 2: Grasping the object
            if gripper_pos > 0.7:
                self.object_grasped = True
                self.current_phase = 3
                reward += 500.0
            else:
                reward -= 0.1

        elif self.current_phase == 3:
            dist_z = abs(ee_global[2] - 0.25)
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist_z) * 50.0
            self.prev_distance = dist_z
            if dist_z < 0.05:
                self.current_phase = 4
                self.prev_distance = None
                reward += 200.0

        elif self.current_phase == 4:
            target_xy = self.target_pos[:2]
            ee_xy = ee_global[:2]
            dist_xy = np.linalg.norm(target_xy - ee_xy) + np.linalg.norm(target_xy - self.base_pose[:2])
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist_xy) * 50.0
            self.prev_distance = dist_xy
            if np.linalg.norm(target_xy - ee_xy) < 0.15:
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

        return reward, terminated

    def step(self, action):
        rclpy.spin_once(self.node, timeout_sec=0.01)

        joint_vels = action[:6] * 0.5
        gripper_command = action[6]
        base_linear_vel = action[7] * 0.5
        base_angular_vel = action[8] * 1.0

        self.shoulder_pub.publish(Float64(data=float(joint_vels[0])))
        self.shoulder_pitch_pub.publish(Float64(data=float(joint_vels[1])))
        self.elbow_pub.publish(Float64(data=float(joint_vels[2])))
        self.wrist_1_pub.publish(Float64(data=float(joint_vels[3])))
        self.wrist_2_pub.publish(Float64(data=float(joint_vels[4])))
        self.wrist_3_pub.publish(Float64(data=float(joint_vels[5])))

        gripper_vel = 0.5 if gripper_command > 0 else -0.5
        self.finger_pub.publish(Float64(data=gripper_vel))

        twist_msg = Twist()
        twist_msg.linear.x = float(base_linear_vel)
        twist_msg.angular.z = float(base_angular_vel)
        self.cmd_vel_pub.publish(twist_msg)

        if self.object_grasped:
            ee_global = self.get_global_ee_pos()
            self.object_pos = ee_global.copy()
            self.object_pos[2] -= 0.05

        time.sleep(0.01)

        obs = self.get_observation()
        reward, terminated = self.compute_reward()

        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.episode_steps = 0
        self.object_grasped = False
        self.current_phase = 0
        self.prev_distance = None

        random_x = 0.5 + np.random.rand() * 0.2
        random_y = -0.2 + np.random.rand() * 0.4
        self.object_start_pos = np.array([random_x, random_y, 0.055])
        self.object_pos = self.object_start_pos.copy()

        self.cmd_vel_pub.publish(Twist())

        if not self._joint_states_received:
            self.node.get_logger().info('Waiting for /joint_states...')
            while not self._joint_states_received:
                rclpy.spin_once(self.node, timeout_sec=0.1)

        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.01)

        return self.get_observation(), {}

    def close(self):
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
