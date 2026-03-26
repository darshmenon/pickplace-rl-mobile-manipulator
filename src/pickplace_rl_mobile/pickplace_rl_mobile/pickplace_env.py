#!/usr/bin/env python3

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float64
import time

class PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for pick-and-place RL training.
    
    Observation: [joint_positions(5), end_effector_pos(3), object_pos(3), object_grasped(1), current_phase(1), base_x(1), base_y(1), base_theta(1)]
    Action: [joint_velocities(5), gripper_control(1), base_linear_vel(1), base_angular_vel(1)]
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()
        
        self.node = Node('pickplace_env_node')
        
        # Action space: 5 arm joints + 1 gripper + 2 base (linear, angular)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(8,), 
            dtype=np.float32
        )
        
        # Observation space: 5 joint pos + 3 ee + 3 obj + 1 grasped + 1 phase + 3 base pose (x, y, theta)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(16,), 
            dtype=np.float32
        )
        
        # Publishers
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        
        # Joint velocity publishers (must match URDF JointController plugin topic names)
        self.shoulder_pub = self.node.create_publisher(Float64, '/shoulder_pan_joint/cmd_vel', 10)
        self.shoulder_pitch_pub = self.node.create_publisher(Float64, '/shoulder_lift_joint/cmd_vel', 10)
        self.elbow_pub = self.node.create_publisher(Float64, '/elbow_joint/cmd_vel', 10)
        self.wrist_roll_pub = self.node.create_publisher(Float64, '/wrist_1_joint/cmd_vel', 10)
        self.wrist_pitch_pub = self.node.create_publisher(Float64, '/wrist_2_joint/cmd_vel', 10)
        self.finger_pub = self.node.create_publisher(Float64, '/finger_joint/cmd_vel', 10)
        
        # Subscribers
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.odom_sub = self.node.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # State variables
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7)
        self.base_pose = np.zeros(3) # x, y, theta
        self.episode_steps = 0
        self.max_episode_steps = 800
        
        # Targets
        self.object_start_pos = np.array([0.6, 0.0, 0.055]) # Local space
        self.target_pos = np.array([0.6, 0.5, 0.1])
        self.object_pos = self.object_start_pos.copy()
        self.object_grasped = False
        
        self.current_phase = 0
        self.prev_distance = None
        
    def joint_state_callback(self, msg):
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[:7])
            self.joint_velocities = np.array(msg.velocity[:7]) if len(msg.velocity) >= 7 else np.zeros(7)
            
    def odom_callback(self, msg):
        # Extract x, y from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion calculation
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)
        
        self.base_pose = np.array([x, y, theta])
    
    def get_end_effector_pos(self):
        """Relative EE pos to base."""
        base_height = 0.2
        shoulder_angle = self.joint_positions[0]
        shoulder_pitch = self.joint_positions[1]
        elbow = self.joint_positions[2]
        
        link1_length = 0.24 
        link2_length = 0.24
        link3_length = 0.20
        
        z = base_height + link1_length
        x_local = link2_length * np.cos(shoulder_pitch) + link3_length * np.cos(shoulder_pitch + elbow)
        z_local = link2_length * np.sin(shoulder_pitch) + link3_length * np.sin(shoulder_pitch + elbow)
        
        x = 0.1 + x_local * np.cos(shoulder_angle)
        y = x_local * np.sin(shoulder_angle)
        z = z + z_local
        
        return np.array([x, y, z])
    
    def get_global_ee_pos(self):
        """Transform local EE pos to global world frame using odometry."""
        local_ee = self.get_end_effector_pos()
        bx, by, btheta = self.base_pose
        
        # 2D Rotation + Translation
        gx = bx + local_ee[0] * np.cos(btheta) - local_ee[1] * np.sin(btheta)
        gy = by + local_ee[0] * np.sin(btheta) + local_ee[1] * np.cos(btheta)
        gz = local_ee[2] # Assuming diff drive base is flat on ground (z=0)
        
        return np.array([gx, gy, gz])
    
    def get_observation(self):
        ee_pos = self.get_end_effector_pos()
        
        obs = np.concatenate([
            self.joint_positions[:5],
            ee_pos,
            self.object_pos,
            [float(self.object_grasped)],
            [float(self.current_phase)],
            self.base_pose
        ])
        
        return obs.astype(np.float32)
    
    def compute_reward(self):
        reward = 0.0
        terminated = False
        
        gripper_pos = self.joint_positions[6]  # finger_joint: 0=open, 0.8=closed
        ee_global = self.get_global_ee_pos()
        
        # Collision Check: Reaching lower than bin while NOT lowering/grasping/releasing
        if ee_global[2] < 0.10 and self.current_phase not in [1, 2, 5]:
            return -100.0, True

        if self.current_phase == 0:  
            target_xy = self.object_pos[:2]
            ee_xy = ee_global[:2]
            base_xy = self.base_pose[:2]
            base_theta = self.base_pose[2]
            
            # Distance from base to absolute object position
            base_dist_xy = np.linalg.norm(target_xy - base_xy)
            
            # The arm needs to extend its global coordinate to the object 
            arm_dist_xy = np.linalg.norm(target_xy - ee_xy)
            
            # --- ORIENTATION CALCULATION ---
            # Calculate the angle from the base to the object
            angle_to_target = np.arctan2(target_xy[1] - base_xy[1], target_xy[0] - base_xy[0])
            
            # Calculate angular difference (shortest path)
            angle_diff = angle_to_target - base_theta
            
            # Normalize angle difference to [-pi, pi]
            while angle_diff > np.pi: angle_diff -= 2 * np.pi
            while angle_diff < -np.pi: angle_diff += 2 * np.pi
            
            abs_angle_diff = abs(angle_diff)
            
            # Total distance metric combines base dist, arm dist, and angular diff
            # We scale angular diff so it's comparable to distance (e.g. pi rad diff ~ 1.0m diff)
            dist_xy = base_dist_xy + arm_dist_xy + (abs_angle_diff * 0.5)
            
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist_xy) * 10.0
                
            self.prev_distance = dist_xy
            
            # Transition condition: arm is near target XY, base is reasonably close, height is safe, and pointing towards object
            if arm_dist_xy < 0.10 and base_dist_xy < 0.8 and abs_angle_diff < 0.5 and ee_global[2] > 0.15:
                self.current_phase = 1
                self.prev_distance = None
                reward += 20.0
                
        elif self.current_phase == 1: 
            dist_z = abs(ee_global[2] - 0.07) 
            
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist_z) * 10.0
                
            self.prev_distance = dist_z
            
            if dist_z < 0.02:
                self.current_phase = 2
                self.prev_distance = None
                reward += 20.0
                
        elif self.current_phase == 2:  
            if gripper_pos > 0.7:  # finger_joint closed (near upper limit 0.8)
                self.object_grasped = True
                self.current_phase = 3
                reward += 50.0
            else:
                reward -= 0.1
                
        elif self.current_phase == 3:  
            dist_z = abs(ee_global[2] - 0.25)
            
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist_z) * 10.0
                
            self.prev_distance = dist_z
            
            if dist_z < 0.05:
                self.current_phase = 4
                self.prev_distance = None
                reward += 20.0
                
        elif self.current_phase == 4:  
            target_xy = self.target_pos[:2]
            ee_xy = ee_global[:2]
            
            base_dist_xy = np.linalg.norm(target_xy - self.base_pose[:2])
            arm_dist_xy = np.linalg.norm(target_xy - ee_xy)
            
            dist_xy = arm_dist_xy + base_dist_xy
            
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist_xy) * 10.0
                
            self.prev_distance = dist_xy
            
            if arm_dist_xy < 0.15:
                self.current_phase = 5
                self.prev_distance = None
                reward += 50.0

        elif self.current_phase == 5: 
            dist = np.linalg.norm(ee_global - self.target_pos)
            
            if self.prev_distance is not None:
                reward += (self.prev_distance - dist) * 10.0
                
            self.prev_distance = dist
            
            if dist < 0.08 and gripper_pos < 0.1:  # finger_joint open (near lower limit 0)
                self.object_grasped = False
                reward += 100.0 
                terminated = True
                
        reward -= 0.01 * np.sum(np.abs(self.joint_velocities[:5]))
        
        return reward, terminated

    def step(self, action):
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        joint_velocities = action[:5] * 0.5
        gripper_command = action[5]
        base_linear_vel = action[6] * 0.5   # Scale max speed
        base_angular_vel = action[7] * 1.0  # Scale turning speed
        
        self.shoulder_pub.publish(Float64(data=float(joint_velocities[0])))
        self.shoulder_pitch_pub.publish(Float64(data=float(joint_velocities[1])))
        self.elbow_pub.publish(Float64(data=float(joint_velocities[2])))
        self.wrist_roll_pub.publish(Float64(data=float(joint_velocities[3])))
        self.wrist_pitch_pub.publish(Float64(data=float(joint_velocities[4])))
        
        gripper_vel = 0.5 if gripper_command > 0 else -0.5
        self.finger_pub.publish(Float64(data=gripper_vel))
        
        # Base Command
        twist_msg = Twist()
        twist_msg.linear.x = float(base_linear_vel)
        twist_msg.angular.z = float(base_angular_vel)
        self.cmd_vel_pub.publish(twist_msg)
        
        if self.object_grasped:
            # We move the object globally based on the global EE!
            ee_global = self.get_global_ee_pos()
            self.object_pos = ee_global.copy()
            self.object_pos[2] -= 0.05
        
        time.sleep(0.01)
        
        obs = self.get_observation()
        
        reward, terminated = self.compute_reward()
        
        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.episode_steps = 0
        self.object_grasped = False
        self.current_phase = 0
        self.prev_distance = None
        
        random_x = 0.5 + np.random.rand() * 0.2
        random_y = -0.2 + np.random.rand() * 0.4 
        self.object_start_pos = np.array([random_x, random_y, 0.055])
        self.object_pos = self.object_start_pos.copy()
        
        # Stop Base
        zero_twist = Twist()
        self.cmd_vel_pub.publish(zero_twist)
        
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.01)
        
        obs = self.get_observation()
        
        return obs, {}

    def close(self):
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
