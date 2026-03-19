#!/usr/bin/env python3
"""
Manipulation RL Node for the Pick-and-Place Mobile Manipulator.

Wraps the trained SAC policy as a proper ROS2 node that:
- Subscribes to perception (detected object pose) instead of using hard-coded positions
- Subscribes to joint states and odometry
- Runs RL inference at 20Hz
- Publishes joint velocity commands and base motion
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
import json
import os


class ManipRLNode(Node):
    def __init__(self):
        super().__init__('manip_rl_node')

        # --- Parameters ---
        self.declare_parameter('model_path', './rl_models/pickplace_final_model.zip')
        self.declare_parameter('inference_rate', 20.0)
        self.declare_parameter('use_perception', True)
        self.declare_parameter('fallback_object_pos', [0.6, 0.0, 0.055])
        self.declare_parameter('target_pos', [0.6, 0.5, 0.1])
        self.declare_parameter('joint_velocity_scale', 0.5)
        self.declare_parameter('base_linear_scale', 0.5)
        self.declare_parameter('base_angular_scale', 1.0)

        self.model_path = self.get_parameter('model_path').value
        self.use_perception = self.get_parameter('use_perception').value
        self.fallback_object_pos = np.array(
            self.get_parameter('fallback_object_pos').value)
        self.target_pos = np.array(
            self.get_parameter('target_pos').value)
        self.joint_vel_scale = self.get_parameter('joint_velocity_scale').value
        self.base_lin_scale = self.get_parameter('base_linear_scale').value
        self.base_ang_scale = self.get_parameter('base_angular_scale').value

        # State variables
        self.joint_positions = np.zeros(7)
        self.base_pose = np.zeros(3)  # x, y, theta
        self.object_pos = self.fallback_object_pos.copy()
        self.object_grasped = False
        self.current_phase = 0
        self.model = None
        self.model_loaded = False
        self.perception_received = False

        # Robot kinematics (same as pickplace_env.py)
        self.base_height = 0.2
        self.link1_length = 0.24
        self.link2_length = 0.24
        self.link3_length = 0.20

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.shoulder_pub = self.create_publisher(
            Float64, '/shoulder_joint/cmd_vel', 10)
        self.shoulder_pitch_pub = self.create_publisher(
            Float64, '/shoulder_pitch_joint/cmd_vel', 10)
        self.elbow_pub = self.create_publisher(
            Float64, '/elbow_joint/cmd_vel', 10)
        self.wrist_roll_pub = self.create_publisher(
            Float64, '/wrist_roll_joint/cmd_vel', 10)
        self.wrist_pitch_pub = self.create_publisher(
            Float64, '/wrist_pitch_joint/cmd_vel', 10)
        self.left_finger_pub = self.create_publisher(
            Float64, '/left_finger_joint/cmd_vel', 10)
        self.right_finger_pub = self.create_publisher(
            Float64, '/right_finger_joint/cmd_vel', 10)

        # --- Subscribers ---
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        if self.use_perception:
            self.perception_sub = self.create_subscription(
                PoseStamped, '/perception/detected_object',
                self.perception_callback, 10)

        self.safety_sub = self.create_subscription(
            String, '/safety/status', self.safety_callback, 10)

        # Safety state
        self.safety_ok = True

        # Load the RL model
        self.load_model()

        # Inference timer
        rate = self.get_parameter('inference_rate').value
        self.timer = self.create_timer(1.0 / rate, self.inference_step)

        self.get_logger().info(
            f'ManipRL node initialized (perception={self.use_perception})')

    def load_model(self):
        """Load the trained SAC model."""
        try:
            from stable_baselines3 import SAC
            if os.path.exists(self.model_path):
                self.model = SAC.load(self.model_path)
                self.model_loaded = True
                self.get_logger().info(f'Loaded RL model from {self.model_path}')
            else:
                self.get_logger().warn(
                    f'Model not found at {self.model_path} — running in demo mode')
        except ImportError:
            self.get_logger().error(
                'stable-baselines3 not installed — cannot load model')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

    def joint_callback(self, msg):
        """Update joint positions."""
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[:7])

    def odom_callback(self, msg):
        """Update base pose from odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)
        self.base_pose = np.array([x, y, theta])

    def perception_callback(self, msg):
        """Update object position from perception node."""
        self.object_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.perception_received = True

    def safety_callback(self, msg):
        """Check safety status."""
        try:
            status = json.loads(msg.data)
            self.safety_ok = not status.get('e_stop', False)
        except (json.JSONDecodeError, KeyError):
            pass

    def get_end_effector_pos(self):
        """Compute local EE position from joint angles."""
        shoulder = self.joint_positions[0]
        shoulder_pitch = self.joint_positions[1]
        elbow = self.joint_positions[2]

        z = self.base_height + self.link1_length
        x_local = (self.link2_length * np.cos(shoulder_pitch) +
                   self.link3_length * np.cos(shoulder_pitch + elbow))
        z_local = (self.link2_length * np.sin(shoulder_pitch) +
                   self.link3_length * np.sin(shoulder_pitch + elbow))

        x = 0.1 + x_local * np.cos(shoulder)
        y = x_local * np.sin(shoulder)
        z = z + z_local

        return np.array([x, y, z])

    def build_observation(self):
        """Build observation vector matching PickPlaceEnv format."""
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

    def publish_action(self, action):
        """Publish joint velocities and base twist from action vector."""
        joint_vels = action[:5] * self.joint_vel_scale
        gripper_cmd = action[5]
        base_linear = action[6] * self.base_lin_scale
        base_angular = action[7] * self.base_ang_scale

        # Joint velocities
        self.shoulder_pub.publish(Float64(data=float(joint_vels[0])))
        self.shoulder_pitch_pub.publish(Float64(data=float(joint_vels[1])))
        self.elbow_pub.publish(Float64(data=float(joint_vels[2])))
        self.wrist_roll_pub.publish(Float64(data=float(joint_vels[3])))
        self.wrist_pitch_pub.publish(Float64(data=float(joint_vels[4])))

        # Gripper
        gripper_vel = 0.5 if gripper_cmd > 0 else -0.5
        self.left_finger_pub.publish(Float64(data=gripper_vel))
        self.right_finger_pub.publish(Float64(data=gripper_vel))

        # Base motion
        twist = Twist()
        twist.linear.x = float(base_linear)
        twist.angular.z = float(base_angular)
        self.cmd_vel_pub.publish(twist)

    def inference_step(self):
        """Run one step of RL inference."""
        if not self.safety_ok:
            # Safety violation — stop everything
            self.publish_action(np.zeros(8))
            return

        if not self.model_loaded:
            # Demo mode: command zero velocities so the arm returns to rest
            self.publish_action(np.zeros(8))
            return

        if self.use_perception and not self.perception_received:
            self.get_logger().debug(
                'Waiting for perception data...', throttle_duration_sec=5.0)
            return

        # Build observation and get action from policy
        obs = self.build_observation()
        action, _ = self.model.predict(obs, deterministic=True)

        # Publish commands
        self.publish_action(action)


def main(args=None):
    rclpy.init(args=args)
    node = ManipRLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
