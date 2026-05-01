#!/usr/bin/env python3
"""
Manipulation RL Node for the Pick-and-Place Mobile Manipulator.

Wraps the trained RL policy as a proper ROS2 node that:
- Subscribes to perception (detected object pose) instead of using hard-coded positions
- Subscribes to joint states and odometry
- Runs RL inference at 20Hz
- Publishes joint velocity commands and base motion
"""

import numpy as np
import gymnasium as gym
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
import json
import os
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


_UR3_DH = [
    (0.0,      0.1519,  np.pi / 2),
    (-0.24365, 0.0,     0.0),
    (-0.21325, 0.0,     0.0),
    (0.0,      0.11235, np.pi / 2),
    (0.0,      0.08535, -np.pi / 2),
    (0.0,      0.0819,  0.0),
]
_ARM_MOUNT_XYZ = np.array([0.0, 0.0, 0.1], dtype=np.float32)
_BASE_SPAWN_Z = 0.08
_ARM_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
_ARM_JOINT_LIMITS = [
    (-2 * np.pi, 2 * np.pi),
    (-2 * np.pi, 2 * np.pi),
    (-np.pi, np.pi),
    (-2 * np.pi, 2 * np.pi),
    (-2 * np.pi, 2 * np.pi),
    (-np.inf, np.inf),
]
_MIMIC_MULTIPLIERS = {
    'left_inner_knuckle_joint': 1.0,
    'left_inner_finger_joint': -1.0,
    'right_outer_knuckle_joint': 1.0,
    'right_inner_knuckle_joint': 1.0,
    'right_inner_finger_joint': -1.0,
}


def _dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,      ca,      d],
        [0.0,     0.0,     0.0,    1.0],
    ], dtype=np.float32)


def _ur3_fk(joint_angles: np.ndarray) -> np.ndarray:
    """Return EE position in the arm base frame using UR3 DH parameters."""
    transform = np.eye(4, dtype=np.float32)
    for theta, (a, d, alpha) in zip(joint_angles[:6], _UR3_DH):
        transform = transform @ _dh_transform(float(theta), float(d), float(a), float(alpha))
    return transform[:3, 3]


class _InferenceSpaceEnv(gym.Env):
    """Minimal env used only to load VecNormalize stats for inference."""

    metadata = {}

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


class ManipRLNode(Node):
    def __init__(self):
        super().__init__('manip_rl_node')

        # --- Parameters ---
        self.declare_parameter('model_path', './rl_models/best_model/best_model.zip')
        self.declare_parameter('inference_rate', 20.0)
        self.declare_parameter('use_perception', True)
        self.declare_parameter('fallback_object_pos', [0.6, 0.0, 0.1325])
        self.declare_parameter('target_pos', [0.6, 0.5, 0.15])
        self.declare_parameter('joint_velocity_scale', 0.5)
        self.declare_parameter('base_linear_scale', 0.5)
        self.declare_parameter('base_angular_scale', 1.0)
        self.declare_parameter('initial_phase', 1)

        self.model_path = self.get_parameter('model_path').value
        self.use_perception = self.get_parameter('use_perception').value
        self.fallback_object_pos = np.array(
            self.get_parameter('fallback_object_pos').value, dtype=np.float32)
        self.target_pos = np.array(
            self.get_parameter('target_pos').value, dtype=np.float32)
        self.joint_vel_scale = self.get_parameter('joint_velocity_scale').value
        self.base_lin_scale = self.get_parameter('base_linear_scale').value
        self.base_ang_scale = self.get_parameter('base_angular_scale').value

        # State variables
        self.joint_positions = np.zeros(9, dtype=np.float32)
        self.joint_velocities = np.zeros(9, dtype=np.float32)
        self.base_pose = np.zeros(3, dtype=np.float32)
        self.object_pos = self.fallback_object_pos.copy()
        self.object_grasped = False
        self.current_phase = int(self.get_parameter('initial_phase').value)
        self.model = None
        self.vecnormalize = None
        self.model_loaded = False
        self.perception_received = False
        self.pregrasp_complete = False
        self.pregrasp_steps = 0
        self.observation_mode = 'legacy27'
        self.prev_action = np.zeros(9, dtype=np.float32)
        self._joint_state_position_map = {}
        self._joint_state_velocity_map = {}

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self._grp_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        

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

    def _resolve_model_path(self, requested_path: str) -> str | None:
        candidates = []
        if requested_path:
            candidates.append(requested_path)
        candidates.extend([
            './rl_models/pickplace_final_model.zip',
            './rl_models/best_model.zip',
            './rl_models/best_model/best_model.zip',
        ])
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            normalized = os.path.abspath(os.path.expanduser(candidate))
            if normalized in seen:
                continue
            seen.add(normalized)
            if os.path.exists(normalized):
                return normalized
        return None

    def _infer_observation_mode(self, model_path: str) -> str:
        try:
            data, _, _ = load_from_zip_file(model_path, device='auto')
            obs_shape = tuple(data['observation_space'].shape)
        except Exception as exc:
            self.get_logger().warn(
                f'Could not inspect checkpoint observation space ({exc}); defaulting to legacy27'
            )
            return 'legacy27'
        if obs_shape == (46,):
            return 'full'
        return 'legacy27'

    def _load_vecnormalize(self, model_path: str) -> None:
        model_dir = os.path.dirname(model_path)
        vecnorm_candidates = [
            os.path.join(model_dir, 'vecnormalize.pkl'),
            os.path.join(model_dir, 'best_vecnormalize.pkl'),
            os.path.join(os.path.dirname(model_dir), 'vecnormalize.pkl'),
        ]
        vecnorm_path = next((path for path in vecnorm_candidates if os.path.exists(path)), None)
        if vecnorm_path is None:
            self.get_logger().warn('VecNormalize stats not found; using raw observations for inference')
            return

        try:
            dummy_env = DummyVecEnv([lambda: _InferenceSpaceEnv(self.obs_dim, self.action_dim)])
            self.vecnormalize = VecNormalize.load(vecnorm_path, dummy_env)
            self.vecnormalize.training = False
            self.vecnormalize.norm_reward = False
            self.get_logger().info(f'Loaded VecNormalize stats from {vecnorm_path}')
        except Exception as exc:
            self.vecnormalize = None
            self.get_logger().warn(f'Could not load VecNormalize stats from {vecnorm_path}: {exc}')

    def load_model(self):
        """Load the trained RL model and matching normalization stats."""
        try:
            from sb3_contrib import TQC

            resolved_model_path = self._resolve_model_path(self.model_path)
            if resolved_model_path is not None:
                self.model_path = resolved_model_path
                self.observation_mode = self._infer_observation_mode(self.model_path)
                self.model = TQC.load(self.model_path)
                self.obs_dim = int(self.model.observation_space.shape[0])
                self.action_dim = int(self.model.action_space.shape[0])
                self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
                self._load_vecnormalize(self.model_path)
                self.model_loaded = True
                self.get_logger().info(
                    f'Loaded {type(self.model).__name__} model from {self.model_path} '
                    f'(obs={self.obs_dim}, action={self.action_dim}, mode={self.observation_mode})'
                )
            else:
                self.get_logger().warn(
                    f'Model not found near requested path {self.model_path} — running in demo mode'
                )
        except ImportError:
            self.get_logger().error(
                'sb3-contrib / stable-baselines3 not installed — cannot load model')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

    def joint_callback(self, msg):
        """Update joint positions."""
        self._joint_state_position_map = dict(zip(msg.name, msg.position))
        self._joint_state_velocity_map = dict(zip(msg.name, msg.velocity))
        for idx, joint_name in enumerate(_ARM_JOINT_NAMES):
            self.joint_positions[idx] = float(self._joint_state_position_map.get(joint_name, 0.0))
            self.joint_velocities[idx] = float(self._joint_state_velocity_map.get(joint_name, 0.0))
        self.joint_positions[6] = float(self._joint_state_position_map.get('finger_joint', 0.0))
        self.joint_velocities[6] = float(self._joint_state_velocity_map.get('finger_joint', 0.0))
        self.joint_positions[7] = float(self._joint_state_position_map.get('left_wheel_joint', 0.0))
        self.joint_positions[8] = float(self._joint_state_position_map.get('right_wheel_joint', 0.0))
        self.joint_velocities[7] = float(self._joint_state_velocity_map.get('left_wheel_joint', 0.0))
        self.joint_velocities[8] = float(self._joint_state_velocity_map.get('right_wheel_joint', 0.0))

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

    def world_point_to_base(self, point_world: np.ndarray) -> np.ndarray:
        bx, by, btheta = self.base_pose
        dx = float(point_world[0]) - float(bx)
        dy = float(point_world[1]) - float(by)
        c = np.cos(-btheta)
        s = np.sin(-btheta)
        return np.array([
            c * dx - s * dy,
            s * dx + c * dy,
            float(point_world[2]) - _BASE_SPAWN_Z,
        ], dtype=np.float32)

    def get_end_effector_pos(self) -> np.ndarray:
        """Compute EE position in world frame using the same FK as the env."""
        ee_in_arm_base = _ur3_fk(self.joint_positions[:6])
        ee_in_base = _ARM_MOUNT_XYZ + np.array(
            [-ee_in_arm_base[0], -ee_in_arm_base[1], ee_in_arm_base[2]],
            dtype=np.float32,
        )
        btheta = float(self.base_pose[2])
        c = np.cos(btheta)
        s = np.sin(btheta)
        return np.array([
            float(self.base_pose[0]) + c * ee_in_base[0] - s * ee_in_base[1],
            float(self.base_pose[1]) + s * ee_in_base[0] + c * ee_in_base[1],
            _BASE_SPAWN_Z + ee_in_base[2],
        ], dtype=np.float32)

    def _angle_error(self, target: float, current: float) -> float:
        error = float(target) - float(current)
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi
        return error

    def _joint_error(self, joint_index: int, target: float) -> float:
        current = float(self.joint_positions[joint_index])
        if joint_index in [0, 1, 3, 4, 5]:
            return self._angle_error(target, current)
        return float(target) - current

    def _limit_safe_joint_vels(self, joint_vels: np.ndarray) -> np.ndarray:
        safe = np.asarray(joint_vels, dtype=np.float32).copy()
        for idx, (lower, upper) in enumerate(_ARM_JOINT_LIMITS):
            pos = float(self.joint_positions[idx])
            if np.isfinite(lower) and pos <= lower + 0.03 and safe[idx] < 0.0:
                safe[idx] = 0.0
            if np.isfinite(upper) and pos >= upper - 0.03 and safe[idx] > 0.0:
                safe[idx] = 0.0
        return safe

    def _scripted_pregrasp_step(self) -> None:
        """Match the training reset pre-grasp before handing off to the policy."""
        obj_pos = self.object_pos
        bx, by, btheta = self.base_pose
        dx = float(obj_pos[0]) - float(bx)
        dy = float(obj_pos[1]) - float(by)

        angle_to = np.arctan2(dy, dx)
        angle_err = self._angle_error(angle_to, btheta)

        twist = Twist()
        twist.angular.z = float(np.clip(angle_err * 2.0, -1.0, 1.0))
        if bx < 0.16 and abs(angle_err) < 0.4:
            twist.linear.x = float(np.clip((0.16 - bx) * 3.0, 0.0, 0.2))
        self.cmd_vel_pub.publish(twist)

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = _ARM_JOINT_NAMES[:4]
        pt = JointTrajectoryPoint()
        desired = [0.0, -1.7, 2.0, -1.0]
        
        target_positions = []
        for idx, target in enumerate(desired):
            err = self._joint_error(idx, target)
            vel = np.clip(err * 2.0, -0.4, 0.4) if abs(err) > 0.05 else 0.0
            # dt is roughly 1/inference_rate, let's use 0.05
            target_positions.append(float(self.joint_positions[idx] + vel * 0.05))
            
        pt.positions = target_positions
        ns = max(int(0.05 * 1e9), 1)
        pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        msg.points = [pt]
        self._arm_pub.publish(msg)

        ee_pos = self.get_end_effector_pos()
        ee_dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        joint_errors = [abs(self._joint_error(idx, target)) for idx, target in enumerate(desired)]
        base_ready = bx >= 0.15 and abs(angle_err) < 0.4
        arm_ready = max(joint_errors) < 0.08
        self.pregrasp_steps += 1
        if ee_dist_xy < 0.30 and base_ready and arm_ready:
            self.cmd_vel_pub.publish(Twist())
            self.pregrasp_complete = True
            self.get_logger().info(
                f'Pregrasp complete after {self.pregrasp_steps} steps '
                f'(ee_xy_dist={ee_dist_xy:.3f}, angle_err={angle_err:.3f})'
            )
        elif self.pregrasp_steps % 200 == 0:
            self.get_logger().warn(
                f'Still pregrasping: ee_xy_dist={ee_dist_xy:.3f}, '
                f'angle_err={angle_err:.3f}, base_x={bx:.3f}, '
                f'joint_err={np.round(joint_errors, 3).tolist()}, '
                f'joints={np.round(self.joint_positions[:4], 3).tolist()}'
            )

    def update_phase(self) -> None:
        """Advance the coarse pick phases using the same thresholds as training."""
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.object_pos
        gripper_pos = float(self.joint_positions[6])

        if self.current_phase == 1:
            grasp_z = float(obj_pos[2]) if obj_pos[2] > 0.02 else 0.055
            dist_z = abs(float(ee_pos[2]) - grasp_z)
            dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            if dist_z < 0.06 and dist_xy < 0.09:
                self.current_phase = 2
                self.get_logger().info('Advanced to phase 2: grasp approach')
        elif self.current_phase == 2:
            xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            z_dist = abs(float(ee_pos[2]) - float(obj_pos[2]))
            if gripper_pos > 0.7 and xy_dist < 0.05 and z_dist < 0.04:
                self.object_grasped = True
                self.current_phase = 3
                self.get_logger().info('Advanced to phase 3: lift attempt')

    def build_observation(self) -> np.ndarray:
        """Build an observation vector matching the trained policy format."""
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.object_pos
        ee_to_obj = obj_pos - ee_pos
        ee_to_target = self.target_pos - ee_pos
        obj_to_target = self.target_pos - obj_pos
        obj_in_base = self.world_point_to_base(obj_pos)
        desired_gripper_pos = 0.8 if self.current_phase in [2, 3, 4] else 0.0
        gripper_error = abs(self.joint_positions[6] - desired_gripper_pos)
        full_obs = np.concatenate([
            self.joint_positions[:6],
            self.joint_velocities[:6],
            [self.joint_positions[6]],
            ee_pos,
            obj_pos,
            ee_to_obj,
            ee_to_target,
            obj_to_target,
            obj_in_base,
            [gripper_error],
            [float(self.object_grasped)],
            [float(self.current_phase)],
            self.base_pose,
            self.prev_action,
        ]).astype(np.float32)
        if self.observation_mode == 'legacy27':
            return np.concatenate([
                full_obs[:22],
                full_obs[32:34],
                full_obs[34:37],
            ]).astype(np.float32)
        return full_obs

    def publish_action(self, action):
        """Publish joint velocities and base twist from action vector."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        joint_vels = action[:6] * self.joint_vel_scale
        joint_vels = self._limit_safe_joint_vels(joint_vels)
        gripper_cmd = float(action[6])
        base_linear = float(action[7]) * self.base_lin_scale
        base_angular = float(action[8]) * self.base_ang_scale
        if self.current_phase in [1, 2, 3]:
            base_linear = 0.0
            base_angular = 0.0

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = _ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        # Convert velocities to position targets based on 1 / inference_rate step duration
        dt = 1.0 / self.get_parameter('inference_rate').value
        target_positions = self.joint_positions[:6] + (joint_vels * dt)
        pt.positions = [float(p) for p in target_positions]
        ns = max(int(dt * 1e9), 1)
        pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        msg.points = [pt]
        self._arm_pub.publish(msg)

        if self._grp_client.server_is_ready():
            goal = GripperCommand.Goal()
            # Map action to 0.0-0.8 range (assuming RL outputs -1 to 1)
            mapped_pos = 0.8 if gripper_cmd > 0 else 0.0
            goal.command.position = float(mapped_pos)
            goal.command.max_effort = 50.0
            self._grp_client.send_goal_async(goal)

        twist = Twist()
        twist.linear.x = base_linear
        twist.angular.z = base_angular
        self.cmd_vel_pub.publish(twist)
        self.prev_action = action.copy()

    def inference_step(self):
        """Run one step of RL inference."""
        if not self.safety_ok:
            self.publish_action(np.zeros(9, dtype=np.float32))
            return

        if not self.model_loaded:
            self.publish_action(np.zeros(9, dtype=np.float32))
            return

        if self.use_perception and not self.perception_received:
            self.get_logger().debug(
                'Waiting for perception data...', throttle_duration_sec=5.0)
            return

        if not self.pregrasp_complete:
            self._scripted_pregrasp_step()
            return

        self.update_phase()

        # Build observation and get action from policy
        obs = self.build_observation()
        if self.vecnormalize is not None:
            obs = self.vecnormalize.normalize_obs(obs.reshape(1, -1))[0]
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
