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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
import time
from pickplace_rl_mobile.domain_randomizer import DomainRandomizer, RandomizationConfig

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


_GRIPPER_MIMIC_MULTIPLIERS = {
    'left_inner_knuckle_joint': 1.0,
    'left_inner_finger_joint': -1.0,
    'right_outer_knuckle_joint': -1.0,
    'right_inner_knuckle_joint': 1.0,
    'right_inner_finger_joint': -1.0,
}


class PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for pick-and-place RL training.

    Observation (46): [joint_pos(6), joint_vel(6), finger_pos(1), ee_pos(3), obj_pos(3),
                       ee_to_obj(3), ee_to_target(3), obj_to_target(3),
                       obj_in_base(3), gripper_error(1), grasped(1), phase(1),
                       base_pose(3), prev_action(9)]
    Action (9):       [joint_vels(6), gripper(1), base_linear(1), base_angular(1)]

    ee_to_obj = obj_pos - ee_pos is the direct tracking vector: if the object moves,
    this immediately reflects the new direction/distance the arm needs to travel.
    """

    def __init__(
        self,
        namespace='',
        ros_domain_id=None,
        gz_partition=None,
        curriculum_stage=0,
        observation_mode='full',
        enable_domain_randomization=True,
    ):
        super().__init__()

        self.gz_partition = gz_partition  # stored for subprocess calls (e.g. _spawn_object)
        self.curriculum_stage = int(curriculum_stage)
        self._pending_curriculum_stage = None
        self.observation_mode = observation_mode
        self.enable_domain_randomization = bool(enable_domain_randomization)

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

        # Observation space: full mode keeps the richer 46-dim state used by newer
        # runs, while legacy27 preserves compatibility with older checkpoints.
        obs_dim = 27 if self.observation_mode == 'legacy27' else 46
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Publishers — relative topic names are prefixed by namespace automatically
        self.cmd_vel_pub = self.node.create_publisher(Twist, 'cmd_vel', 10)
        
        # Arm publisher
        self._arm_pub = self.node.create_publisher(JointTrajectory, 'arm_controller/joint_trajectory', 10)
        
        # Gripper action client
        self._grp_client = ActionClient(self.node, GripperCommand, 'gripper_controller/gripper_cmd')
        
        self._arm_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

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
        self._joint_state_names = []
        self._joint_state_position_map = {}
        self._joint_state_velocity_map = {}
        self.base_pose = np.zeros(3)  # x, y, theta
        self.episode_steps = 0
        self.max_episode_steps = self.episode_step_limit()

        # Targets
        self.object_start_pos = np.array([0.6, 0.0, 0.1325])  # world frame — on top of platform
        self.target_pos = np.array([0.6, 0.5, 0.15])
        self.object_pos = self.object_start_pos.copy()
        self.object_grasped = False
        self.grasp_verified = False
        self.real_object_pos = None  # updated from Gazebo world pose
        self.grasp_verify_steps = 0
        self.grasp_attempts = 0
        self.verified_grasps = 0
        self.max_phase_reached = 0
        self.last_dist_to_obj = np.inf
        self.episode_success = False
        self.stage_success = False

        self.current_phase = 0
        self.prev_distance = None
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.randomizer = None
        if self.enable_domain_randomization:
            self.randomizer = DomainRandomizer(RandomizationConfig(
                obj_x_range=(0.45, 0.75),
                obj_y_range=(-0.15, 0.15),
                obj_z_base=0.1325,
                target_x_range=(0.45, 0.75),
                target_y_range=(0.3, 0.6),
                target_z=float(self.target_pos[2]),
                randomize_target_pos=True,
                randomize_observations=True,
                randomize_actions=True,
                randomize_object_size=True,
                randomize_physics=True,
            ))
            self._apply_stage_randomization(self.curriculum_stage)

    def joint_state_callback(self, msg):
        n = len(msg.position)
        self._joint_state_names = list(msg.name)
        self._joint_state_position_map = {
            name: float(pos) for name, pos in zip(msg.name, msg.position)
        }
        self._joint_state_velocity_map = {
            name: float(vel) for name, vel in zip(msg.name, msg.velocity)
        }
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

    def _ee_world_from_joints(self, joint_angles: np.ndarray) -> np.ndarray:
        ee_in_arm_base = ur3_fk(joint_angles[:6])
        ee_flipped = np.array([-ee_in_arm_base[0], -ee_in_arm_base[1], ee_in_arm_base[2]])
        local_ee = _ARM_MOUNT_XYZ + ee_flipped
        bx, by, btheta = self.base_pose
        return np.array([
            bx + local_ee[0] * np.cos(btheta) - local_ee[1] * np.sin(btheta),
            by + local_ee[0] * np.sin(btheta) + local_ee[1] * np.cos(btheta),
            _BASE_SPAWN_Z + local_ee[2],
        ])

    def _ee_jacobian_world(self) -> np.ndarray:
        """Numerical EE position Jacobian in world frame for stable approach assist."""
        q = self.joint_positions[:6].astype(float).copy()
        base_pos = self._ee_world_from_joints(q)
        jac = np.zeros((3, 6), dtype=np.float64)
        eps = 1e-4
        for i in range(6):
            q_step = q.copy()
            q_step[i] += eps
            jac[:, i] = (self._ee_world_from_joints(q_step) - base_pos) / eps
        return jac

    def _approach_assist_joint_vels(self) -> np.ndarray:
        """
        Small Cartesian pull toward the current curriculum target.

        The RL action remains in charge, but this removes the dead-start problem where
        phase 1 never reaches the object closely enough to expose phase 2 grasp rewards.
        """
        if self.current_phase not in [1, 2]:
            return np.zeros(6, dtype=np.float32)

        obj_pos = self.real_object_pos if self.real_object_pos is not None else self.object_pos
        ee_pos = self.get_global_ee_pos()
        target = obj_pos.copy()
        if self.current_phase == 1:
            target[2] = obj_pos[2] if obj_pos[2] > 0.02 else 0.055

        error = target - ee_pos
        error_norm = float(np.linalg.norm(error))
        if error_norm < 0.025:
            return np.zeros(6, dtype=np.float32)

        desired_ee_vel = np.clip(error * 3.5, -0.35, 0.35)
        jac = self._ee_jacobian_world()
        damping = 0.04
        jj_t = jac @ jac.T
        joint_vels = jac.T @ np.linalg.solve(jj_t + (damping ** 2) * np.eye(3), desired_ee_vel)
        return np.clip(joint_vels, -0.65, 0.65).astype(np.float32)

    def world_to_base_vector(self, vec_world: np.ndarray) -> np.ndarray:
        """Rotate a world-frame vector into the base frame using odometry yaw."""
        btheta = self.base_pose[2]
        return np.array([
            vec_world[0] * np.cos(btheta) + vec_world[1] * np.sin(btheta),
            -vec_world[0] * np.sin(btheta) + vec_world[1] * np.cos(btheta),
            vec_world[2],
        ])

    def world_point_to_base(self, point_world: np.ndarray) -> np.ndarray:
        """Express a world-frame point relative to the robot base frame."""
        base_world = np.array([self.base_pose[0], self.base_pose[1], _BASE_SPAWN_Z])
        return self.world_to_base_vector(point_world - base_world)

    def curriculum_target_phase(self) -> int:
        stage_targets = {
            0: 5,  # full task
            1: 2,  # reach/alignment
            2: 3,  # verified grasp
            3: 4,  # lift
            4: 5,  # transport
            5: 5,  # full placement
        }
        return stage_targets.get(self.curriculum_stage, 5)

    def episode_step_limit(self) -> int:
        """Shorter early-stage episodes improve reset rate and sample efficiency."""
        limits = {
            0: 1000,  # full task
            1: 150,   # reach/alignment only
            2: 250,   # grasp attempts
            3: 350,   # verified grasp + lift
            4: 700,   # transport almost-full task
            5: 1000,
        }
        return limits.get(self.curriculum_stage, 1000)

    # Randomization ranges widen as curriculum advances — early stages stay narrow so the
    # policy can learn the basic motion before facing the full distribution.
    _STAGE_RANDOMIZATION = {
        0: dict(obj_x_range=(0.45, 0.75), obj_y_range=(-0.15, 0.15), randomize_target_pos=True),
        1: dict(obj_x_range=(0.57, 0.63), obj_y_range=(-0.03, 0.03), randomize_target_pos=False),
        2: dict(obj_x_range=(0.54, 0.66), obj_y_range=(-0.06, 0.06), randomize_target_pos=False),
        3: dict(obj_x_range=(0.50, 0.70), obj_y_range=(-0.10, 0.10), randomize_target_pos=False),
        4: dict(obj_x_range=(0.47, 0.73), obj_y_range=(-0.13, 0.13), randomize_target_pos=True),
        5: dict(obj_x_range=(0.45, 0.75), obj_y_range=(-0.15, 0.15), randomize_target_pos=True),
    }

    def _apply_stage_randomization(self, stage: int) -> None:
        if self.randomizer is None:
            return
        params = self._STAGE_RANDOMIZATION.get(stage, self._STAGE_RANDOMIZATION[0])
        self.randomizer.config.obj_x_range = params['obj_x_range']
        self.randomizer.config.obj_y_range = params['obj_y_range']
        self.randomizer.config.randomize_target_pos = params['randomize_target_pos']

    def set_curriculum_stage(self, curriculum_stage: int) -> None:
        """Queue curriculum changes so they take effect cleanly on the next reset."""
        next_stage = int(curriculum_stage)
        if next_stage == self.curriculum_stage:
            self._pending_curriculum_stage = None
            return
        self._pending_curriculum_stage = next_stage

    def curriculum_completed(self) -> bool:
        if self.curriculum_stage <= 0:
            return self.episode_success
        if self.curriculum_stage == 1:
            return self.current_phase >= 2
        if self.curriculum_stage == 2:
            return self.grasp_verified or self.verified_grasps > 0
        if self.curriculum_stage == 3:
            return self.current_phase >= 4
        if self.curriculum_stage == 4:
            return self.current_phase >= 5
        return self.episode_success

    def get_observation(self) -> np.ndarray:
        ee_pos = self.get_global_ee_pos()
        obj_pos = self.real_object_pos if self.real_object_pos is not None else self.object_pos
        # ee_to_obj: direct tracking vector — if the object moves, this updates instantly
        ee_to_obj = obj_pos - ee_pos
        ee_to_target = self.target_pos - ee_pos
        obj_to_target = self.target_pos - obj_pos
        obj_in_base = self.world_point_to_base(obj_pos)
        desired_gripper_pos = 0.8 if self.current_phase in [2, 3, 4] else 0.0
        gripper_error = abs(self.joint_positions[6] - desired_gripper_pos)
        full_obs = np.concatenate([
            self.joint_positions[:6],    # arm joint positions
            self.joint_velocities[:6],   # arm joint velocities
            [self.joint_positions[6]],   # finger_joint position
            ee_pos,                      # EE in world frame
            obj_pos,                     # object in world frame (live from Gazebo bridge)
            ee_to_obj,                   # vector from EE to object — arm tracks this to zero
            ee_to_target,                # vector from EE to final place target
            obj_to_target,               # vector from object to final place target
            obj_in_base,                 # object location in the base frame
            [gripper_error],             # phase-aware gripper opening error
            [float(self.object_grasped)],
            [float(self.current_phase)],
            self.base_pose,
            self.prev_action,
        ])
        if self.observation_mode == 'legacy27':
            obs = np.concatenate([
                full_obs[:22],   # joints, EE/object positions, ee_to_obj
                full_obs[32:34], # grasped, phase
                full_obs[34:37], # base pose
            ]).astype(np.float32)
        else:
            obs = full_obs.astype(np.float32)
        if self.randomizer is not None:
            obs = self.randomizer.add_observation_noise(obs)
        return obs

    def get_joint_position(self, joint_name: str, default=np.nan) -> float:
        return float(self._joint_state_position_map.get(joint_name, default))

    def get_joint_velocity(self, joint_name: str, default=np.nan) -> float:
        return float(self._joint_state_velocity_map.get(joint_name, default))

    def get_gripper_joint_snapshot(self) -> dict:
        finger = self.get_joint_position('finger_joint', self.joint_positions[6])
        snapshot = {
            'finger_joint': finger,
        }
        for joint_name, multiplier in _GRIPPER_MIMIC_MULTIPLIERS.items():
            pos = self.get_joint_position(joint_name)
            snapshot[joint_name] = pos
            snapshot[f'{joint_name}_tracking_error'] = abs(pos - finger * multiplier) if not np.isnan(pos) else np.nan
        return snapshot

    def wait(self, steps: int = 1, action: np.ndarray | None = None):
        if action is None:
            action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        obs = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(steps):
            obs, reward, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                break
        return obs, reward, terminated, truncated, info

    def compute_reward(self):
        reward = 0.0
        terminated = False

        gripper_pos = self.joint_positions[6]  # finger_joint: 0=open, ~0.8=closed
        ee_global = self.get_global_ee_pos()
        local_ee = self.get_end_effector_pos()
        self.max_phase_reached = max(self.max_phase_reached, self.current_phase)

        # Base tipping penalty: if robot falls over, base height deviates significantly
        # Normal base z is ~0.08 (spawn height). If it tilts, z changes dramatically.
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

        # Base-over-object penalty: chassis (38×38cm) has driven on top of the cube.
        # Transform object into base frame and check against half-extents (0.19m each axis).
        if not self.object_grasped:
            bx, by, btheta = self.base_pose
            dx = obj_pos[0] - bx
            dy = obj_pos[1] - by
            dx_local =  dx * np.cos(btheta) + dy * np.sin(btheta)
            dy_local = -dx * np.sin(btheta) + dy * np.cos(btheta)
            if abs(dx_local) < 0.225 and abs(dy_local) < 0.175:
                return -300.0, True  # base crushed the object — end episode

        # Arm-backward constraint: arm pointing behind chassis center risks self-collision.
        # local_ee[0] < 0 means EE is behind the arm mount; < -0.10m is clearly bad.
        if local_ee[0] < -0.10:
            reward -= 5.0 + 20.0 * abs(local_ee[0] + 0.10)

        # EE into ground constraint (tighter): during non-grasp phases EE should stay
        # well above the floor (0.05m vs the hard 0.03m termination threshold).
        if ee_global[2] < 0.05 and self.current_phase not in [1, 2, 5]:
            reward -= 30.0 * (1.0 - ee_global[2] / 0.05)

        if self.current_phase == 1:
            # Phase 1: Lower arm to grasp height AND approach object XY
            grasp_z = obj_pos[2] if obj_pos[2] > 0.02 else 0.055
            dist_z = abs(ee_global[2] - grasp_z)
            dist_xy = np.linalg.norm(ee_global[:2] - obj_pos[:2])
            self.last_dist_to_obj = dist_xy
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

            # Transition once the gripper is close enough for the grasp phase to
            # finish alignment. A slightly wider gate keeps training from getting
            # stranded in approach-only episodes.
            if dist_z < 0.12 and dist_xy < 0.20:
                self.current_phase = 2
                self.prev_distance = None
                reward += 100.0

        elif self.current_phase == 2:
            # Phase 2: Approach object and grasp
            ref_obj = self.real_object_pos if self.real_object_pos is not None else self.object_pos

            # For a 6cm cube, side-grasp target = object center (EE at same height).
            # No upward offset — fingers wrap around the sides at cube midpoint.
            xy_dist = np.linalg.norm(ee_global[:2] - ref_obj[:2])
            z_dist = abs(ee_global[2] - ref_obj[2])
            dist_to_obj = np.linalg.norm(ee_global - ref_obj)
            self.last_dist_to_obj = dist_to_obj

            # Dense reward: guide EE toward object center
            if self.prev_distance is not None:
                delta = self.prev_distance - dist_to_obj
                reward += delta * 100.0 if delta > 0 else delta * 400.0  # 4× harsher when retreating
            self.prev_distance = dist_to_obj

            # Proximity bonus: reward staying near object
            if dist_to_obj < 0.10:
                reward += 8.0 * (1.0 - dist_to_obj / 0.10)

            # Touch-range bonus: EE is at the object surface (~3cm = half cube side)
            if dist_to_obj < 0.04:
                reward += 15.0 * (1.0 - dist_to_obj / 0.04)

            # Very close bonus: within 3cm of center
            if dist_to_obj < 0.03:
                reward += 10.0

            # Reward true side-grasp alignment explicitly so the policy doesn't
            # learn to close while hovering high or offset.
            if xy_dist < 0.05:
                reward += 10.0 * (1.0 - xy_dist / 0.05)
            if z_dist < 0.04:
                reward += 8.0 * (1.0 - z_dist / 0.04)
            if xy_dist < 0.04 and z_dist < 0.03:
                reward += 12.0

            # Reward gripper closing when already near object (actively encourage grasping)
            if gripper_pos > 0.4 and xy_dist < 0.04 and z_dist < 0.03:
                reward += 8.0 * gripper_pos  # more reward the more closed the gripper

            # Penalty for opening gripper when very close (don't retreat from grasp)
            if gripper_pos < 0.2 and dist_to_obj < 0.05:
                reward -= 5.0

            # Closing while vertically misaligned tends to smack or skim the cube
            # instead of wrapping it, so make that behavior clearly unattractive.
            if gripper_pos > 0.5 and z_dist > 0.04:
                reward -= 10.0 * min(z_dist, 0.10)

            # A close, closed gripper starts a lift attempt. The large grasp
            # reward is withheld until the real Gazebo object actually rises.
            if gripper_pos > 0.7 and xy_dist < 0.05 and z_dist < 0.04:
                self.object_grasped = True
                self.grasp_verified = False
                self.current_phase = 3
                self.grasp_verify_steps = 0
                self.prev_distance = None
                self.grasp_attempts += 1
                reward += 50.0
            elif gripper_pos > 0.7 and dist_to_obj >= 0.08:
                # Penalty scales with distance: closing far away = much worse than closing nearby
                reward -= 0.5 + dist_to_obj * 5.0

            # Wrist orientation reward: nudge gripper to horizontal (wrist_2 ≈ 0)
            # so fingers are parallel to ground for side-grasp of cube
            wrist_orient_err = abs(self.joint_positions[4])  # wrist_2_joint
            reward -= wrist_orient_err * 0.3

        elif self.current_phase == 3:
            # Penalise opening gripper while lifting — object will fall
            if gripper_pos < 0.3:
                reward -= 20.0
            # EE should be rising during lift; penalise staying near ground
            if ee_global[2] < 0.10:
                reward -= 8.0 * (1.0 - ee_global[2] / 0.10)

            # Verify grasp: real object should rise with EE; if it stays on the floor, abort.
            # grasp_verify_steps increments unconditionally so the timeout fires even
            # when the Gazebo TF bridge is temporarily silent.
            self.grasp_verify_steps += 1
            if self.real_object_pos is not None:
                obj_xy_err = np.linalg.norm(self.real_object_pos[:2] - ee_global[:2])
                if self.real_object_pos[2] >= 0.08:
                    if not self.grasp_verified:
                        self.grasp_verified = True
                        self.verified_grasps += 1
                        reward += 1000.0
                    reward += max(0.0, 5.0 * (1.0 - obj_xy_err / 0.05))
            if not self.grasp_verified and self.grasp_verify_steps > 30:
                # Object didn't lift within the verify window — grasp failed, back to phase 2.
                # Return immediately so the rest of the phase-3 block (lift tracking,
                # prev_distance update) does not overwrite phase-2 state.
                self.object_grasped = False
                self.current_phase = 2
                self.prev_distance = None
                return reward - 250.0, False
            dist_z = abs(ee_global[2] - 0.25)
            if self.prev_distance is not None:
                delta = self.prev_distance - dist_z
                reward += delta * 100.0 if delta > 0 else delta * 200.0
            self.prev_distance = dist_z
            # Bonus for being near lift height
            if dist_z < 0.08:
                reward += 5.0 * (1.0 - dist_z / 0.08)
            if self.grasp_verified and dist_z < 0.05:
                self.current_phase = 4
                self.prev_distance = None
                reward += 200.0

        elif self.current_phase == 4:
            # Penalise opening gripper during transport — object will fall
            if gripper_pos < 0.3:
                reward -= 20.0
            # EE should stay elevated during transport
            if ee_global[2] < 0.12:
                reward -= 8.0 * (1.0 - ee_global[2] / 0.12)

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
                self.grasp_verified = False
                self.episode_success = True
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

        if self.curriculum_completed():
            self.stage_success = True
            if self.curriculum_stage > 0 and not self.episode_success:
                reward += 300.0
                terminated = True

        return reward, terminated

    def step(self, action):
        rclpy.spin_once(self.node, timeout_sec=0.005)
        action = np.asarray(action, dtype=np.float32)

        # Position delta control: target = current + delta, then P-drive toward target.
        # Max delta per step = 0.25 rad → faster reach toward object.
        delta = action[:6] * 0.25
        target_joints = self.joint_positions[:6] + delta
        # P-controller: velocity = (target - current) * gain, clamped to ±0.5 rad/s
        joint_vels = np.clip((target_joints - self.joint_positions[:6]) * 10.0, -0.5, 0.5)
        approach_assist = self._approach_assist_joint_vels()
        if self.current_phase == 1:
            joint_vels = np.clip(approach_assist + 0.2 * joint_vels, -0.75, 0.75)
        elif self.current_phase == 2:
            joint_vels = np.clip(approach_assist + 0.5 * joint_vels, -0.75, 0.75)
        if self.current_phase in [1, 2]:
            # Wrist roll does not move the EE position; old checkpoints sometimes
            # spin it continuously, which destabilizes Gazebo and wastes episodes.
            joint_vels[5] = np.clip(-0.5 * self.joint_positions[5], -0.5, 0.5)

        gripper_command = -1.0 if self.current_phase == 1 else action[6]

        # Lock base during manipulation phases (1-3) — arm reaches from pregrasp position
        if self.current_phase in [1, 2, 3]:
            base_linear_vel = 0.0
            base_angular_vel = 0.0
        else:
            base_linear_vel = float(action[7]) * 0.5
            base_angular_vel = float(action[8]) * 1.0

        msg = JointTrajectory()
        # Leave stamp at zero so ros2_control executes immediately in sim time.
        # Stamping with wall time schedules goals far in the future under Gazebo.
        msg.joint_names = self._arm_joint_names
        pt = JointTrajectoryPoint()
        # Gazebo training often advances around 5-10 Hz, so a 50 ms target
        # makes the arm crawl. Command farther ahead while keeping velocities
        # clamped above for stable, visible EE motion.
        dt = 0.20
        target_positions = self.joint_positions[:6] + (joint_vels * dt)
        pt.positions = [float(p) for p in target_positions]
        ns = max(int(dt * 1e9), 1)
        pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        msg.points = [pt]
        self._arm_pub.publish(msg)

        if self._grp_client.server_is_ready():
            goal = GripperCommand.Goal()
            mapped_pos = 0.8 if gripper_command > 0 else 0.0
            goal.command.position = float(mapped_pos)
            goal.command.max_effort = 50.0
            self._grp_client.send_goal_async(goal)

        twist_msg = Twist()
        twist_msg.linear.x = base_linear_vel
        twist_msg.angular.z = base_angular_vel
        self.cmd_vel_pub.publish(twist_msg)

        if self.grasp_verified:
            ee_global = self.get_global_ee_pos()
            self.object_pos = ee_global.copy()
            self.object_pos[2] -= 0.05

        # Let ROS process any immediately pending state updates without adding a fixed
        # per-step sleep cost to long training runs.
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self.prev_action = action.copy()

        obs = self.get_observation()
        reward, terminated = self.compute_reward()

        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps
        self.max_phase_reached = max(self.max_phase_reached, self.current_phase)

        info = {
            'phase': int(self.current_phase),
            'max_phase': int(self.max_phase_reached),
            'object_grasped': bool(self.object_grasped),
            'grasp_verified': bool(self.grasp_verified),
            'grasp_attempts': int(self.grasp_attempts),
            'verified_grasps': int(self.verified_grasps),
            'real_object_z': float(self.real_object_pos[2]) if self.real_object_pos is not None else float('nan'),
            'object_height_delta': float((self.real_object_pos[2] - self.object_start_pos[2])) if self.real_object_pos is not None else float('nan'),
            'dist_to_obj': float(self.last_dist_to_obj),
            'finger_joint': float(self.get_joint_position('finger_joint', self.joint_positions[6])),
            'is_success': bool(self.episode_success),
            'curriculum_stage': int(self.curriculum_stage),
            'curriculum_target_phase': int(self.curriculum_target_phase()),
            'stage_success': bool(self.stage_success),
        }

        return obs, reward, terminated, truncated, info

    def _scripted_pregrasp(self):
        """
        Face the object, drive forward only until the front caster (at chassis_x + 0.36m)
        would reach the bin back-wall (outer face ≈ 0.40m), then extend the arm.
        Runs for up to 300 steps before handing off to RL.
        """
        # Bin back-wall outer face ≈ 0.405m; caster front = chassis_x + 0.24m (caster at 0.18m + radius 0.06m).
        # Keep chassis_x ≤ 0.16m so the caster stays clear of the wall.
        SAFE_CHASSIS_X = 0.16

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

            # Keep reset neutral and let the phase-1 Cartesian assist do the
            # actual approach. This avoids stale wrist/arm states carrying over
            # between Gazebo episodes.
            pan_err = 0.0 - self.joint_positions[0]
            s_err   = 0.0 - self.joint_positions[1]
            e_err   = 0.0 - self.joint_positions[2]
            w1_err  = 0.0 - self.joint_positions[3]
            w2_err  = 0.0 - self.joint_positions[4]
            w3_err  = 0.0 - self.joint_positions[5]

            pan_vel = np.clip(pan_err * 2.0, -0.4, 0.4) if abs(pan_err) > 0.05 else 0.0
            s_vel   = np.clip(s_err   * 2.0, -0.4, 0.4) if abs(s_err)   > 0.05 else 0.0
            e_vel   = np.clip(e_err   * 2.0, -0.4, 0.4) if abs(e_err)   > 0.05 else 0.0
            w1_vel  = np.clip(w1_err  * 2.0, -0.4, 0.4) if abs(w1_err)  > 0.05 else 0.0
            w2_vel  = np.clip(w2_err  * 2.0, -0.4, 0.4) if abs(w2_err)  > 0.05 else 0.0
            w3_vel  = np.clip(w3_err  * 2.0, -0.4, 0.4) if abs(w3_err)  > 0.05 else 0.0

            msg = JointTrajectory()
            # Immediate execution in sim time; see step() for why this is zero.
            msg.joint_names = self._arm_joint_names
            pt = JointTrajectoryPoint()
            target_positions = [
                float(self.joint_positions[0] + pan_vel * 0.05),
                float(self.joint_positions[1] + s_vel * 0.05),
                float(self.joint_positions[2] + e_vel * 0.05),
                float(self.joint_positions[3] + w1_vel * 0.05),
                float(self.joint_positions[4] + w2_vel * 0.05),
                float(self.joint_positions[5] + w3_vel * 0.05),
            ]
            pt.positions = target_positions
            ns = max(int(0.05 * 1e9), 1)
            pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            msg.points = [pt]
            self._arm_pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.005)

            # Break once the arm has actually moved into the pre-grasp pose and
            # the robot is facing the object. Previously this could trigger at
            # the zero joint pose, handing RL an approach state with the gripper
            # laterally offset from the cube.
            ee_global = self.get_global_ee_pos()
            ee_dist_xy = np.linalg.norm(ee_global[:2] - obj_pos[:2])
            arm_pose_err = max(abs(pan_err), abs(s_err), abs(e_err), abs(w1_err), abs(w2_err), abs(w3_err))
            if ee_dist_xy < 0.32 and abs(angle_err) < 0.4 and arm_pose_err < 0.15:
                self.cmd_vel_pub.publish(Twist())
                break

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

        if self._pending_curriculum_stage is not None:
            self.curriculum_stage = self._pending_curriculum_stage
            self._pending_curriculum_stage = None
            self._apply_stage_randomization(self.curriculum_stage)

        self.episode_steps = 0
        self.object_grasped = False
        self.grasp_verified = False
        self.base_pose = np.zeros(3, dtype=np.float32)
        self.current_phase = 1  # start directly at lowering phase (base already positioned)
        self.max_episode_steps = self.episode_step_limit()
        self.prev_distance = None
        self.grasp_verify_steps = 0
        self.grasp_attempts = 0
        self.verified_grasps = 0
        self.max_phase_reached = self.current_phase
        self.last_dist_to_obj = np.inf
        self.episode_success = False
        self.stage_success = False
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        if self.randomizer is not None:
            if seed is not None:
                self.randomizer.seed(seed)
            self.randomizer.reset_episode()
            self.object_start_pos = self.randomizer.randomize_object_position().astype(np.float32)
            self.target_pos = self.randomizer.randomize_target_position().astype(np.float32)
        else:
            # Randomize object XY ±3cm so the policy generalises, not memorises one spot
            rng = np.random.default_rng(seed)
            ox = 0.6 + rng.uniform(-0.03, 0.03)
            oy = 0.0 + rng.uniform(-0.03, 0.03)
            oz = 0.1325  # on top of platform (platform top=0.10m + cube half=0.0325m)
            self.object_start_pos = np.array([ox, oy, oz], dtype=np.float32)
            self.target_pos = np.array([0.6, 0.5, 0.15], dtype=np.float32)
        self.object_pos = self.object_start_pos.copy()
        self.real_object_pos = None
        ox, oy, oz = self.object_start_pos
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
