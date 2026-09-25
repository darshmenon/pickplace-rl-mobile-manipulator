#!/usr/bin/env python3

import os
import re
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

from pickplace_rl_mobile.pickplace_env_base import (
    PickPlaceEnvBase, ur3_fk, _PLATFORM_TOP,
)

# Re-exported for backward compatibility with callers that imported these
# constants/helpers from pickplace_env before the MuJoCo-portable split.
__all__ = ['PickPlaceEnv', 'ur3_fk']


class PickPlaceEnv(PickPlaceEnvBase):
    """Gazebo/ROS2-backed pick-and-place environment.

    All reward/observation/curriculum/control-law logic lives in
    PickPlaceEnvBase (pickplace_env_base.py) so it is shared byte-for-byte
    with PickPlaceEnvMuJoCo (pickplace_env_mujoco.py). This class only
    implements the Gazebo-specific I/O: ROS publishers/subscribers, the
    gz-transport object-pose poll, and gz-service object spawn/physics calls.
    """

    def __init__(
        self,
        namespace='',
        ros_domain_id=None,
        gz_partition=None,
        curriculum_stage=0,
        observation_mode='full',
        enable_domain_randomization=True,
        enable_assist=True,
    ):
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

        # Spin ROS callbacks in a background thread so joint states and odometry
        # are always fresh — spin_once in step() was too slow to catch messages.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Directly poll Gazebo for pickup_object pose via gz CLI as fallback when
        # the ROS TF bridge doesn't deliver messages to this node (DDS discovery gap).
        self._gz_poll_stop = threading.Event()
        self._gz_poll_thread = threading.Thread(
            target=self._gz_pose_poll_loop, daemon=True)
        self._gz_poll_thread.start()

        super().__init__(
            curriculum_stage=curriculum_stage,
            observation_mode=observation_mode,
            enable_domain_randomization=enable_domain_randomization,
            enable_assist=enable_assist,
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

    # ------------------------------------------------------------------
    # ROS callbacks — populate the shared base class's state fields.
    # ------------------------------------------------------------------

    def _gz_pose_poll_loop(self):
        """Poll Gazebo directly for pickup_object pose at ~4 Hz.

        Bypasses the ROS TF bridge, which has intermittent DDS discovery issues
        that prevent the world_pose_callback from firing. The gz CLI connects
        directly to the Gazebo transport layer using GZ_PARTITION.
        """
        env = os.environ.copy()
        if self.gz_partition:
            env['GZ_PARTITION'] = self.gz_partition
        while not self._gz_poll_stop.is_set():
            try:
                result = subprocess.run(
                    ['gz', 'topic', '-e', '-n', '1', '-t',
                     '/world/pickplace_world/dynamic_pose/info'],
                    env=env, capture_output=True, text=True, timeout=3.0
                )
                if result.returncode == 0:
                    output = result.stdout
                    in_pickup = False
                    in_position = False
                    x = y = z = None
                    for line in output.splitlines():
                        stripped = line.strip()
                        if 'name: "pickup_object"' in line:
                            in_pickup = True
                            in_position = False
                            x = y = z = None
                        elif in_pickup and stripped.startswith('name:') and 'pickup_object' not in line:
                            break  # moved past the pickup_object pose block
                        if in_pickup:
                            if stripped == 'position {':
                                in_position = True
                            elif in_position and stripped == '}':
                                in_position = False
                            elif stripped.startswith('orientation'):
                                break  # past position, done with this pose
                        if in_position:
                            mx = re.search(r'\bx:\s*([-\d.eE+]+)', line)
                            my = re.search(r'\by:\s*([-\d.eE+]+)', line)
                            mz = re.search(r'\bz:\s*([-\d.eE+]+)', line)
                            if mx:
                                x = float(mx.group(1))
                            if my:
                                y = float(my.group(1))
                            if mz:
                                z = float(mz.group(1))
                    if x is not None and y is not None and z is not None:
                        self.real_object_pos = np.array([x, y, z])
            except Exception:
                pass
            self._gz_poll_stop.wait(timeout=0.25)

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

    # ------------------------------------------------------------------
    # Backend hooks (see PickPlaceEnvBase)
    # ------------------------------------------------------------------

    def _drive_arm(self, target_positions, dt):
        msg = JointTrajectory()
        # Leave stamp at zero so ros2_control executes immediately in sim time.
        # Stamping with wall time schedules goals far in the future under Gazebo.
        msg.joint_names = self._arm_joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in target_positions]
        ns = max(int(dt * 1e9), 1)
        pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        msg.points = [pt]
        self._arm_pub.publish(msg)

    def _drive_gripper(self, target_open_frac):
        if self._grp_client.server_is_ready():
            goal = GripperCommand.Goal()
            goal.command.position = float(target_open_frac)
            goal.command.max_effort = 50.0
            self._grp_client.send_goal_async(goal)

    def _drive_base(self, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        self.cmd_vel_pub.publish(twist_msg)

    def _advance_physics(self, dt):
        # Gazebo runs asynchronously in real time; give it one controller cycle
        # to process the trajectory command and send back updated joint states.
        time.sleep(0.01)

    def _refresh_state(self):
        pass  # joint_positions/base_pose/real_object_pos are updated by ROS callbacks

    def _wait_until_ready(self):
        if not self._joint_states_received:
            self.node.get_logger().info('Waiting for /joint_states...')
            while not self._joint_states_received:
                time.sleep(0.1)

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

    def _respawn_object_randomized(self, x, y, z):
        """Delete and recreate pickup_object with randomized color, mass, size, friction.

        Falls back to set_pose if the randomizer has no episode params (e.g. randomization disabled).
        Writes a temp SDF file so the SDF string doesn't have to be shell-escaped.
        """
        if self.randomizer is None:
            self._spawn_object(x, y, z)
            return z

        params = self.randomizer.get_episode_params()
        r, g, b, _ = self.randomizer.get_object_color_rgba()
        mass = max(0.1, 0.5 + params.get('mass_noise', 0.0))
        friction = float(np.clip(params.get('friction', 1.0) * 1.5, 0.3, 3.0))
        size = float(np.clip(params.get('obj_size', 0.065), 0.045, 0.085))
        z = _PLATFORM_TOP + size / 2.0
        # Inertia for a solid cube: I = m * side^2 / 6
        inertia = mass * size * size / 6.0

        shape = params.get('shape', 'box')
        rad = size / 2.0
        if shape == 'cylinder':
            # Solid cylinder standing upright: radius=size/2, height=size
            ixx = mass * (3 * rad * rad + size * size) / 12.0
            izz = mass * rad * rad / 2.0
            geom = (
                f'<cylinder><radius>{rad:.4f}</radius>'
                f'<length>{size:.4f}</length></cylinder>'
            )
            inertia_xml = (
                f'<ixx>{ixx:.6f}</ixx><iyy>{ixx:.6f}</iyy><izz>{izz:.6f}</izz>'
            )
        elif shape == 'sphere':
            # Solid sphere: radius=size/2
            isph = 2.0 * mass * rad * rad / 5.0
            geom = f'<sphere><radius>{rad:.4f}</radius></sphere>'
            inertia_xml = (
                f'<ixx>{isph:.6f}</ixx><iyy>{isph:.6f}</iyy><izz>{isph:.6f}</izz>'
            )
        else:
            geom = (
                f'<box><size>{size:.4f} {size:.4f} {size:.4f}</size></box>'
            )
            inertia_xml = (
                f'<ixx>{inertia:.6f}</ixx><iyy>{inertia:.6f}</iyy>'
                f'<izz>{inertia:.6f}</izz>'
            )

        sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="pickup_object">
    <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{mass:.4f}</mass>
        <inertia>{inertia_xml}</inertia>
      </inertial>
      <velocity_decay><linear>0.5</linear><angular>1.0</angular></velocity_decay>
      <visual name="visual">
        <geometry>{geom}</geometry>
        <material>
          <ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient>
          <diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>{geom}</geometry>
        <surface>
          <friction><ode><mu>{friction:.3f}</mu><mu2>{friction:.3f}</mu2></ode></friction>
          <contact><ode><kp>100000</kp><kd>500</kd><max_vel>0.01</max_vel><min_depth>0.001</min_depth></ode></contact>
        </surface>
      </collision>
    </link>
  </model>
</sdf>"""

        sdf_path = f'/tmp/pickup_object_{self.gz_partition or "default"}.sdf'
        with open(sdf_path, 'w') as f:
            f.write(sdf)

        env = os.environ.copy()
        if self.gz_partition:
            env['GZ_PARTITION'] = self.gz_partition

        # Remove existing model
        subprocess.run(
            ['gz', 'service', '-s', '/world/pickplace_world/remove',
             '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '2000', '--req', 'name: "pickup_object" type: 2'],
            env=env, capture_output=True
        )
        time.sleep(0.1)

        # Spawn new model from temp SDF file
        subprocess.run(
            ['gz', 'service', '-s', '/world/pickplace_world/create',
             '--reqtype', 'gz.msgs.EntityFactory', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '3000', '--req', f'sdf_filename: "{sdf_path}"'],
            env=env, capture_output=True
        )
        time.sleep(0.15)
        return z

    def _randomize_gravity(self):
        """Perturb gravity z slightly via gz set_physics service for sim-to-real transfer."""
        if self.randomizer is None:
            return
        params = self.randomizer.get_episode_params()
        gravity_z = -9.81 + float(params.get('gravity_noise', 0.0))
        env = os.environ.copy()
        if self.gz_partition:
            env['GZ_PARTITION'] = self.gz_partition
        req = f'gravity: {{x: 0 y: 0 z: {gravity_z:.4f}}}'
        subprocess.run(
            ['gz', 'service', '-s', '/world/pickplace_world/set_physics',
             '--reqtype', 'gz.msgs.Physics', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '1000', '--req', req],
            env=env, capture_output=True
        )

    def close(self):
        self._gz_poll_stop.set()
        self._executor.shutdown()
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
