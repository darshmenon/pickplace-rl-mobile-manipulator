#!/usr/bin/env python3
"""
ROS 2 node that runs the whole-body controller (wbc_baseline.py) on the Gazebo
robot, as a classical drop-in for manip_rl_node.

The controller needs a Jacobian, so the node keeps a kinematic shadow of the
robot in MuJoCo: every tick it copies the measured arm joints, base pose and
detected cube pose into the shadow model, asks the controller for its
normalised 9-D action, and publishes it exactly the way manip_rl_node
publishes a policy action (joint trajectory point, gripper goal, /cmd_vel).

  ros2 run pickplace_rl_mobile wbc_node
"""

import json

import numpy as np
import mujoco
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo
from pickplace_rl_mobile.wbc_baseline import (
    WholeBodyController, _ACTION_TO_VEL, _ARM_DT, _ARM_MAX_VEL, _LIFT_Z)

_ARM_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
_GRIPPER_CLOSED_POS = 0.8  # Gazebo Robotiq finger_joint target for "closed"


class WbcNode(Node):
    def __init__(self):
        super().__init__('wbc_node')
        self.declare_parameter('rate', 5.0)  # the controller's step is 0.2 s
        self.declare_parameter('target_pos', [0.6, 0.5, 0.15])
        self.declare_parameter('fallback_object_pos', [0.6, 0.0, 0.1325])
        self.declare_parameter('use_perception', True)
        self.declare_parameter('gravity_ff', 0.0)        # Gazebo holds the arm itself
        self.declare_parameter('grasp_finger_pos', 0.35)  # finger_joint reading = cube held
        self.declare_parameter('grasp_hold_ticks', 3)

        self.rate = float(self.get_parameter('rate').value)
        self.target = np.array(self.get_parameter('target_pos').value, dtype=float)
        self.object_pos = np.array(self.get_parameter('fallback_object_pos').value, dtype=float)
        self.use_perception = bool(self.get_parameter('use_perception').value)
        self.grasp_finger = float(self.get_parameter('grasp_finger_pos').value)
        self.hold_ticks = int(self.get_parameter('grasp_hold_ticks').value)

        # kinematic shadow of the robot, used only for Jacobians and phase tests
        self.env = PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True,
                                      curriculum_stage=0, ee_site='pinch')
        self.ctrl = WholeBodyController(
            self.env, ee_site='pinch',
            grav_ff=float(self.get_parameter('gravity_ff').value))
        self.env.target_pos = self.target.astype(np.float32)

        self.q = np.zeros(6)
        self.finger = 0.0
        self.base = np.zeros(3)
        self.have_joints = self.have_odom = False
        self.perception_received = not self.use_perception
        self.safety_ok = True
        self.phase = 1
        self.closed_ticks = 0
        self.done = False

        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        if self.use_perception:
            self.create_subscription(PoseStamped, '/perception/detected_object',
                                     self._on_object, 10)
        self.create_subscription(String, '/safety/status', self._on_safety, 10)
        self._arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._grp = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info('WBC node ready')

    # ------------------------------------------------------------------ inputs
    def _on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in _ARM_JOINT_NAMES):
            self.q = np.array([pos[n] for n in _ARM_JOINT_NAMES])
            self.finger = float(pos.get('finger_joint', 0.0))
            self.have_joints = True

    def _on_odom(self, msg):
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = np.arctan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        self.base = np.array([p.x, p.y, yaw])
        self.have_odom = True

    def _on_object(self, msg):
        p = msg.pose.position
        self.object_pos = np.array([p.x, p.y, p.z])
        self.perception_received = True

    def _on_safety(self, msg):
        try:
            self.safety_ok = not json.loads(msg.data).get('e_stop', False)
        except (json.JSONDecodeError, AttributeError):
            pass

    # ------------------------------------------------------------------ shadow
    def _sync_shadow(self):
        env, d = self.env, self.env.data
        d.qpos[env._arm_qpos_adr] = self.q
        d.qpos[env._gripper_qpos_adr] = self.finger
        d.qpos[env._base_x_qpos], d.qpos[env._base_y_qpos] = self.base[:2]
        d.qpos[env._base_yaw_qpos] = self.base[2]
        a = env._object_qpos_adr
        d.qpos[a:a + 3] = self.object_pos
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(env.model, d)
        env._pull_state()
        env.current_phase = self.phase

    def _update_phase(self):
        pinch = self.env.data.site_xpos[self.ctrl.pinch_id]
        # fingers closed AND the cube between them (closing on air also reads as closed)
        held = (self.finger > self.grasp_finger
                and np.linalg.norm(self.object_pos - pinch) < 0.05)
        if self.phase in (1, 2):
            self.phase = 2 if self.ctrl.hold > 0 else 1
            self.closed_ticks = self.closed_ticks + 1 if held else 0
            if self.closed_ticks >= self.hold_ticks:
                self.phase = 3
                self.get_logger().info('phase 3: lift')
        elif self.phase == 3:
            if not held:  # dropped it, start over
                self.phase, self.closed_ticks = 1, 0
                self.ctrl.reset()
            elif pinch[2] >= _LIFT_Z - 0.03:
                self.phase = 4
                self.get_logger().info('phase 4: transport')
        elif self.phase == 4:
            if np.linalg.norm(pinch[:2] - self.target[:2]) < 0.15:
                self.phase = 5
                self.get_logger().info('phase 5: place')

    # ------------------------------------------------------------------ output
    def _stop(self):
        self._cmd_pub.publish(Twist())

    def _tick(self):
        if not (self.have_joints and self.have_odom and self.perception_received):
            return
        if not self.safety_ok or self.done:
            self._stop()
            return
        self._sync_shadow()
        self._update_phase()
        self.env.current_phase = self.phase
        a = self.ctrl.act()
        if self.phase == 5 and a[6] < 0 and self.finger < 0.1:
            self.done = True
            self._stop()
            self.get_logger().info('place complete')
            return
        self._publish(a)

    def _publish(self, a):
        vel = np.clip(_ACTION_TO_VEL * a[:6], -_ARM_MAX_VEL, _ARM_MAX_VEL)
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in self.q + vel * _ARM_DT]
        ns = int(_ARM_DT * 1e9)
        pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = _ARM_JOINT_NAMES
        traj.points = [pt]
        self._arm_pub.publish(traj)

        if self._grp.server_is_ready():
            goal = GripperCommand.Goal()
            goal.command.position = _GRIPPER_CLOSED_POS if a[6] > 0 else 0.0
            goal.command.max_effort = 50.0
            self._grp.send_goal_async(goal)

        twist = Twist()
        twist.linear.x = float(a[7]) * 0.5
        twist.angular.z = float(a[8]) * 1.0
        self._cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WbcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
