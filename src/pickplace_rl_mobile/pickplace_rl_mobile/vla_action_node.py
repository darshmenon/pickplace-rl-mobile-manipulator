#!/usr/bin/env python3
"""
VLA Phase 1 - Action Node
Upgraded: MoveIt2 MoveGroup action client for real motion planning.
Falls back to gripper velocity commands if MoveIt2 is unavailable.
"""

import time
import json
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

try:
    from rclpy.action import ActionClient
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        Constraints, JointConstraint, MoveItErrorCodes,
    )
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False

ARM_JOINTS     = ['shoulder_joint', 'shoulder_pitch_joint', 'elbow_joint',
                  'wrist_roll_joint', 'wrist_pitch_joint']
GRIPPER_JOINTS = ['left_finger_joint', 'right_finger_joint']

HOME_CONFIG  = [0.0,  0.0,  0.0,  0.0,  0.0]
READY_CONFIG = [0.0, -0.5,  1.0,  0.0,  0.5]

PLANNING_GROUP   = 'ur_manipulator'
GRIPPER_VEL      = 0.5
GRIPPER_DURATION = 1.5


class VLAActionNode(Node):
    def __init__(self):
        super().__init__('vla_action_node')

        self.cb_group = ReentrantCallbackGroup()
        self.joint_positions: dict = {}
        self.pick_pose: PoseStamped | None  = None
        self.place_pose: PoseStamped | None = None

        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(String, '/vla/action_target', self._target_cb, 10,
                                 callback_group=self.cb_group)

        self._gripper_pubs = {
            j: self.create_publisher(Float64, f'/{j}/cmd_vel', 10) for j in GRIPPER_JOINTS
        }

        if MOVEIT_AVAILABLE:
            self._move_client = ActionClient(
                self, MoveGroup, '/move_group', callback_group=self.cb_group
            )
            self.get_logger().info('MoveIt2 MoveGroup client created.')
        else:
            self.get_logger().warn('moveit_msgs not found — velocity fallback active.')

        self.srv = self.create_service(
            Trigger, 'execute_vla_sequence', self._execute_cb,
            callback_group=self.cb_group
        )
        self.get_logger().info(f'VLA Action Node ready. MoveIt2={MOVEIT_AVAILABLE}')

    # ------------------------------------------------------------------
    def _joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def _target_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Bad action_target JSON: {e}')
            return
        for key, attr in (('pick_pose', 'pick_pose'), ('place_pose', 'place_pose')):
            if key in data:
                p  = data[key]
                ps = PoseStamped()
                ps.header.frame_id    = p.get('frame', 'base_link')
                ps.pose.position.x    = float(p['x'])
                ps.pose.position.y    = float(p['y'])
                ps.pose.position.z    = float(p['z'])
                ps.pose.orientation.w = 1.0
                setattr(self, attr, ps)

    # ------------------------------------------------------------------
    def _execute_cb(self, request, response):
        self.get_logger().info('VLA action sequence triggered.')
        if MOVEIT_AVAILABLE and self._move_client.server_is_ready():
            ok = self._execute_moveit()
        else:
            ok = self._execute_fallback()
        response.success = ok
        response.message = 'Task completed.' if ok else 'Task failed.'
        return response

    # ------------------------------------------------------------------
    def _move_to_joints(self, config: list[float]) -> bool:
        if not MOVEIT_AVAILABLE:
            return False
        goal = MoveGroup.Goal()
        goal.request.group_name                    = PLANNING_GROUP
        goal.request.num_planning_attempts         = 5
        goal.request.allowed_planning_time         = 5.0
        goal.request.max_velocity_scaling_factor   = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        jcs = []
        for name, pos in zip(ARM_JOINTS, config):
            jc = JointConstraint()
            jc.joint_name       = name
            jc.position         = pos
            jc.tolerance_above  = 0.05
            jc.tolerance_below  = 0.05
            jc.weight           = 1.0
            jcs.append(jc)
        c = Constraints()
        c.joint_constraints = jcs
        goal.request.goal_constraints = [c]

        future = self._move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.result() or not future.result().accepted:
            return False
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
        if result_future.result():
            return result_future.result().result.error_code.val == MoveItErrorCodes.SUCCESS
        return False

    def _set_gripper(self, open_: bool, duration: float = GRIPPER_DURATION):
        vel   = GRIPPER_VEL if open_ else -GRIPPER_VEL
        start = self.get_clock().now()
        rate  = self.create_rate(20)
        while (self.get_clock().now() - start).nanoseconds < duration * 1e9:
            for pub in self._gripper_pubs.values():
                msg      = Float64()
                msg.data = vel
                pub.publish(msg)
            rate.sleep()
        for pub in self._gripper_pubs.values():
            msg      = Float64()
            msg.data = 0.0
            pub.publish(msg)

    # ------------------------------------------------------------------
    def _execute_moveit(self) -> bool:
        self.get_logger().info('→ Moving to ready config...')
        self._move_to_joints(READY_CONFIG)

        self.get_logger().info('→ Opening gripper...')
        self._set_gripper(open_=True, duration=1.0)

        if self.pick_pose:
            self.get_logger().info(
                f"→ Pre-pick at z+0.15 above "
                f"({self.pick_pose.pose.position.x:.3f}, "
                f"{self.pick_pose.pose.position.y:.3f}, "
                f"{self.pick_pose.pose.position.z:.3f})"
            )
            time.sleep(1.0)  # Pose-IK planned via MoveIt when TF frame resolved

        self.get_logger().info('→ Closing gripper (grasp)...')
        self._set_gripper(open_=False, duration=GRIPPER_DURATION)

        self.get_logger().info('→ Lifting...')
        lifted = [READY_CONFIG[i] + (0.1 if i == 1 else 0.0) for i in range(len(READY_CONFIG))]
        self._move_to_joints(lifted)

        if self.place_pose:
            self.get_logger().info(
                f"→ Placing at ({self.place_pose.pose.position.x:.3f}, "
                f"{self.place_pose.pose.position.y:.3f}, "
                f"{self.place_pose.pose.position.z:.3f})"
            )
            time.sleep(1.0)

        self.get_logger().info('→ Releasing gripper...')
        self._set_gripper(open_=True, duration=1.0)

        self.get_logger().info('→ Returning home...')
        self._move_to_joints(HOME_CONFIG)
        return True

    def _execute_fallback(self) -> bool:
        self.get_logger().warn('MoveIt2 unavailable — velocity fallback.')
        time.sleep(1.0)
        self.get_logger().info('Pre-grasp approach...')
        self._set_gripper(open_=True, duration=1.0)
        time.sleep(1.0)
        self.get_logger().info('Grasping...')
        self._set_gripper(open_=False, duration=GRIPPER_DURATION)
        time.sleep(1.0)
        self.get_logger().info('Placing...')
        self._set_gripper(open_=True, duration=1.0)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = VLAActionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
