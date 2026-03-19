#!/usr/bin/env python3
"""
Navigate-and-Pick Node  (NEW)
Bridges autonomous navigation with VLA pick-and-place.

Given a pick instruction, this node:
  1. Looks up the target object in the object memory
  2. Navigates the base to within arm-reach of the object (Nav2)
  3. Triggers the VLA pipeline to perform the pick-and-place
  4. Optionally navigates to the drop zone after placement

This is the missing link between the navigation stack and the manipulation stack.

Topics / Services
-----------------
Sub:  /vla/object_map        — timestamped object positions (from object_memory_node)
Sub:  /vla/task_feedback     — completion status (from coordinator)
Sub:  /odom                  — robot base pose
Pub:  /vla_instruction       — emits pick instruction when base is in position
Srv:  /navigate_and_pick     — (Trigger) externally trigger nav+pick for current task
Action client: navigate_to_pose (Nav2)
"""

import json
import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

try:
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False

# How close the base needs to be before we attempt a pick (metres)
PICK_RADIUS   = 0.7
# Offset from the object so the arm can reach it (metres, behind the object)
APPROACH_DIST = 0.5


class NavigateAndPickNode(Node):
    def __init__(self):
        super().__init__('navigate_and_pick')

        self.cb_group = ReentrantCallbackGroup()

        self._object_map: dict          = {}
        self._base_x:     float         = 0.0
        self._base_y:     float         = 0.0
        self._pending:    dict | None   = None  # {color, destination, place_xyz}
        self._nav_active: bool          = False

        self.create_subscription(String,   '/vla/object_map',    self._map_cb,    10)
        self.create_subscription(String,   '/vla/task_feedback', self._feedback_cb, 10)
        self.create_subscription(Odometry, '/odom',              self._odom_cb,   10)

        self._instr_pub = self.create_publisher(String, '/vla_instruction', 10)

        self.create_service(Trigger, '/navigate_and_pick', self._nav_pick_cb,
                            callback_group=self.cb_group)

        if NAV2_AVAILABLE:
            self._nav_client = ActionClient(
                self, NavigateToPose, 'navigate_to_pose',
                callback_group=self.cb_group
            )
            self.get_logger().info('Nav2 action client created.')
        else:
            self._nav_client = None
            self.get_logger().warn('nav2_msgs not found — navigation disabled.')

        self.get_logger().info(
            'Navigate-and-Pick ready.\n'
            '  Call: ros2 service call /navigate_and_pick std_srvs/Trigger'
        )

    # ------------------------------------------------------------------
    def _map_cb(self, msg: String):
        try:
            self._object_map = json.loads(msg.data)
        except Exception:
            pass

    def _odom_cb(self, msg: Odometry):
        self._base_x = msg.pose.pose.position.x
        self._base_y = msg.pose.pose.position.y

    def _feedback_cb(self, msg: String):
        try:
            fb = json.loads(msg.data)
        except Exception:
            return
        if fb.get('status') == 'completed' and self._pending:
            self.get_logger().info('[Nav&Pick] Pick completed. Task done.')
            self._pending    = None
            self._nav_active = False

    # ------------------------------------------------------------------
    def _nav_pick_cb(self, request, response):
        if self._nav_active:
            response.success = False
            response.message = 'Already navigating.'
            return response

        if not self._object_map:
            response.success = False
            response.message = 'Object map is empty. No objects detected yet.'
            return response

        # Pick the closest detected object
        target_label, target_entry = min(
            self._object_map.items(),
            key=lambda kv: self._dist(kv[1]['x'], kv[1]['y'])
        )
        self.get_logger().info(f'[Nav&Pick] Target: "{target_label}" at '
                               f'({target_entry["x"]:.2f}, {target_entry["y"]:.2f})')

        dist = self._dist(target_entry['x'], target_entry['y'])

        # Already within reach — skip navigation
        if dist <= PICK_RADIUS:
            self.get_logger().info('[Nav&Pick] Already in reach. Triggering pick directly.')
            self._trigger_pick(target_label)
            response.success = True
            response.message = f'Picking {target_label} from current position.'
            return response

        # Navigate to approach position
        if NAV2_AVAILABLE and self._nav_client is not None \
                and self._nav_client.server_is_ready():
            goal_x, goal_y = self._approach_pose(target_entry['x'], target_entry['y'])
            self._pending    = {'color': target_label, 'entry': target_entry}
            self._nav_active = True
            self._send_nav_goal(goal_x, goal_y, target_label)
            response.success = True
            response.message = f'Navigating to {target_label}.'
        else:
            self.get_logger().warn('[Nav&Pick] Nav2 unavailable. Triggering pick in place.')
            self._trigger_pick(target_label)
            response.success = True
            response.message = f'Picking {target_label} without navigation.'

        return response

    # ------------------------------------------------------------------
    def _approach_pose(self, obj_x: float, obj_y: float) -> tuple[float, float]:
        """Compute a position APPROACH_DIST metres behind the object from the robot."""
        dx = obj_x - self._base_x
        dy = obj_y - self._base_y
        dist = math.hypot(dx, dy) or 1.0
        # Stand APPROACH_DIST m away from the object, facing it
        return (obj_x - dx / dist * APPROACH_DIST,
                obj_y - dy / dist * APPROACH_DIST)

    def _dist(self, obj_x: float, obj_y: float) -> float:
        return math.hypot(obj_x - self._base_x, obj_y - self._base_y)

    def _send_nav_goal(self, goal_x: float, goal_y: float, label: str):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id    = 'map'
        goal_msg.pose.header.stamp       = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x    = goal_x
        goal_msg.pose.pose.position.y    = goal_y
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f'[Nav&Pick] Navigating to ({goal_x:.2f}, {goal_y:.2f})...')

        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self._nav_goal_response(f, label))

    def _nav_goal_response(self, future, label: str):
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().error('[Nav&Pick] Nav2 goal rejected.')
            self._nav_active = False
            return
        gh.get_result_async().add_done_callback(lambda f: self._nav_result(f, label))

    def _nav_result(self, future, label: str):
        try:
            future.result()
            self.get_logger().info(f'[Nav&Pick] Arrived. Triggering pick for "{label}".')
            self._trigger_pick(label)
        except Exception as e:
            self.get_logger().error(f'[Nav&Pick] Navigation failed: {e}')
            self._nav_active = False

    def _trigger_pick(self, label: str):
        msg      = String()
        msg.data = f'pick the {label} and place it in the tray'
        self._instr_pub.publish(msg)
        self.get_logger().info(f'[Nav&Pick] Published pick instruction for "{label}".')


def main(args=None):
    rclpy.init(args=args)
    node = NavigateAndPickNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
