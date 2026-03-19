#!/usr/bin/env python3
"""
Navigate-and-Pick Node
Bridges autonomous navigation with VLA pick-and-place.

Given a pick instruction, this node:
  1. Looks up the target object in the object memory (camera_link frame)
  2. Transforms the object position to map frame via TF2
  3. Navigates the base to within arm-reach of the object (Nav2)
  4. Triggers the VLA pipeline to perform the pick-and-place

Topics / Services
-----------------
Sub:  /vla/object_map        — timestamped object positions (camera_link frame)
Sub:  /vla/task_feedback     — completion status (from coordinator)
Pub:  /vla_instruction       — emits pick instruction when base is in position
Srv:  /navigate_and_pick     — (Trigger) trigger nav+pick for closest detected object
Action client: navigate_to_pose (Nav2)
"""

import json
import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PointStamped

try:
    from tf2_ros import Buffer, TransformListener, LookupException, \
        ConnectivityException, ExtrapolationException
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

try:
    from tf2_geometry_msgs import do_transform_point
    TF2_GEOM_AVAILABLE = True
except ImportError:
    TF2_GEOM_AVAILABLE = False

try:
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False

# How close the base needs to be before we attempt a pick (metres, in map frame)
PICK_RADIUS   = 0.7
# Stand this far from the object so the arm can reach it
APPROACH_DIST = 0.5
# TF lookup timeout (seconds)
TF_TIMEOUT    = 0.5


class NavigateAndPickNode(Node):
    def __init__(self):
        super().__init__('navigate_and_pick')

        self.cb_group = ReentrantCallbackGroup()

        self._object_map: dict        = {}
        self._pending:    dict | None = None
        self._nav_active: bool        = False

        self.create_subscription(String, '/vla/object_map',    self._map_cb,      10)
        self.create_subscription(String, '/vla/task_feedback', self._feedback_cb, 10)

        self._instr_pub = self.create_publisher(String, '/vla_instruction', 10)

        self.create_service(Trigger, '/navigate_and_pick', self._nav_pick_cb,
                            callback_group=self.cb_group)

        # TF2 buffer for camera_link → map transforms
        if TF2_AVAILABLE:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer = None
            self.get_logger().warn('tf2_ros not found — frame transforms disabled.')

        if not TF2_GEOM_AVAILABLE:
            self.get_logger().warn(
                'tf2_geometry_msgs not found — install it for proper frame transforms.'
            )

        if NAV2_AVAILABLE:
            self._nav_client = ActionClient(
                self, NavigateToPose, 'navigate_to_pose',
                callback_group=self.cb_group
            )
        else:
            self._nav_client = None
            self.get_logger().warn('nav2_msgs not found — navigation disabled.')

        self.get_logger().info(
            f'Navigate-and-Pick ready. '
            f'TF2={TF2_AVAILABLE and TF2_GEOM_AVAILABLE}, Nav2={NAV2_AVAILABLE}\n'
            '  Trigger: ros2 service call /navigate_and_pick std_srvs/srv/Trigger "{}"'
        )

    # ------------------------------------------------------------------
    def _map_cb(self, msg: String):
        try:
            self._object_map = json.loads(msg.data)
        except Exception:
            pass

    def _feedback_cb(self, msg: String):
        try:
            fb = json.loads(msg.data)
        except Exception:
            return
        if fb.get('status') == 'completed' and self._pending:
            self.get_logger().info('[Nav&Pick] Pick completed.')
            self._pending    = None
            self._nav_active = False

    # ------------------------------------------------------------------
    def _to_map_frame(self, cam_x: float, cam_y: float, cam_z: float) \
            -> tuple[float, float] | None:
        """
        Transform a point in camera_link frame to (map_x, map_y).
        Returns None if TF2 is unavailable or the transform fails.
        """
        if not TF2_AVAILABLE or not TF2_GEOM_AVAILABLE or self._tf_buffer is None:
            return None

        pt              = PointStamped()
        pt.header.frame_id    = 'camera_link'
        pt.header.stamp       = rclpy.time.Time().to_msg()  # latest available
        pt.point.x            = cam_x
        pt.point.y            = cam_y
        pt.point.z            = cam_z

        try:
            transform = self._tf_buffer.lookup_transform(
                'map', 'camera_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=TF_TIMEOUT)
            )
            pt_map = do_transform_point(pt, transform)
            return pt_map.point.x, pt_map.point.y
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'[Nav&Pick] TF2 lookup failed: {e}')
            return None

    def _robot_map_pos(self) -> tuple[float, float] | None:
        """Get robot position in map frame via TF2 (base_link → map)."""
        if not TF2_AVAILABLE or self._tf_buffer is None:
            return None
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=TF_TIMEOUT)
            )
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def _dist_map(self, map_x: float, map_y: float) -> float:
        robot_pos = self._robot_map_pos()
        if robot_pos is None:
            return float('inf')
        return math.hypot(map_x - robot_pos[0], map_y - robot_pos[1])

    def _approach_pose(self, obj_x: float, obj_y: float) -> tuple[float, float]:
        """APPROACH_DIST metres behind the object, facing it from the robot's side."""
        robot_pos = self._robot_map_pos() or (0.0, 0.0)
        dx = obj_x - robot_pos[0]
        dy = obj_y - robot_pos[1]
        dist = math.hypot(dx, dy) or 1.0
        return (obj_x - dx / dist * APPROACH_DIST,
                obj_y - dy / dist * APPROACH_DIST)

    # ------------------------------------------------------------------
    def _nav_pick_cb(self, _request, response):
        if self._nav_active:
            response.success = False
            response.message = 'Already navigating.'
            return response

        if not self._object_map:
            response.success = False
            response.message = 'Object map is empty. No objects detected yet.'
            return response

        # Resolve each object's map-frame position
        candidates = []
        for label, entry in self._object_map.items():
            map_pos = self._to_map_frame(entry['x'], entry['y'], entry['z'])
            if map_pos is None:
                # TF2 unavailable — fall back to camera-frame z as a rough distance proxy
                # (z ≈ depth in metres from camera, adequate for closest-object selection)
                candidates.append((label, entry, None, entry['z']))
            else:
                candidates.append((label, entry, map_pos,
                                   self._dist_map(map_pos[0], map_pos[1])))

        # Pick the closest object by distance
        target_label, target_entry, target_map_pos, dist = min(
            candidates, key=lambda c: c[3]
        )

        self.get_logger().info(
            f'[Nav&Pick] Target: "{target_label}" | '
            f'cam=({target_entry["x"]:.2f}, {target_entry["y"]:.2f}, {target_entry["z"]:.2f}) | '
            f'dist≈{dist:.2f}m'
        )

        # Already within reach — skip navigation
        if dist <= PICK_RADIUS:
            self.get_logger().info('[Nav&Pick] Already in reach. Triggering pick directly.')
            self._trigger_pick(target_label)
            response.success = True
            response.message = f'Picking {target_label} from current position.'
            return response

        # Navigate if we have a map-frame position and Nav2
        if target_map_pos is not None \
                and NAV2_AVAILABLE \
                and self._nav_client is not None \
                and self._nav_client.server_is_ready():
            goal_x, goal_y = self._approach_pose(target_map_pos[0], target_map_pos[1])
            self._pending    = {'label': target_label}
            self._nav_active = True
            self._send_nav_goal(goal_x, goal_y, target_label)
            response.success = True
            response.message = f'Navigating to {target_label} at map ({goal_x:.2f}, {goal_y:.2f}).'
        else:
            reason = 'TF2 unavailable' if target_map_pos is None else 'Nav2 unavailable'
            self.get_logger().warn(
                f'[Nav&Pick] {reason} — triggering pick in place.'
            )
            self._trigger_pick(target_label)
            response.success = True
            response.message = f'Picking {target_label} without navigation ({reason}).'

        return response

    # ------------------------------------------------------------------
    def _send_nav_goal(self, goal_x: float, goal_y: float, label: str):
        goal_msg                                  = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id             = 'map'
        goal_msg.pose.header.stamp                = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x             = goal_x
        goal_msg.pose.pose.position.y             = goal_y
        goal_msg.pose.pose.orientation.w          = 1.0

        self.get_logger().info(f'[Nav&Pick] Nav2 goal → ({goal_x:.2f}, {goal_y:.2f})')
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
        self.get_logger().info(f'[Nav&Pick] → /vla_instruction: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    node = NavigateAndPickNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
