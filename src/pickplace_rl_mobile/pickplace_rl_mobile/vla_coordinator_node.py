#!/usr/bin/env python3
"""
VLA Phase 4 - Coordinator Node
Orchestrates the full pipeline:
  Language → Task Planner → Object Memory → Action Node → Feedback loop

Topic interfaces
----------------
Sub: /vla/current_task         (String JSON)  — current atomic task from task planner
Sub: /vla/object_map           (String JSON)  — persistent object memory
Sub: /vla/detected_object_pose (PoseStamped)  — live tracked pose from vision node
Pub: /vla/action_target        (String JSON)  — pick/place poses for action node
Pub: /vla/task_feedback        (String JSON)  — completion status back to task planner
Srv: /execute_vla_sequence     (Trigger)      — triggers action node execution
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped


class VLACoordinatorNode(Node):
    def __init__(self):
        super().__init__('vla_coordinator_node')

        self._object_map:   dict             = {}
        self._current_task: dict             = {}
        self._live_pose:    PoseStamped|None = None

        # Subscriptions
        self.create_subscription(String,      '/vla/current_task',         self._task_cb,  10)
        self.create_subscription(String,      '/vla/object_map',           self._map_cb,   10)
        self.create_subscription(PoseStamped, '/vla/detected_object_pose', self._pose_cb,  10)

        # Publishers
        self._target_pub   = self.create_publisher(String, '/vla/action_target',   10)
        self._feedback_pub = self.create_publisher(String, '/vla/task_feedback',   10)

        # Service client to execute action
        self._action_client = self.create_client(Trigger, '/execute_vla_sequence')

        self.get_logger().info(
            'VLA Coordinator ready.\n'
            '  Input:  ros2 topic pub /vla_instruction std_msgs/String "data: \'pick blue cube\'"'
        )

    # ------------------------------------------------------------------
    def _map_cb(self, msg: String):
        try:
            self._object_map = json.loads(msg.data)
        except Exception:
            pass

    def _pose_cb(self, msg: PoseStamped):
        self._live_pose = msg

    def _task_cb(self, msg: String):
        try:
            task = json.loads(msg.data)
        except Exception:
            return

        # Ignore duplicate dispatches of the same task
        if task == self._current_task:
            return
        self._current_task = task

        color = task.get('color')
        dest  = task.get('destination', 'tray')
        action = task.get('action', 'pick_and_place')

        self.get_logger().info(
            f'[Coordinator] Task received: action={action} color={color} dest={dest}'
        )

        # Resolve pick pose from object memory or live tracked pose
        pick_xyz = self._resolve_object(color)
        if pick_xyz is None:
            self.get_logger().warn(
                f'[Coordinator] Object "{color}" not in memory or vision. Waiting...'
            )
            return

        # Resolve place pose from task
        place_xyz = task.get('place_xyz', {'x': 0.6, 'y': 0.0, 'z': 0.1, 'frame': 'base_link'})

        # Publish poses to action node
        target_msg      = String()
        target_msg.data = json.dumps({'pick_pose': pick_xyz, 'place_pose': place_xyz})
        self._target_pub.publish(target_msg)

        self.get_logger().info(
            f'[Coordinator] Pick=({pick_xyz["x"]:.3f},{pick_xyz["y"]:.3f},{pick_xyz["z"]:.3f}) '
            f'Place=({place_xyz["x"]:.3f},{place_xyz["y"]:.3f},{place_xyz["z"]:.3f})'
        )

        # Call action node (non-blocking service check to avoid holding up spin)
        if not self._action_client.service_is_ready():
            self.get_logger().error('[Coordinator] Action service not ready.')
            self._publish_feedback('failed')
            return

        req    = Trigger.Request()
        future = self._action_client.call_async(req)
        future.add_done_callback(self._action_done_cb)

    # ------------------------------------------------------------------
    def _resolve_object(self, color: str | None) -> dict | None:
        """Look up pick pose from memory, then fall back to live tracked pose."""
        if color and color in self._object_map:
            entry = self._object_map[color]
            return {'x': entry['x'], 'y': entry['y'], 'z': entry['z'], 'frame': 'camera_link'}

        # Try any partial colour match in memory keys
        if color:
            for label, entry in self._object_map.items():
                if color in label:
                    return {'x': entry['x'], 'y': entry['y'], 'z': entry['z'],
                            'frame': 'camera_link'}

        # Fall back to live tracked pose
        if self._live_pose:
            p = self._live_pose.pose.position
            return {'x': p.x, 'y': p.y, 'z': p.z,
                    'frame': self._live_pose.header.frame_id}
        return None

    def _action_done_cb(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'[Coordinator] Action success: {result.message}')
                self._publish_feedback('completed')
            else:
                self.get_logger().warn(f'[Coordinator] Action failed: {result.message}')
                self._publish_feedback('failed')
        except Exception as e:
            self.get_logger().error(f'[Coordinator] Service exception: {e}')
            self._publish_feedback('failed')

    def _publish_feedback(self, status: str):
        msg      = String()
        msg.data = json.dumps({'status': status, 'task': self._current_task})
        self._feedback_pub.publish(msg)
        self._current_task = {}  # Reset so next dispatch is accepted


def main(args=None):
    rclpy.init(args=args)
    node = VLACoordinatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
