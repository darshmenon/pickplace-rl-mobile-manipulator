#!/usr/bin/env python3
"""
VLA Task Planner Node  (NEW)
Decomposes high-level natural language goals into ordered pick-and-place task queues.

Supported patterns
------------------
- Single:   "pick the red cube and place in tray"
- Sort all: "sort all cubes by colour"
- Clear:    "clear all objects from the table"
- Stack:    "stack all blocks"
- Sequential: coordinator feeds feedback to advance the queue
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']

# Default drop positions for each colour bin
COLOR_BINS: dict[str, dict] = {
    'red':     {'x': 0.60, 'y':  0.40, 'z': 0.10, 'frame': 'base_link'},
    'blue':    {'x': 0.60, 'y': -0.40, 'z': 0.10, 'frame': 'base_link'},
    'green':   {'x': 0.70, 'y':  0.30, 'z': 0.10, 'frame': 'base_link'},
    'yellow':  {'x': 0.70, 'y': -0.30, 'z': 0.10, 'frame': 'base_link'},
    'orange':  {'x': 0.65, 'y':  0.45, 'z': 0.10, 'frame': 'base_link'},
    'purple':  {'x': 0.65, 'y': -0.45, 'z': 0.10, 'frame': 'base_link'},
    'default': {'x': 0.60, 'y':  0.00, 'z': 0.10, 'frame': 'base_link'},
}

_SORT_WORDS  = {'sort', 'organise', 'organize', 'arrange', 'group', 'separate'}
_STACK_WORDS = {'stack', 'pile', 'tower'}
_CLEAR_WORDS = {'clear', 'remove', 'clean'}
_ALL_WORDS   = {'all', 'every', 'each'}


def _has(words: set[str], text: str) -> bool:
    return bool(words & set(text.split()))


def decompose(cmd: dict, world_state: dict) -> list[dict]:
    """Return list of atomic pick-and-place tasks from a structured command."""
    raw   = cmd.get('raw', '').lower()
    color = cmd.get('color')
    dest  = cmd.get('destination', 'tray')
    tasks: list[dict] = []

    # --- Sort all detected objects into colour bins ---
    if _has(_SORT_WORDS, raw) and _has(_ALL_WORDS, raw):
        for c in COLORS:
            if c in world_state:
                tasks.append({
                    'action':      'pick_and_place',
                    'color':       c,
                    'destination': f'{c}_bin',
                    'place_xyz':   COLOR_BINS[c],
                })
        return tasks

    # --- Clear everything to the default tray ---
    if _has(_CLEAR_WORDS, raw) and _has(_ALL_WORDS, raw):
        for c in world_state:
            tasks.append({
                'action':      'pick_and_place',
                'color':       c,
                'destination': 'default_bin',
                'place_xyz':   COLOR_BINS['default'],
            })
        return tasks

    # --- Stack all visible objects at centre ---
    if _has(_STACK_WORDS, raw):
        base_z = 0.055
        for i, c in enumerate(c for c in COLORS if c in world_state):
            tasks.append({
                'action':      'pick_and_place',
                'color':       c,
                'destination': 'stack',
                'place_xyz':   {'x': 0.50, 'y': 0.00,
                                'z': round(base_z + i * 0.04, 3),
                                'frame': 'base_link'},
            })
        return tasks

    # --- Default: single pick-and-place ---
    place_xyz = COLOR_BINS.get(dest, COLOR_BINS['default'])
    if color:
        place_xyz = COLOR_BINS.get(color, COLOR_BINS.get(dest, COLOR_BINS['default']))
    tasks.append({
        'action':      cmd.get('action', 'pick_and_place'),
        'color':       color,
        'destination': dest,
        'place_xyz':   place_xyz,
    })
    return tasks


class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('vla_task_planner')

        self.world_state: dict  = {}
        self.task_queue: list   = []
        self.current_idx: int   = 0
        self.busy: bool         = False

        self.create_subscription(String, '/vla/world_state',        self._world_cb,    10)
        self.create_subscription(String, '/vla/structured_command', self._cmd_cb,      10)
        self.create_subscription(String, '/vla/task_feedback',      self._feedback_cb, 10)

        self.task_pub   = self.create_publisher(String, '/vla/current_task',   10)
        self.queue_pub  = self.create_publisher(String, '/vla/task_queue',     10)
        self.status_pub = self.create_publisher(String, '/vla/planner_status', 10)

        # Retry timer: re-send current task if coordinator hasn't acknowledged yet
        # Uses a long interval (2s) to avoid flooding — the coordinator deduplicates
        self.create_timer(2.0, self._dispatch)
        self.get_logger().info(
            'Task Planner ready. Supports: single, sort-all, clear-all, stack.'
        )

    # ------------------------------------------------------------------
    def _world_cb(self, msg: String):
        try:
            self.world_state = json.loads(msg.data)
        except Exception:
            pass

    def _cmd_cb(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except Exception:
            return
        tasks = decompose(cmd, self.world_state)
        if not tasks:
            self.get_logger().warn('Task decomposition produced no tasks.')
            return
        self.task_queue  = tasks
        self.current_idx = 0
        self.busy        = True
        self.get_logger().info(f'Task queue: {len(tasks)} task(s) loaded.')
        self._pub_queue()
        self._pub_status('running')

    def _feedback_cb(self, msg: String):
        try:
            fb = json.loads(msg.data)
        except Exception:
            return
        if fb.get('status') == 'completed':
            self.current_idx += 1
            if self.current_idx >= len(self.task_queue):
                self.busy = False
                self.get_logger().info('All tasks complete.')
                self._pub_status('idle')
            else:
                self.get_logger().info(
                    f'Advancing to task {self.current_idx + 1}/{len(self.task_queue)}.'
                )

    # ------------------------------------------------------------------
    def _dispatch(self):
        if not self.busy or not self.task_queue:
            return
        if self.current_idx >= len(self.task_queue):
            return
        task = self.task_queue[self.current_idx]
        msg       = String()
        msg.data  = json.dumps(task)
        self.task_pub.publish(msg)

    def _pub_queue(self):
        msg      = String()
        msg.data = json.dumps({'queue': self.task_queue, 'current': self.current_idx})
        self.queue_pub.publish(msg)

    def _pub_status(self, status: str):
        msg      = String()
        msg.data = json.dumps({'status': status, 'queue_length': len(self.task_queue),
                               'current': self.current_idx})
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
