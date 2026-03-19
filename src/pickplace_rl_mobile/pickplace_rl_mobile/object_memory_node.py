#!/usr/bin/env python3
"""
VLA Object Memory Node  (NEW)
Maintains a persistent, timestamped map of all detected objects.
Survives momentary occlusions by keeping last-known position for up to DECAY_SECONDS.

Topics
------
Sub:  /vla/world_state       (std_msgs/String JSON)  — live detections from vision node
Pub:  /vla/object_map        (std_msgs/String JSON)  — full memory with timestamps
Pub:  /vla/object_summary    (std_msgs/String JSON)  — count + label list
Srv:  /vla/query_object_map  (std_srvs/Trigger)      — returns full map as JSON
"""

import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

DECAY_SECONDS = 30.0


class ObjectMemoryNode(Node):
    def __init__(self):
        super().__init__('vla_object_memory')

        self.declare_parameter('decay_seconds', DECAY_SECONDS)
        self._decay = self.get_parameter('decay_seconds').value

        # label → {x, y, z, score, first_seen, last_seen, observations}
        self._map: dict = {}

        self.create_subscription(String, '/vla/world_state', self._world_cb, 10)

        self.map_pub     = self.create_publisher(String, '/vla/object_map',     10)
        self.summary_pub = self.create_publisher(String, '/vla/object_summary', 10)

        self.create_service(Trigger, '/vla/query_object_map', self._query_cb)

        self.create_timer(1.0, self._publish)
        self.create_timer(5.0, self._decay_pass)

        self.get_logger().info(
            f'Object Memory Node ready. Decay: {self._decay:.0f}s. '
            'Query via /vla/query_object_map.'
        )

    # ------------------------------------------------------------------
    def _world_cb(self, msg: String):
        try:
            state: dict = json.loads(msg.data)
        except Exception:
            return
        now = time.time()
        for label, pos in state.items():
            if label in self._map:
                entry = self._map[label]
                entry.update({
                    'x': pos['x'], 'y': pos['y'], 'z': pos['z'],
                    'score':     pos.get('score', 0.8),
                    'last_seen': now,
                })
                entry['observations'] = entry.get('observations', 0) + 1
            else:
                self._map[label] = {
                    'x': pos['x'], 'y': pos['y'], 'z': pos['z'],
                    'score':        pos.get('score', 0.8),
                    'first_seen':   now,
                    'last_seen':    now,
                    'observations': 1,
                }

    def _decay_pass(self):
        now   = time.time()
        stale = [k for k, v in self._map.items() if now - v['last_seen'] > self._decay]
        for k in stale:
            self.get_logger().debug(f'Evicting stale object: {k}')
            del self._map[k]

    # ------------------------------------------------------------------
    def _publish(self):
        msg      = String()
        msg.data = json.dumps(self._map)
        self.map_pub.publish(msg)

        summary  = {'count': len(self._map), 'objects': list(self._map.keys())}
        sm       = String()
        sm.data  = json.dumps(summary)
        self.summary_pub.publish(sm)

    def _query_cb(self, request, response):
        response.success = True
        response.message = json.dumps(self._map)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMemoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
