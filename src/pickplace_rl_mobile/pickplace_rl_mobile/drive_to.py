#!/usr/bin/env python3
"""Drive the base forward until odometry reaches --x, then stop and exit.

The whole-body node picks best with the base parked, so wbc_pickplace.launch.py drive:=true
uses this to bring the robot up to the cube first.

  python3 -m pickplace_rl_mobile.drive_to --x 0.78
"""

import argparse

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def drive_speed(err, speed, tol, min_speed, allow_reverse=False):
    """Speed for the remaining signed distance: zero inside the tolerance, a proportional
    approach that never drops below the creep speed, capped at the cruise speed. Driving
    backwards (negative err) only happens when allow_reverse is set."""
    if abs(err) <= tol or (err < 0 and not allow_reverse):
        return 0.0
    mag = min(speed, max(min_speed, 1.5 * abs(err)))
    return mag if err > 0 else -mag


class DriveTo(Node):
    def __init__(self, target, speed, tol, min_speed=0.08, allow_reverse=False):
        super().__init__('drive_to')
        self.target, self.speed, self.tol, self.min_speed = target, speed, tol, min_speed
        self.allow_reverse = allow_reverse
        self.x = None
        self.stopped = 0
        self.done = False
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.05, self._tick)

    def _on_odom(self, msg):
        self.x = msg.pose.pose.position.x

    def _tick(self):
        if self.x is None:
            return
        err = self.target - self.x
        cmd = Twist()
        if abs(err) > self.tol and (err > 0 or self.allow_reverse):
            cmd.linear.x = drive_speed(err, self.speed, self.tol, self.min_speed, self.allow_reverse)
        else:
            self.stopped += 1
            if self.stopped == 1:
                self.get_logger().info(f'reached odom x={self.x:.3f}')
            if self.stopped >= 30:  # ~1.5 s of zero command so the base settles
                self.done = True
        self.pub.publish(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--x', type=float, default=0.78, help='odom x to stop at')
    p.add_argument('--speed', type=float, default=0.3)
    p.add_argument('--tol', type=float, default=0.01)
    p.add_argument('--allow-reverse', action='store_true', help='also drive backwards to reach --x (return to start)')
    p.add_argument('--min-speed', type=float, default=0.08, help='creep speed near the target')
    args, _ = p.parse_known_args()
    rclpy.init()
    node = DriveTo(args.x, args.speed, args.tol, args.min_speed, args.allow_reverse)
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
