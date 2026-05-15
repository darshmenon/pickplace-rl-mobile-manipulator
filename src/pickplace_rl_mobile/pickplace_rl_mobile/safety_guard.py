#!/usr/bin/env python3
"""
Safety Guard Node for the Pick-and-Place Mobile Manipulator.

Monitors robot state and sensor data to enforce safety constraints:
- Joint position limits
- End-effector workspace boundaries
- LiDAR-based obstacle proximity
- Emergency stop capability
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json


class SafetyGuard(Node):
    def __init__(self):
        super().__init__('safety_guard')

        # --- Parameters ---
        self.declare_parameter('joint_limit_margin', 0.1)  # radians from limit
        self.declare_parameter('min_obstacle_distance', 0.25)  # meters
        self.declare_parameter('ee_min_height', 0.02)  # meters
        self.declare_parameter('ee_max_reach', 0.75)  # meters from base
        self.declare_parameter('workspace_radius', 3.0)  # meters from origin
        self.declare_parameter('monitor_rate', 20.0)  # Hz

        self.joint_limit_margin = self.get_parameter('joint_limit_margin').value
        self.min_obstacle_dist = self.get_parameter('min_obstacle_distance').value
        self.ee_min_height = self.get_parameter('ee_min_height').value
        self.ee_max_reach = self.get_parameter('ee_max_reach').value
        self.workspace_radius = self.get_parameter('workspace_radius').value

        # UR3 joint limits matching pickplace_env.py _UR3_JOINT_LOW/HIGH
        self.joint_limits = {
            'shoulder_pan_joint':  [-6.2832, 6.2832],
            'shoulder_lift_joint': [-6.2832, 6.2832],
            'elbow_joint':         [-3.1416, 3.1416],
            'wrist_1_joint':       [-6.2832, 6.2832],
            'wrist_2_joint':       [-6.2832, 6.2832],
            'wrist_3_joint':       [-6.2832, 6.2832],
            'finger_joint':        [0.0, 0.8],
        }

        # UR3 DH-derived kinematics constants (matching pickplace_env.py)
        self.base_height = 0.08   # chassis spawn z
        self.arm_mount_z = 0.10   # arm base above chassis

        # State
        self.joint_positions = {}
        self.joint_names = []
        self.base_pose = np.zeros(3)
        self.min_lidar_distance = float('inf')
        self.e_stop_active = False
        self.violations = []

        # --- Subscribers ---
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # --- Publishers ---
        self.status_pub = self.create_publisher(String, '/safety/status', 10)
        self.estop_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Monitor timer
        rate = self.get_parameter('monitor_rate').value
        self.timer = self.create_timer(1.0 / rate, self.monitor_safety)

        self.get_logger().info('Safety Guard initialized — monitoring joint limits, workspace, obstacles')

    def joint_callback(self, msg):
        """Update joint positions."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
        self.joint_names = list(msg.name)

    def odom_callback(self, msg):
        """Update base pose."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)
        self.base_pose = np.array([x, y, theta])

    def scan_callback(self, msg):
        """Update minimum LiDAR distance."""
        ranges = np.array(msg.ranges)
        valid = ranges[(ranges > msg.range_min) & (ranges < msg.range_max)]
        if len(valid) > 0:
            self.min_lidar_distance = float(np.min(valid))
        else:
            self.min_lidar_distance = float('inf')

    def compute_ee_position(self):
        """EE position in world frame using UR3 DH FK (matches pickplace_env.py)."""
        _DH = [
            (0.0,      0.1519,  np.pi / 2),
            (-0.24365, 0.0,     0.0),
            (-0.21325, 0.0,     0.0),
            (0.0,      0.11235, np.pi / 2),
            (0.0,      0.08535, -np.pi / 2),
            (0.0,      0.0819,  0.0),
        ]
        joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]
        angles = np.array([self.joint_positions.get(n, 0.0) for n in joint_names])

        T = np.eye(4)
        for i, (a, d, alpha) in enumerate(_DH):
            ct, st = np.cos(angles[i]), np.sin(angles[i])
            ca, sa = np.cos(alpha), np.sin(alpha)
            T = T @ np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [0.,       sa,      ca,       d],
                [0.,       0.,      0.,      1.],
            ])
        ee_arm = T[:3, 3]
        # 180° yaw on base_link_inertia: flip x,y then add mount offset
        ee_local = np.array([-ee_arm[0], -ee_arm[1], ee_arm[2]]) + np.array([0., 0., 0.10])
        # World frame via odometry yaw
        bx, by, btheta = self.base_pose
        return np.array([
            bx + ee_local[0] * np.cos(btheta) - ee_local[1] * np.sin(btheta),
            by + ee_local[0] * np.sin(btheta) + ee_local[1] * np.cos(btheta),
            self.base_height + ee_local[2],
        ])

    def check_joint_limits(self):
        """Check if any joints are near their limits."""
        violations = []
        for name, limits in self.joint_limits.items():
            pos = self.joint_positions.get(name, 0.0)
            lower, upper = limits
            margin = self.joint_limit_margin

            if pos <= lower + margin:
                violations.append(f'{name} near lower limit ({pos:.3f} <= {lower + margin:.3f})')
            elif pos >= upper - margin:
                violations.append(f'{name} near upper limit ({pos:.3f} >= {upper - margin:.3f})')
        return violations

    def check_workspace_bounds(self):
        """Check end-effector and base workspace limits."""
        violations = []

        # Check base position is within workspace
        base_dist = np.sqrt(self.base_pose[0]**2 + self.base_pose[1]**2)
        if base_dist > self.workspace_radius:
            violations.append(
                f'Base outside workspace (dist={base_dist:.2f}m > {self.workspace_radius}m)')

        # Check EE position
        ee_pos = self.compute_ee_position()
        if ee_pos[2] < self.ee_min_height:
            violations.append(
                f'EE too low (z={ee_pos[2]:.3f}m < {self.ee_min_height}m)')

        ee_reach = np.sqrt(ee_pos[0]**2 + ee_pos[1]**2)
        if ee_reach > self.ee_max_reach:
            violations.append(
                f'EE overextended (reach={ee_reach:.2f}m > {self.ee_max_reach}m)')

        return violations

    def check_obstacle_proximity(self):
        """Check LiDAR for nearby obstacles."""
        violations = []
        if self.min_lidar_distance < self.min_obstacle_dist:
            violations.append(
                f'Obstacle too close (dist={self.min_lidar_distance:.2f}m < {self.min_obstacle_dist}m)')
        return violations

    def emergency_stop(self):
        """Send zero velocity on all axes."""
        if not self.e_stop_active:
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            self.e_stop_active = True

        stop_msg = Twist()
        self.estop_cmd_pub.publish(stop_msg)

    def monitor_safety(self):
        """Main safety monitoring loop."""
        if not self.joint_positions:
            return  # No data yet

        self.violations = []

        # Run all checks
        self.violations.extend(self.check_joint_limits())
        self.violations.extend(self.check_workspace_bounds())
        self.violations.extend(self.check_obstacle_proximity())

        # Determine safety status
        if self.violations:
            severity = 'WARNING'
            # Critical violations trigger e-stop
            critical_keywords = ['Obstacle too close', 'EE too low']
            is_critical = any(
                kw in v for v in self.violations for kw in critical_keywords)

            if is_critical:
                severity = 'CRITICAL'
                self.emergency_stop()
            else:
                self.e_stop_active = False
        else:
            severity = 'OK'
            self.e_stop_active = False

        # Publish status
        status = {
            'severity': severity,
            'violations': self.violations,
            'ee_position': self.compute_ee_position().tolist(),
            'min_obstacle_dist': round(self.min_lidar_distance, 3),
            'e_stop': self.e_stop_active
        }

        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)

        if self.violations:
            self.get_logger().warn(
                f'Safety [{severity}]: {"; ".join(self.violations)}')


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
