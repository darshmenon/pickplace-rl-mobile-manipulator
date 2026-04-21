#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import time
import numpy as np


_MIMIC_MULTIPLIERS = {
    '/left_inner_knuckle_joint/cmd_vel': 1.0,
    '/left_inner_finger_joint/cmd_vel': -1.0,
    '/right_outer_knuckle_joint/cmd_vel': -1.0,
    '/right_inner_knuckle_joint/cmd_vel': 1.0,
    '/right_inner_finger_joint/cmd_vel': -1.0,
}


class SmartPickPlace(Node):
    def __init__(self):
        super().__init__('smart_pick_place')

        # Publishers follow the current mobile_ur3 controller topics.
        self.shoulder_pub = self.create_publisher(Float64, '/shoulder_pan_joint/cmd_vel', 10)
        self.shoulder_pitch_pub = self.create_publisher(Float64, '/shoulder_lift_joint/cmd_vel', 10)
        self.elbow_pub = self.create_publisher(Float64, '/elbow_joint/cmd_vel', 10)
        self.wrist_1_pub = self.create_publisher(Float64, '/wrist_1_joint/cmd_vel', 10)
        self.wrist_2_pub = self.create_publisher(Float64, '/wrist_2_joint/cmd_vel', 10)
        self.wrist_3_pub = self.create_publisher(Float64, '/wrist_3_joint/cmd_vel', 10)
        self.finger_pub = self.create_publisher(Float64, '/finger_joint/cmd_vel', 10)
        self.mimic_pubs = [
            (self.create_publisher(Float64, topic, 10), multiplier)
            for topic, multiplier in _MIMIC_MULTIPLIERS.items()
        ]

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.joint_positions = np.zeros(7)
        self.joint_names = []

        # Known kinematics parameters based on env logic
        self.base_height = 0.2
        self.link1_length = 0.24
        self.link2_length = 0.24
        self.link3_length = 0.20
        self.base_offset_x = 0.1

        time.sleep(1.0)
        self.get_logger().info('Smart Pick and Place Node initialized.')

    def joint_state_callback(self, msg):
        self.joint_names = msg.name
        self.joint_positions = np.array(msg.position[:7])

    def publish_velocities(self, velocities):
        self.shoulder_pub.publish(Float64(data=float(velocities[0])))
        self.shoulder_pitch_pub.publish(Float64(data=float(velocities[1])))
        self.elbow_pub.publish(Float64(data=float(velocities[2])))
        self.wrist_1_pub.publish(Float64(data=float(velocities[3])))
        self.wrist_2_pub.publish(Float64(data=float(velocities[4])))
        self.wrist_3_pub.publish(Float64(data=0.0))

    def control_gripper(self, open_gripper=True, duration=1.0):
        vel = 0.5 if open_gripper else -0.5
        msg = Float64(data=vel)
        start_time = time.time()
        while time.time() - start_time < duration:
            self.finger_pub.publish(msg)
            for pub, multiplier in self.mimic_pubs:
                pub.publish(Float64(data=float(vel * multiplier)))
            time.sleep(0.05)

        msg.data = 0.0
        self.finger_pub.publish(msg)
        for pub, _ in self.mimic_pubs:
            pub.publish(Float64(data=0.0))

    def compute_forward_kinematics(self, q):
        shoulder_angle, shoulder_pitch, elbow = q[0], q[1], q[2]

        z = self.base_height + self.link1_length
        x_local = self.link2_length * np.cos(shoulder_pitch) + self.link3_length * np.cos(shoulder_pitch + elbow)
        z_local = self.link2_length * np.sin(shoulder_pitch) + self.link3_length * np.sin(shoulder_pitch + elbow)

        x = self.base_offset_x + x_local * np.cos(shoulder_angle)
        y = x_local * np.sin(shoulder_angle)
        z = z + z_local

        return np.array([x, y, z])

    def compute_jacobian(self, q):
        shoulder_angle, shoulder_pitch, elbow = q[0], q[1], q[2]

        x_local = self.link2_length * np.cos(shoulder_pitch) + self.link3_length * np.cos(shoulder_pitch + elbow)

        dx_dshoulder = -x_local * np.sin(shoulder_angle)
        dy_dshoulder = x_local * np.cos(shoulder_angle)
        dz_dshoulder = 0.0

        dx_local_dpitch = -self.link2_length * np.sin(shoulder_pitch) - self.link3_length * np.sin(shoulder_pitch + elbow)
        dz_local_dpitch = self.link2_length * np.cos(shoulder_pitch) + self.link3_length * np.cos(shoulder_pitch + elbow)

        dx_dpitch = dx_local_dpitch * np.cos(shoulder_angle)
        dy_dpitch = dx_local_dpitch * np.sin(shoulder_angle)
        dz_dpitch = dz_local_dpitch

        dx_local_delbow = -self.link3_length * np.sin(shoulder_pitch + elbow)
        dz_local_delbow = self.link3_length * np.cos(shoulder_pitch + elbow)

        dx_delbow = dx_local_delbow * np.cos(shoulder_angle)
        dy_delbow = dx_local_delbow * np.sin(shoulder_angle)
        dz_delbow = dz_local_delbow

        J = np.array([
            [dx_dshoulder, dx_dpitch, dx_delbow],
            [dy_dshoulder, dy_dpitch, dy_delbow],
            [dz_dshoulder, dz_dpitch, dz_delbow]
        ])
        return J

    def move_to_target(self, target_pos, tolerance=0.03, max_steps=300):
        # Very simple inverse kinematics loop using Jacobian transpose mapping
        rate = self.create_rate(20)
        gain = 1.0

        for step in range(max_steps):
            current_q = self.joint_positions[:3]
            current_pos = self.compute_forward_kinematics(current_q)

            error = target_pos - current_pos
            if np.linalg.norm(error) < tolerance:
                # Reached within tolerance
                self.publish_velocities(np.zeros(5))
                return True

            J = self.compute_jacobian(current_q)

            # Dampened pseudoinverse or simply transpose for safety
            J_pinv = np.linalg.pinv(J)
            q_dot = gain * J_pinv.dot(error)

            # Limit velocities
            max_vel = 0.5
            q_dot = np.clip(q_dot, -max_vel, max_vel)

            vels = np.zeros(5)
            vels[0] = q_dot[0]
            vels[1] = q_dot[1]
            vels[2] = q_dot[2]

            self.publish_velocities(vels)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

        self.publish_velocities(np.zeros(5))
        return False

    def execute_sequence(self, object_pos, target_drop_pos):
        self.get_logger().info('Opening Gripper...')
        self.control_gripper(open_gripper=True, duration=1.0)

        # Add offset for grasping
        approach_pos = np.copy(object_pos)
        approach_pos[2] += 0.15 # Move above the object first
        grasp_pos = np.copy(object_pos)
        grasp_pos[2] += 0.05 # Lower slightly to grip

        self.get_logger().info(f'Approaching object above at {approach_pos}')
        success = self.move_to_target(approach_pos)

        if success:
            self.get_logger().info('Lowering to grasp...')
            self.move_to_target(grasp_pos, tolerance=0.02)

            self.get_logger().info('Closing Gripper...')
            self.control_gripper(open_gripper=False, duration=1.5)

            self.get_logger().info('Lifting object...')
            self.move_to_target(approach_pos)

            self.get_logger().info(f'Moving to drop target {target_drop_pos}...')
            drop_approach = np.copy(target_drop_pos)
            drop_approach[2] += 0.15
            self.move_to_target(drop_approach)

            self.get_logger().info('Lowering to place...')
            self.move_to_target(target_drop_pos, tolerance=0.03)

            self.get_logger().info('Releasing object...')
            self.control_gripper(open_gripper=True, duration=1.0)

            self.get_logger().info('Returning to home...')
            home_pos = np.array([0.4, 0.0, 0.4])
            self.move_to_target(home_pos)

            self.get_logger().info('Sequence complete!')
        else:
            self.get_logger().error('Failed to approach object.')

def main(args=None):
    rclpy.init(args=args)
    demo = SmartPickPlace()

    # We can randomize the object start position around [0.6, 0.0]
    random_x = 0.5 + np.random.rand() * 0.2 # 0.5 to 0.7
    random_y = -0.2 + np.random.rand() * 0.4 # -0.2 to 0.2
    obj_pos = np.array([random_x, random_y, 0.055])

    target_pos = np.array([0.6, 0.5, 0.1])

    try:
        demo.execute_sequence(object_pos=obj_pos, target_drop_pos=target_pos)
    except KeyboardInterrupt:
        pass
    finally:
        demo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
