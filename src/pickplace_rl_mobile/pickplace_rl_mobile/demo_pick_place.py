#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time


_MIMIC_MULTIPLIERS = {
    '/left_inner_knuckle_joint/cmd_vel': 1.0,
    '/left_inner_finger_joint/cmd_vel': -1.0,
    '/right_outer_knuckle_joint/cmd_vel': -1.0,
    '/right_inner_knuckle_joint/cmd_vel': 1.0,
    '/right_inner_finger_joint/cmd_vel': -1.0,
}


class PickPlaceDemo(Node):
    def __init__(self):
        super().__init__('pick_place_demo')

        # Publishers follow the current mobile_ur3 joint/controller topics.
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

        # Give publishers time to connect
        time.sleep(1.0)
        self.get_logger().info('Starting pick and place sequence...')

    def move_joint(self, pub, velocity, duration):
        msg = Float64()
        msg.data = velocity

        # Send velocity commands for the duration
        start_time = time.time()
        while time.time() - start_time < duration:
            pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        # Stop joint
        msg.data = 0.0
        pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.1)

    def move_gripper(self, velocity, duration):
        msg = Float64()
        msg.data = velocity
        start_time = time.time()
        while time.time() - start_time < duration:
            self.finger_pub.publish(msg)
            for pub, multiplier in self.mimic_pubs:
                pub.publish(Float64(data=float(velocity * multiplier)))
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        msg.data = 0.0
        self.finger_pub.publish(msg)
        for pub, _ in self.mimic_pubs:
            pub.publish(Float64(data=0.0))
        rclpy.spin_once(self, timeout_sec=0.1)

    def execute_sequence(self):
        # 1. Open gripper
        self.get_logger().info('Opening gripper...')
        self.move_gripper(0.5, 1.0)

        # 2. Lower arm towards object
        self.get_logger().info('Lowering arm...')
        self.move_joint(self.shoulder_pitch_pub, 0.5, 1.5)
        self.move_joint(self.elbow_pub, -0.3, 1.0)

        # 3. Close gripper (grasp)
        self.get_logger().info('Grasping object...')
        self.move_gripper(-0.5, 1.5)

        # 4. Lift arm with object
        self.get_logger().info('Lifting object...')
        self.move_joint(self.shoulder_pitch_pub, -0.6, 1.5)
        self.move_joint(self.elbow_pub, 0.3, 1.0)

        # 5. Rotate shoulder to target
        self.get_logger().info('Moving to target...')
        self.move_joint(self.shoulder_pub, 0.5, 2.0)

        # 6. Lower arm to target bin
        self.get_logger().info('Lowering to target...')
        self.move_joint(self.shoulder_pitch_pub, 0.4, 1.0)

        # 7. Release object
        self.get_logger().info('Releasing object...')
        self.move_gripper(0.5, 1.0)

        # 8. Return to home position
        self.get_logger().info('Returning to home...')
        self.move_joint(self.shoulder_pitch_pub, -0.4, 1.0)
        self.move_joint(self.shoulder_pub, -0.5, 2.0)

        self.get_logger().info('Sequence complete!')

def main(args=None):
    rclpy.init(args=args)
    demo = PickPlaceDemo()

    try:
        demo.execute_sequence()
    except KeyboardInterrupt:
        pass
    finally:
        demo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
