#!/usr/bin/env python3
"""
VLA Phase 2 - Vision Node
Upgraded: OWLv2 open-vocabulary detection + HSV color fallback.
Accepts full text queries (e.g. "red mug", "blue cube") via /vla_track_object.
"""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np

try:
    import cv2
    from cv_bridge import CvBridge
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# HSV fallback colour ranges
COLOR_RANGES = {
    'red':    ([0,   120,  70],  [10,  255, 255]),
    'blue':   ([100, 150,  50],  [140, 255, 255]),
    'green':  ([40,   50,  50],  [80,  255, 255]),
    'yellow': ([20,  100, 100],  [30,  255, 255]),
    'orange': ([10,  120,  70],  [20,  255, 255]),
    'purple': ([130,  50,  50],  [160, 255, 255]),
}

_owlv2_model = None
_owlv2_processor = None
_USE_OWLV2 = False


def _load_owlv2() -> bool:
    global _owlv2_model, _owlv2_processor, _USE_OWLV2
    try:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        model_id = 'google/owlv2-base-patch16-finetuned'
        _owlv2_processor = Owlv2Processor.from_pretrained(model_id)
        _owlv2_model = Owlv2ForObjectDetection.from_pretrained(model_id)
        _owlv2_model.eval()
        _USE_OWLV2 = True
        return True
    except Exception:
        return False


def _detect_owlv2(rgb_image: np.ndarray, queries: list[str], threshold: float = 0.1) -> list[dict]:
    """Returns list of {label, score, cx, cy} dicts."""
    if not _USE_OWLV2 or not queries:
        return []
    try:
        from PIL import Image as PILImage
        import torch
        pil_img = PILImage.fromarray(rgb_image)
        inputs = _owlv2_processor(text=queries, images=pil_img, return_tensors='pt')
        with torch.no_grad():
            outputs = _owlv2_model(**inputs)
        target_sizes = torch.tensor([pil_img.size[::-1]])
        # post_process_object_detection is the stable API across transformers versions
        results = _owlv2_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        detections = []
        for score, label_idx, box in zip(results['scores'], results['labels'], results['boxes']):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                'label': queries[label_idx],
                'score': float(score),
                'cx':    int((x1 + x2) / 2),
                'cy':    int((y1 + y2) / 2),
            })
        return detections
    except Exception:
        return []


class VLAVisionNode(Node):
    def __init__(self):
        super().__init__('vla_vision_node')

        self.declare_parameter('use_owlv2', True)
        self.declare_parameter('detection_threshold', 0.1)

        self.bridge = CvBridge() if CV2_AVAILABLE else None
        self.depth_image = None
        self.latest_rgb: np.ndarray | None = None
        self.detected_objects: dict = {}
        self.tracked_query = 'blue cube'

        use_owlv2 = self.get_parameter('use_owlv2').value
        self.threshold = self.get_parameter('detection_threshold').value

        if use_owlv2:
            self.get_logger().info('Loading OWLv2 open-vocabulary detector (first run ~60s)...')
            if _load_owlv2():
                self.get_logger().info('OWLv2 loaded. Open-vocabulary detection active.')
            else:
                self.get_logger().warn(
                    'OWLv2 unavailable (pip install transformers torch pillow). HSV fallback active.'
                )

        if CV2_AVAILABLE:
            self.create_subscription(Image, '/camera/image_raw', self._color_cb, 10)
            self.create_subscription(Image, '/camera/depth',     self._depth_cb, 10)
        self.create_subscription(String, '/vla_track_object', self._cmd_cb, 10)

        self.pose_pub  = self.create_publisher(PoseStamped, '/vla/detected_object_pose', 10)
        self.world_pub = self.create_publisher(String,      '/vla/world_state',          10)

        self.create_timer(0.5, self._publish_world_state)
        self.get_logger().info(f"VLA Vision Node ready. Tracking: '{self.tracked_query}'")

    def _cmd_cb(self, msg: String):
        self.tracked_query = msg.data.strip()
        self.get_logger().info(f"Now tracking: '{self.tracked_query}'")

    def _depth_cb(self, msg: Image):
        if CV2_AVAILABLE:
            try:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception:
                pass

    def _backproject(self, cx: int, cy: int) -> tuple[float, float, float]:
        depth_m = 0.5
        if self.depth_image is not None:
            h, w = self.depth_image.shape[:2]
            if 0 <= cy < h and 0 <= cx < w:
                raw = float(self.depth_image[cy, cx])
                if raw > 0:
                    depth_m = raw
        fx = fy = 554.0
        x = (cx - 320) * depth_m / fx
        y = (cy - 240) * depth_m / fy
        return x, y, depth_m

    def _publish_pose(self, x: float, y: float, z: float):
        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = 'camera_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

    def _color_cb(self, msg: Image):
        if not CV2_AVAILABLE:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # OWLv2 open-vocabulary path
        if _USE_OWLV2 and self.latest_rgb is not None:
            queries = list({self.tracked_query} | {f'{c} object' for c in COLOR_RANGES})
            detections = _detect_owlv2(self.latest_rgb, queries, self.threshold)
            for det in detections:
                x, y, z = self._backproject(det['cx'], det['cy'])
                self.detected_objects[det['label']] = {
                    'x': round(x, 3), 'y': round(y, 3), 'z': round(z, 3),
                    'score': round(det['score'], 3),
                }
                if det['label'] == self.tracked_query:
                    self._publish_pose(x, y, z)
            return

        # HSV fallback path
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        for color, (lower, upper) in COLOR_RANGES.items():
            lo = np.array(lower, dtype=np.uint8)
            hi = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 300:
                continue
            M = cv2.moments(largest)
            if M['m00'] <= 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, z = self._backproject(cx, cy)
            self.detected_objects[color] = {
                'x': round(x, 3), 'y': round(y, 3), 'z': round(z, 3), 'score': 0.8
            }
            if color in self.tracked_query:
                self._publish_pose(x, y, z)

    def _publish_world_state(self):
        msg = String()
        msg.data = json.dumps(self.detected_objects)
        self.world_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VLAVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
