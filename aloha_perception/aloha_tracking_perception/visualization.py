#!/usr/bin/env python3
"""Optional viewer. ROS callback receives only; GUI runs on the main thread."""
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class LatestFrameViewer(Node):
    def __init__(self):
        super().__init__('aloha_tracking_viewer')
        self.bridge = CvBridge()
        self.frame = None
        self.create_subscription(Image, '/aloha/perception/debug_image',
                                 self._receive, qos_profile_sensor_data)
        self.get_logger().info('Viewer subscribed to /aloha/perception/debug_image')

    def _receive(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn('Display conversion: %s' % exc, throttle_duration_sec=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = LatestFrameViewer()
    try:
        cv2.namedWindow('ALOHA Tracking', cv2.WINDOW_NORMAL)
        while rclpy.ok():
            # Process subscription callbacks; GUI stays on the main thread.
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.frame is not None:
                cv2.imshow('ALOHA Tracking', node.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('ALOHA Tracking', cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
