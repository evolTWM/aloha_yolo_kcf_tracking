#!/usr/bin/env python3
"""YOLO acquisition, KCF tracking and optional aligned RealSense XYZ."""
import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from ultralytics import YOLO

from .tracking import YoloKcfTracker, backproject, depth_at_target, make_kcf


class RealSenseDetection(Node):
    def __init__(self):
        super().__init__('detector')
        defaults = {
            'rgb_topic': '/cam_high/camera/color/image_raw',
            'depth_topic': '/cam_high/camera/aligned_depth_to_color/image_raw',
            'info_topic': '/cam_high/camera/color/camera_info',
            'output_topic': '/aloha/perception/detections',
            'model': 'yolov8m.pt',
            'target_classes': ['truck', 'train'],
            'confidence': 0.30,
            'search_interval': 0.5,
            'max_depth_age': 0.10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        param = lambda name: self.get_parameter(name).value
        self.confidence = float(param('confidence'))
        self.max_depth_age = float(param('max_depth_age'))
        if not np.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError('confidence must be in (0, 1]')
        if not np.isfinite(self.max_depth_age) or self.max_depth_age < 0:
            raise ValueError('max_depth_age must be finite and nonnegative')

        # Fail clearly at startup if this ROS Python cannot create KCF.
        make_kcf(cv2)
        self.tracking = YoloKcfTracker(
            lambda: make_kcf(cv2), float(param('search_interval')))
        self.model = YOLO(param('model'))
        targets = set(param('target_classes'))
        self.target_ids = [index for index, name in self.model.names.items()
                           if name in targets]
        if not self.target_ids:
            raise ValueError(f'No target classes {targets} in model {self.model.names}')
        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_msg = None
        self._logged_event = ''

        # Keep only the latest frame. KCF receives RGB independently of depth:
        # invalid/missing depth suppresses XYZ, not the box or debug image.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.rgb_sub = self.create_subscription(Image, param('rgb_topic'), self._rgb_cb, qos)
        self.depth_sub = self.create_subscription(Image, param('depth_topic'), self._depth_cb, qos)
        self.info_sub = self.create_subscription(CameraInfo, param('info_topic'), self._info_cb, qos)
        self.det_pub = self.create_publisher(String, param('output_topic'), 10)
        self.debug_pub = self.create_publisher(Image, '/aloha/perception/debug_image', qos)
        self.get_logger().info(
            f'YOLO -> KCF ready; targets={targets}. No Kalman filtering enabled.')

    def _info_cb(self, msg):
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        if np.isfinite([fx, fy, cx, cy]).all() and fx > 0 and fy > 0:
            self.camera_info = msg

    def _depth_cb(self, msg):
        self.depth_msg = msg

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _detect(self, frame):
        candidates = []
        results = self.model(frame, verbose=False, conf=self.confidence,
                             classes=self.target_ids)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id not in self.target_ids:
                    continue
                confidence = float(box.conf.item())
                if confidence < self.confidence:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                candidates.append((self.model.names[class_id], confidence,
                                   (x1, y1, x2 - x1, y2 - y1)))
        return candidates

    def _xyz(self, rgb_msg, bbox):
        info, depth = self.camera_info, self.depth_msg
        if info is None or depth is None:
            return None
        if (info.width, info.height) != (rgb_msg.width, rgb_msg.height):
            return None
        if (depth.width, depth.height) != (rgb_msg.width, rgb_msg.height):
            return None
        if not rgb_msg.header.frame_id or not (
                rgb_msg.header.frame_id == depth.header.frame_id == info.header.frame_id):
            return None
        if abs(self._stamp(rgb_msg) - self._stamp(depth)) > self.max_depth_age:
            return None
        if depth.encoding != '16UC1':
            return None
        try:
            depth_mm = self.bridge.imgmsg_to_cv2(depth, desired_encoding='16UC1')
            sample = depth_at_target(depth_mm, bbox)
        except Exception as exc:
            self.get_logger().warn(f'Depth conversion: {exc}', throttle_duration_sec=3.0)
            return None
        if sample is None:
            return None
        u, v, z = sample
        return backproject(u, v, z, (info.k[0], info.k[4], info.k[2], info.k[5]))

    def _rgb_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'RGB conversion: {exc}', throttle_duration_sec=3.0)
            return
        # Never feed a box, label or red center point back into KCF.
        target = self.tracking.step(frame, time.monotonic(), self._detect)
        display = frame.copy()
        detections = []
        if target is not None:
            x, y, w, h = target.bbox
            u, v = x + w // 2, y + h // 2
            # Acquisition is yellow for one frame; KCF updates are green.
            color = (0, 255, 0) if target.method == 'KCF' else (0, 255, 255)
            cv2.rectangle(display, (x, y), (x + w - 1, y + h - 1), color, 2)
            cv2.circle(display, (u, v), 5, (0, 0, 255), -1)
            label_y = min(max(y - 10, 65), max(65, msg.height - 45))
            self._text(display, f'{target.label} [{target.method}] Pixel: ({u}, {v})',
                       (max(5, min(x, msg.width - 300)), label_y), color)
            coords = self._xyz(msg, target.bbox)
            if coords is not None:
                cx3, cy3, cz3 = [float(value) for value in coords]
                self._text(display, f'XYZ: ({cx3:.3f}, {cy3:.3f}, {cz3:.3f}) m',
                           (5, msg.height - 15), color)
                detections.append({
                    'label': target.label,
                    # This is the acquisition score, NOT a new KCF confidence.
                    'confidence': round(target.detection_confidence, 3),
                    'confidence_source': 'last_yolo_detection',
                    'tracking_method': target.method,
                    'u': u, 'v': v,
                    'cx_3d': round(cx3, 4), 'cy_3d': round(cy3, 4), 'cz_3d': round(cz3, 4),
                    'frame_id': msg.header.frame_id,
                    'stamp': self._stamp(msg), 'source': 'realsense',
                })
            else:
                self._text(display, 'XYZ unavailable (depth/info missing, invalid or stale)',
                           (5, msg.height - 15), (0, 200, 255))

        status = (f'{self.tracking.state}  YOLO calls: {self.tracking.yolo_calls}'
                  f'  KCF updates: {self.tracking.kcf_updates}')
        self._text(display, status, (5, 22), (255, 255, 255))
        if target is None:
            self._text(display, 'Searching for truck/train...', (5, 45), (0, 200, 255))
        event = self.tracking.last_event
        if event and event != self._logged_event:
            self.get_logger().info(event)
            self._logged_event = event
        self.det_pub.publish(String(data=json.dumps(detections)))
        preview = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
        preview.header = msg.header
        self.debug_pub.publish(preview)

    @staticmethod
    def _text(display, text, position, color):
        cv2.putText(display, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealSenseDetection()
        # Single-threaded executor serializes KCF state and latest depth/info.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
