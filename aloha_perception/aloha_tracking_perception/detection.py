#!/usr/bin/env python3
"""
ALOHA overhead-camera object detector
Uses a supplied YOLO model and aligned RealSense Depth
Publishes labeled camera-relative XYZ coordinates in meters.
"""
import os
import time
import rclpy, threading, json
import message_filters
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, numpy as np
import torch

if __package__:
    from .workspace_overlay import WorkspaceOverlay
else:
    from workspace_overlay import WorkspaceOverlay

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class RealSenseDetection(Node):
    def __init__(self):
        super().__init__('detector')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('model', 'yolov8m.pt')
        configured_device = self.get_parameter('device').value
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if configured_device == 'auto' else configured_device
        self.get_logger().info(f"YOLO device: {self.device}")
        self.bridge  = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_info = None
        self.depth = None
        self._lock = threading.Lock()
        self._has_display = bool(os.environ.get('DISPLAY'))

        # YOLO
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics not available.")

        model_path = os.path.expanduser(self.get_parameter('model').value)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.get_logger().info(f"Loaded {model_path} with classes: {self.model.names}")

        self.target_classes = ['truck', 'train', 'car']

        # Subscriptions
        self.declare_parameter('rgb_topic', '/cam_high/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/cam_high/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/cam_high/camera/color/camera_info')

        self.rgb_sub = message_filters.Subscriber(self, Image, self.get_parameter('rgb_topic').value, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.get_parameter('depth_topic').value, qos_profile=qos_profile_sensor_data)
        self.image_sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.03)

        self.image_sync.registerCallback(self._images_cb)

        self.create_subscription(CameraInfo, self.get_parameter('info_topic').value, self._info_cb, qos_profile_sensor_data)

        self.declare_parameter('output_topic', '/aloha/perception/detections')
        self.det_pub = self.create_publisher(String, self.get_parameter('output_topic').value, 10)
        self.debug_pub = self.create_publisher(CompressedImage, '/aloha/perception/debug_image/compressed', qos_profile_sensor_data)
        self.debug_raw_pub = self.create_publisher(Image, '/aloha/perception/debug_image', 1)
        self.declare_parameter('workspace_topic', '/aloha/perception/detections_workspace_filtered')
        self.declare_parameter('workspace_frame', 'table_workspace')
        self.workspace_overlay = WorkspaceOverlay(frame=self.get_parameter('workspace_frame').value)
        self.last_workspace_overlay = None
        self.workspace_sub = self.create_subscription(
            String, self.get_parameter('workspace_topic').value, self._workspace_cb, 10
        )
        self._frame_count = 0
        self.get_logger().info('RealSense detection node ready')

    def _has_debug_viewer(self):
        return (self.debug_raw_pub.get_subscription_count() > 0
                or self.debug_pub.get_subscription_count() > 0)

    def _publish_debug(self, image, header):
        if self.debug_raw_pub.get_subscription_count() > 0:
            image = np.ascontiguousarray(image, dtype=np.uint8)
            raw = Image()
            raw.header = header
            raw.height, raw.width = image.shape[:2]
            raw.encoding = 'bgr8'
            raw.is_bigendian = 0
            raw.step = raw.width * 3
            raw.data = image.tobytes()
            self.debug_raw_pub.publish(raw)
        if self.debug_pub.get_subscription_count() > 0:
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                preview = CompressedImage()
                preview.header = header
                preview.format = 'jpeg'
                preview.data = encoded.tobytes()
                self.debug_pub.publish(preview)

    def _workspace_cb(self, msg):
        if not self._has_debug_viewer():
            self.workspace_overlay.clear()
            return
        try:
            detections = json.loads(msg.data)
        except (ValueError, TypeError):
            self.workspace_overlay.clear()
            return
        rendered = self.workspace_overlay.render(
            detections, self.get_clock().now().nanoseconds / 1e9
        )
        if rendered is not None:
            image, header = rendered
            self._publish_debug(image, header)
            self.last_workspace_overlay = time.monotonic()

    def _info_cb(self, msg):
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
            self.get_logger().warn(
                'Received invalid camera parameters'
            )
            return

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.camera_info = msg
    def _images_cb(self, rgb_msg, depth_msg):
        if self.camera_info is None:
            return
        
        if (self.camera_info.header.frame_id != rgb_msg.header.frame_id) or (self.camera_info.width != rgb_msg.width) or (self.camera_info.height != rgb_msg.height):
            self.get_logger().warn('Color Camera does not match color image')
            return

        if depth_msg.encoding != '16UC1':
            self.get_logger().warn(
                f'Expected 16UC1 depth, received {depth_msg.encoding}'
            )
            return

        if (rgb_msg.width, rgb_msg.height) != (
            depth_msg.width, depth_msg.height
        ):
            self.get_logger().warn(
                'Color and aligned depth dimensions do not match'
            )
            return

        if (
            not rgb_msg.header.frame_id
            or rgb_msg.header.frame_id != depth_msg.header.frame_id
        ):
            self.get_logger().warn(
                'Color and aligned depth coordinate frames do not match'
            )
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        except Exception as exc:
            self.get_logger().warn(f'Could not convert depth image: {exc}')
            return

        with self._lock:
            self.depth = depth

        self._rgb_cb(rgb_msg)

    def _get_depth(self, u, v):
        with self._lock:
            if self.depth is None:
                return 0.0
            h, w = self.depth.shape
            u = max(0, min(u, w-1))
            v = max(0, min(v, h-1))
            patch = self.depth[
                max(0,v-8):min(h,v+8),
                max(0,u-8):min(w,u+8)]
            valid = patch[patch > 0]
            if len(valid) == 0:
                return 0.0
            return float(np.median(valid)) / 1000.0

    def _backproject(self, u, v, depth_m):
        if depth_m < 0.1 or self.fx is None:
            return None
        cx3 = (u - self.cx) * depth_m / self.fx
        cy3 = (v - self.cy) * depth_m / self.fy
        cz3 = depth_m
        return cx3, cy3, cz3

    def _rgb_cb(self, msg):
        if self.fx is None:
            return
        t0 = time.perf_counter()
        self._frame_count += 1
        want_debug = self._has_debug_viewer() and self._frame_count % 2 == 0

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            return

        raw_detections = []
        display = bgr.copy() if want_debug else bgr

        # yolo detection
        if self.model is not None:
            try:
                # Car/train/truck boxes for the same toy must compete across
                # classes; otherwise class-specific NMS can retain duplicates.
                results = self.model(
                    bgr, device=self.device, verbose=False, conf=0.30,
                    classes=[2, 6, 7], agnostic_nms=True, iou=0.50,
                )
                if results:
                    boxes = results[0].boxes.cpu()
                    for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                        x1, y1, x2, y2 = map(int, xyxy)
                        raw_detections.append((self.model.names[int(cls_id)], conf, (x1 + x2) // 2, (y1 + y2) // 2))
                    self.get_logger().info(f'Detected {len(raw_detections)} | Timing: {results[0].speed}', throttle_duration_sec=2.0)
            except Exception as e:
                self.get_logger().warn(f'YOLO error: {e}')

        detections = []

        for label, conf, u, v in raw_detections:
            depth_m = self._get_depth(u, v)
            if depth_m < 0.1:
                self.get_logger().warn(
                    f'{label} detected but invalid depth at ({u}, {v}): {depth_m:.3f} m',
                    throttle_duration_sec=2.0
                )
                continue
            coords = self._backproject(u, v, depth_m)
            if coords is None:
                continue
            cx3, cy3, cz3 = coords

            detections.append({
                'label':      label,
                'confidence': round(conf, 3),
                'cx_3d':      round(cx3, 4),
                'cy_3d':      round(cy3, 4),
                'cz_3d':      round(cz3, 4),
                'frame_id':   msg.header.frame_id,
                'stamp':      msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'source':     'realsense',
                'pixel_center': [u, v],
            })

            if want_debug:
                # Draw the center point
                cv2.circle(display, (u, v), 6, (0, 0, 255), -1)

                # Filtered XYZ arrives asynchronously on workspace_topic.
                # Only the matching source frame will receive that overlay.
                cv2.putText(display, f"{label} {conf:.0%}", (max(0,u-20), max(80,v-15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if want_debug and detections:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.workspace_overlay.remember(
                stamp, display, msg.header,
                [tuple(d['pixel_center']) for d in detections]
            )

        self.det_pub.publish(String(data=json.dumps(detections)))

        if want_debug:
            # Show a useful preview without presenting old XYZ as current.
            waiting = self.last_workspace_overlay is None or time.monotonic()-self.last_workspace_overlay > .5
            if not detections or waiting:
                message = 'Waiting for filtered workspace coordinates' if detections else 'No valid object detection'
                cv2.rectangle(display, (0,0), (min(display.shape[1]-1,850),40), (25,25,25), -1)
                cv2.putText(display, message, (10,27), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,220,255), 2)
                self._publish_debug(display, msg.header)

        self.get_logger().info(f'callback total: {(time.perf_counter() - t0) * 1000:.1f} ms', throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node._has_display:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()