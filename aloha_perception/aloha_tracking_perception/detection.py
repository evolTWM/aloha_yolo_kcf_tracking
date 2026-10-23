#!/usr/bin/env python3
"""
ALOHA overhead-camera object detector
Uses a supplied YOLO model and aligned RealSense Depth
Publishes labeled camera-relative XYZ coordinates in meters.
"""
print("LATEST UPDATE (WORKING FOR THE MOST PART)")
import os
import rclpy, threading, json
import message_filters
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

class RealSenseDetection(Node):
    def __init__(self):
        super().__init__('detector')

        self.bridge  = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_info = None
        self.depth = None
        self._lock = threading.Lock()
        # Debug preview window needs a real X11/Wayland display -- fine with
        # someone logged into the GUI session, fatal (Qt xcb plugin abort)
        # running headless over SSH, which is how this node runs on REAL-1.
        self._has_display = bool(os.environ.get('DISPLAY'))

        # YOLO
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics not available.")

        self.model = YOLO('yolov8m.pt')
        self.get_logger().info(f"Loaded pretrained YOLOv8m with classes: {self.model.names}")

        self.target_classes = ['truck', 'train']

        # self.target_classes = ['fire-truck', 'screwdriver', 'cell-phone']

        # Subscriptions
        self.declare_parameter('rgb_topic', '/cam_high/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/cam_high/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/cam_high/camera/color/camera_info')

        self.rgb_sub = message_filters.Subscriber(self, Image, self.get_parameter('rgb_topic').value, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.get_parameter('depth_topic').value, qos_profile=qos_profile_sensor_data)
        self.image_sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=5, slop=0.04)

        self.image_sync.registerCallback(self._images_cb)

        self.create_subscription(CameraInfo, self.get_parameter('info_topic').value, self._info_cb, qos_profile_sensor_data)

        self.declare_parameter('output_topic', '/aloha/perception/detections')
        self.det_pub = self.create_publisher(String, self.get_parameter('output_topic').value, 10)
        self.debug_pub = self.create_publisher(Image, '/aloha/perception/debug_image', qos_profile_sensor_data)
        self.get_logger().info('RealSense detection node ready')

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

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return

        raw_detections = []
        display = bgr.copy()

        # yolo detection
        if self.model is not None:
            try:
                results = self.model(bgr, verbose=False, conf=0.30)
                if not results:    
                    self.get_logger().warn("YOLO returned no results")
                else:
                    display = results[0].plot()
                    observed = [(self.model.names[int(box.cls.item())], round(float(box.conf.item()), 3)) for box in results[0].boxes]

                self.get_logger().info(f'YOLO sees: {observed}', throttle_duration_sec=2.0)
                self.get_logger().info(f"Device: {next(self.model.model.parameters()).device}, Timing: {results[0].speed}", throttle_duration_sec=5.0)
                for r in results:
                    for box in r.boxes:
                        cls_name = self.model.names[int(box.cls)]
                        if cls_name not in self.target_classes:
                            continue
                        conf = float(box.conf)
                        if conf < 0.30:
                            continue
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        u = (x1+x2)//2
                        v = (y1+y2)//2
                        raw_detections.append((cls_name, conf, u, v))
            except Exception as e:
                self.get_logger().warn(f'YOLO error: {e}')


        if not raw_detections:
            self.det_pub.publish(String(data='[]'))
            preview = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
            preview.header = msg.header
            self.debug_pub.publish(preview)
            return

        detections = []

        for label, conf, u, v in raw_detections:
            depth_m = self._get_depth(u, v)
            if depth_m < 0.1:
                self.get_logger().warn(
                    f'Truck detected but invalid depth at ({u}, {v}): {depth_m:.3f} m',
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
                'stamp':   msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'source':     'realsense',
            })

            # Draw the center point
            cv2.circle(display, (u, v), 6, (0, 0, 255), -1)

            # Show pixel coordinates, depth, and 3D coordinates
            cv2.putText(display, f"{label} {conf:.0%}", (u + 10, v - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(display, f"Pixel: ({u}, {v})  Depth: {depth_m:.3f} m", (u + 10, v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(display, f"XYZ: ({cx3:.3f}, {cy3:.3f}, {cz3:.3f}) m", (u + 10, v + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )

        self.det_pub.publish(String(data=json.dumps(detections)))
        preview = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
        preview.header = msg.header
        self.debug_pub.publish(preview)

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
