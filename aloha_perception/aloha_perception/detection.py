#!/usr/bin/env python3
"""YOLO once -> KCF on later frames -> aligned RealSense XYZ; no GUI here."""
import json
import time

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ultralytics import YOLO

from aloha_perception.tracking import (
    ConstantVelocityKalman, backproject, clip_bbox,
    depth_at_target, make_kcf,
)


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class RealSenseDetection(Node):
    def __init__(self):
        super().__init__('realsense_detection')
        self.declare_parameter('model_path', 'yolov8s.pt')
        self.declare_parameter('target_class', 'truck')
        self.declare_parameter('confidence', 0.40)
        self.declare_parameter('rgb_topic', '/cam_high/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/cam_high/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/cam_high/camera/color/camera_info')
        self.declare_parameter('detection_topic', '/aloha/perception/detections')
        self.declare_parameter('debug_topic', '/aloha/perception/debug_image')
        self.declare_parameter('synchronizer_slop', 0.06)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('max_lost_frames', 3)
        self.declare_parameter('max_position_jump_m', 0.30)
        self.declare_parameter('debug_every_n_frames', 1)

        self.target_class = str(self.get_parameter('target_class').value)
        self.min_conf = float(self.get_parameter('confidence').value)
        model_path = str(self.get_parameter('model_path').value)
        self.model = YOLO(model_path)
        self.get_logger().info('Loaded YOLO model %s; classes=%s' % (model_path, self.model.names))
        if self.target_class not in self.model.names.values():
            raise ValueError('Target class %r not in loaded model classes' % self.target_class)
        # Fail early if there is no KCF implementation, before touching camera streams.
        make_kcf(cv2)

        self.bridge = CvBridge()
        self.tracker = None
        self.mode = 'DETECTING'
        self.tracked_box = None
        self.last_conf = 0.0
        self.lost_frames = 0
        self.last_stamp = None
        self.filter = ConstantVelocityKalman(process_acceleration=1.0, measurement_sigma=0.015)
        self.intrinsics = None
        self.info_frame = None
        self.info_size = None
        self.counter = 0
        self.last_log_time = 0.0

        self.det_pub = self.create_publisher(String, str(self.get_parameter('detection_topic').value), 10)
        # Image QoS depth=1 avoids queueing stale preview images.
        self.debug_pub = self.create_publisher(Image, str(self.get_parameter('debug_topic').value), qos_profile_sensor_data)
        self.reset_service = self.create_service(Trigger, '~/reset_target', self._reset_cb)

        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        info_topic = str(self.get_parameter('info_topic').value)
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=6,
            slop=float(self.get_parameter('synchronizer_slop').value),
        )
        self.sync.registerCallback(self._images_cb)
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data)
        self.get_logger().info('Ready: DETECTING -> TRACKING -> reacquire on loss; GUI disabled in detector')

    def _reset_target(self):
        self.tracker = None
        self.tracked_box = None
        self.mode = 'DETECTING'
        self.lost_frames = 0
        self.last_stamp = None
        self.filter.reset()

    def _reset_cb(self, request, response):
        self._reset_target()
        response.success = True
        response.message = 'Target reset; YOLO will detect a new truck'
        return response

    def _info_cb(self, msg):
        k = msg.k
        vals = (k[0], k[4], k[2], k[5])
        if not np.isfinite(vals).all() or vals[0] <= 0 or vals[1] <= 0:
            return
        self.intrinsics = vals
        self.info_frame = msg.header.frame_id
        self.info_size = (msg.width, msg.height)

    def _detect(self, bgr):
        results = self.model(bgr, verbose=False)
        candidates = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if self.model.names[cls_id] != self.target_class:
                    continue
                conf = float(box.conf.item())
                if conf < self.min_conf:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                bbox = clip_bbox((x1, y1, x2-x1, y2-y1), bgr.shape[1], bgr.shape[0])
                if bbox is not None:
                    candidates.append((conf, bbox))
        return max(candidates, key=lambda pair: pair[0]) if candidates else None

    def _publish_debug(self, bgr, header, box=None, xyz=None):
        n = max(1, int(self.get_parameter('debug_every_n_frames').value))
        self.counter += 1
        if self.counter % n:
            return
        preview = bgr.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
        title = self.mode
        if xyz is not None:
            title += ' XYZ=(%.3f, %.3f, %.3f)m' % tuple(xyz)
        cv2.putText(preview, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0) if self.mode == 'TRACKING' else (0, 180, 255), 2)
        image_msg = self.bridge.cv2_to_imgmsg(preview, encoding='bgr8')
        image_msg.header = header
        self.debug_pub.publish(image_msg)

    def _images_cb(self, rgb_msg, depth_msg):
        # These checks must pass before using pixels and camera intrinsics.
        if self.intrinsics is None:
            return
        if self.info_frame != rgb_msg.header.frame_id or self.info_size != (rgb_msg.width, rgb_msg.height):
            self.get_logger().warn('CameraInfo does not match RGB frame', throttle_duration_sec=5.0)
            return
        if rgb_msg.width != depth_msg.width or rgb_msg.height != depth_msg.height:
            self.get_logger().warn('RGB and depth sizes differ', throttle_duration_sec=5.0)
            return
        if depth_msg.encoding != '16UC1':
            self.get_logger().warn('Expected 16UC1 aligned depth, got %s' % depth_msg.encoding,
                                   throttle_duration_sec=5.0)
            return
        if not rgb_msg.header.frame_id or rgb_msg.header.frame_id != depth_msg.header.frame_id:
            self.get_logger().warn('RGB/depth optical frames differ', throttle_duration_sec=5.0)
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        except Exception as exc:
            self.get_logger().error('Image conversion failed: %s' % exc, throttle_duration_sec=3.0)
            return

        stamp = stamp_seconds(rgb_msg.header.stamp)
        dt = stamp - self.last_stamp if self.last_stamp is not None else 1.0/15.0
        self.last_stamp = stamp
        if dt < 0 or dt > 1.0:
            self.filter.reset()
            dt = 1.0 / 15.0

        if self.mode == 'DETECTING':
            t0 = time.perf_counter()
            try:
                candidate = self._detect(bgr)
            except Exception as exc:
                self.get_logger().error('YOLO failed: %s' % exc, throttle_duration_sec=3.0)
                self.det_pub.publish(String(data='[]'))
                self._publish_debug(bgr, rgb_msg.header)
                return
            if candidate is not None:
                confidence, bbox = candidate
                tracker = make_kcf(cv2)
                try:
                    initialized = tracker.init(bgr, bbox)
                    if initialized is False:
                        raise RuntimeError('KCF init returned False')
                except Exception as exc:
                    self.get_logger().warn('KCF init failed: %s' % exc, throttle_duration_sec=3.0)
                else:
                    self.tracker = tracker
                    self.tracked_box = bbox
                    self.last_conf = confidence
                    self.mode = 'TRACKING'
                    self.lost_frames = 0
                    self.filter.reset()
                    self.get_logger().info('Truck acquired; YOLO inference %.3fs; switching to KCF' %
                                           (time.perf_counter() - t0))
        else:
            ok, proposed = False, None
            try:
                ok, proposed = self.tracker.update(bgr)
            except Exception as exc:
                self.get_logger().warn('KCF update failed: %s' % exc, throttle_duration_sec=3.0)
            bbox = clip_bbox(proposed, bgr.shape[1], bgr.shape[0]) if ok else None
            if bbox is not None:
                self.tracked_box = bbox
                self.lost_frames = 0
            else:
                self.lost_frames += 1
                if self.lost_frames >= max(1, int(self.get_parameter('max_lost_frames').value)):
                    self.get_logger().warn('Tracking lost; returning to YOLO')
                    self._reset_target()

        payload = []
        xyz = None
        if self.mode == 'TRACKING' and self.tracked_box is not None and self.lost_frames == 0:
            sample = depth_at_target(depth, self.tracked_box,
                                     depth_max=float(self.get_parameter('max_depth_m').value))
            if sample is not None:
                u, v, z = sample
                measured = backproject(u, v, z, self.intrinsics)
                if measured is not None:
                    # Protect against sudden depth/background jumps; do not report stale XYZ.
                    if self.filter.x is not None and np.linalg.norm(measured - self.filter.x[:3]) > \
                            float(self.get_parameter('max_position_jump_m').value):
                        self.get_logger().warn('Rejected implausible XYZ jump', throttle_duration_sec=3.0)
                    else:
                        xyz = self.filter.update(measured, dt)
                        payload = [{
                            'label': self.target_class,
                            'confidence': round(self.last_conf, 3),  # initial YOLO confidence, NOT tracker confidence
                            'cx_3d': round(float(xyz[0]), 4),
                            'cy_3d': round(float(xyz[1]), 4),
                            'cz_3d': round(float(xyz[2]), 4),
                            'frame_id': rgb_msg.header.frame_id,
                            'stamp': stamp,
                            'source': 'realsense',
                            'tracking_state': 'TRACKING',
                        }]
        # [] is deliberately sent for lost / invalid-depth frames: never reuse stale coordinates.
        self.det_pub.publish(String(data=json.dumps(payload)))
        self._publish_debug(bgr, rgb_msg.header,
                            box=self.tracked_box if self.mode == 'TRACKING' else None, xyz=xyz)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealSenseDetection()
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
