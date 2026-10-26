"""Exercise the ROS adapter with message/publisher stubs and real OpenCV."""
import importlib
import json
import sys
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from aloha_tracking_perception.tracking import YoloKcfTracker


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


@unittest.skipIf(cv2 is None, 'OpenCV not installed in test Python')
class DetectionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_modules = {
            'rclpy': module('rclpy'),
            'rclpy.node': module('rclpy.node', Node=object),
            'rclpy.qos': module('rclpy.qos', QoSProfile=Mock(), ReliabilityPolicy=Mock(),
                               DurabilityPolicy=Mock(), HistoryPolicy=Mock()),
            'cv_bridge': module('cv_bridge', CvBridge=Mock()),
            'sensor_msgs': module('sensor_msgs'),
            'sensor_msgs.msg': module('sensor_msgs.msg', Image=object, CameraInfo=object),
            'std_msgs': module('std_msgs'),
            'std_msgs.msg': module('std_msgs.msg', String=lambda **kw: NS(**kw)),
            'ultralytics': module('ultralytics', YOLO=Mock()),
        }
        with patch.dict(sys.modules, fake_modules):
            cls.Node = importlib.import_module('aloha_tracking_perception.detection').RealSenseDetection

    def setUp(self):
        self.frame = np.zeros((120, 160, 3), np.uint8)
        self.header = NS(frame_id='color_optical', stamp=NS(sec=10, nanosec=0))
        self.rgb = NS(header=self.header, width=160, height=120, data=self.frame)
        self.node = object.__new__(self.Node)
        self.node.bridge = NS(imgmsg_to_cv2=lambda msg, **kw: msg.data,
                              cv2_to_imgmsg=lambda frame, **kw: NS(data=frame))
        self.node.depth_msg = None
        self.node.camera_info = None
        self.node.max_depth_age = 0.1
        self.node.det_pub = Mock()
        self.node.debug_pub = Mock()
        self.node.get_logger = lambda: Mock()
        self.node._logged_event = ''

    def attach_depth(self, seconds=10, encoding='16UC1', frame_id='color_optical'):
        self.node.camera_info = NS(width=160, height=120, header=self.header,
                                   k=[100, 0, 80, 0, 100, 60, 0, 0, 1])
        self.node.depth_msg = NS(width=160, height=120, encoding=encoding,
                                header=NS(frame_id=frame_id, stamp=NS(sec=seconds, nanosec=0)),
                                data=np.full((120, 160), 1000, np.uint16))

    def test_rgb_tracks_and_publishes_green_box_without_depth(self):
        frames_seen = []
        def detect(frame):
            frames_seen.append(frame.copy())
            return [('truck', 0.9, (20, 30, 30, 20))]
        tracker = NS(init=lambda frame, bbox: None,
                     update=lambda frame: (frames_seen.append(frame.copy()) or True,
                                           (25, 30, 30, 20)))
        self.node.tracking = YoloKcfTracker(lambda: tracker)
        self.node._detect = detect
        self.node._rgb_cb(self.rgb)
        self.node._rgb_cb(self.rgb)
        self.assertEqual(self.node.tracking.yolo_calls, 1)
        self.assertEqual(self.node.tracking.kcf_updates, 1)
        self.assertEqual(self.node.debug_pub.publish.call_count, 2)
        self.assertEqual(json.loads(self.node.det_pub.publish.call_args.args[0].data), [])
        preview = self.node.debug_pub.publish.call_args.args[0]
        self.assertIs(preview.header, self.header)
        self.assertEqual(tuple(preview.data[49, 30]), (0, 255, 0))  # green lower edge
        self.assertEqual(tuple(preview.data[40, 40]), (0, 0, 255))  # red center
        for frame in frames_seen:
            self.assertFalse(frame.any())  # KCF never sees drawn overlays
        self.assertFalse(self.frame.any())

    def test_valid_depth_backprojects_camera_xyz(self):
        self.attach_depth()
        coords = self.node._xyz(self.rgb, (20, 30, 30, 20))
        np.testing.assert_allclose(coords, [-0.45, -0.20, 1.0])

    def test_unusable_depth_is_never_reported_as_xyz(self):
        for seconds, encoding, frame_id in [(9, '16UC1', 'color_optical'),
                                             (10, '32FC1', 'color_optical'),
                                             (10, '16UC1', 'depth_optical')]:
            with self.subTest(seconds=seconds, encoding=encoding, frame_id=frame_id):
                self.attach_depth(seconds, encoding, frame_id)
                self.assertIsNone(self.node._xyz(self.rgb, (20, 30, 30, 20)))
        self.attach_depth()
        self.node.depth_msg.data[:] = 0
        self.assertIsNone(self.node._xyz(self.rgb, (20, 30, 30, 20)))

    def test_depth_stream_ending_does_not_reuse_old_xyz(self):
        self.attach_depth()
        tracker = NS(init=lambda frame, bbox: None,
                     update=lambda frame: (True, (20, 30, 30, 20)))
        self.node.tracking = YoloKcfTracker(lambda: tracker)
        self.node._detect = lambda frame: [('truck', 0.9, (20, 30, 30, 20))]
        self.node._rgb_cb(self.rgb)
        result = json.loads(self.node.det_pub.publish.call_args.args[0].data)[0]
        self.assertEqual(result['cz_3d'], 1.0)
        self.assertEqual(result['confidence_source'], 'last_yolo_detection')
        self.rgb.header = NS(frame_id='color_optical', stamp=NS(sec=11, nanosec=0))
        self.node._rgb_cb(self.rgb)
        self.assertEqual(json.loads(self.node.det_pub.publish.call_args.args[0].data), [])
        self.assertEqual(self.node.tracking.state, 'TRACKING')

    def test_loss_frame_clears_detections(self):
        self.attach_depth()
        tracker = NS(init=lambda frame, bbox: None,
                     update=lambda frame: (False, (20, 30, 30, 20)))
        self.node.tracking = YoloKcfTracker(lambda: tracker)
        self.node._detect = lambda frame: [('truck', 0.9, (20, 30, 30, 20))]
        self.node._rgb_cb(self.rgb)
        self.node._rgb_cb(self.rgb)
        self.assertEqual(json.loads(self.node.det_pub.publish.call_args.args[0].data), [])
        self.assertEqual(self.node.tracking.state, 'SEARCHING')


if __name__ == '__main__':
    unittest.main()
