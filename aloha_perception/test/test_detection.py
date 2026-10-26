"""Test KCF additions through the original detector with ROS/model stubs."""
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


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class NodeStub:
    def __init__(self, name):
        self.params = {}
        self.logger = Mock()

    def declare_parameter(self, name, value):
        self.params[name] = value

    def get_parameter(self, name):
        return NS(value=self.params[name])

    def get_logger(self):
        return self.logger

    def create_subscription(self, *args):
        return Mock()

    def create_publisher(self, *args):
        return Mock()


class BridgeStub:
    def imgmsg_to_cv2(self, msg, encoding):
        return msg.data

    def cv2_to_imgmsg(self, frame, encoding):
        return NS(data=frame)


class FakeKcf:
    def __init__(self, updates=(), init_result=None):
        self.updates = iter(updates)
        self.init_result = init_result
        self.frames = []

    def init(self, frame, bbox):
        self.frames.append(frame.copy())
        self.bbox = bbox
        return self.init_result

    def update(self, frame):
        self.frames.append(frame.copy())
        value = next(self.updates)
        if isinstance(value, Exception):
            raise value
        return value


@unittest.skipIf(cv2 is None, 'OpenCV is required for detector integration tests')
class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_modules = {
            'rclpy': module('rclpy'),
            'rclpy.node': module('rclpy.node', Node=NodeStub),
            'rclpy.qos': module('rclpy.qos', qos_profile_sensor_data=object()),
            'message_filters': module('message_filters', Subscriber=Mock(),
                                      ApproximateTimeSynchronizer=Mock()),
            'cv_bridge': module('cv_bridge', CvBridge=BridgeStub),
            'sensor_msgs': module('sensor_msgs'),
            'sensor_msgs.msg': module('sensor_msgs.msg', Image=object, CameraInfo=object),
            'std_msgs': module('std_msgs'),
            'std_msgs.msg': module('std_msgs.msg', String=lambda **kw: NS(**kw)),
            'ultralytics': module('ultralytics', YOLO=Mock()),
        }
        with patch.dict(sys.modules, fake_modules):
            cls.detector = importlib.import_module('aloha_tracking_perception.detection')

    def setUp(self):
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.header = NS(frame_id='color_optical', stamp=NS(sec=10, nanosec=0))
        self.rgb = NS(header=self.header, width=160, height=120, data=self.frame)
        self.depth = NS(header=self.header, width=160, height=120, encoding='16UC1',
                        data=np.full((120, 160), 1000, np.uint16))
        self.info = NS(header=self.header, width=160, height=120,
                       k=[100, 0, 80, 0, 100, 60, 0, 0, 1])
        self.model = Mock()
        self.model.names = {7: 'truck', 6: 'train', 0: 'person'}
        self.model.model.parameters.return_value = iter([NS(device='cpu')])
        self.model.return_value = self.results()
        self.kcf = FakeKcf()
        self.make_patch = patch.object(self.detector, 'make_kcf', return_value=self.kcf)
        self.make_patch.start()
        self.addCleanup(self.make_patch.stop)
        with patch.object(self.detector, 'YOLO', return_value=self.model):
            self.node = self.detector.RealSenseDetection()
        self.node._info_cb(self.info)

    def results(self, class_id=7, conf=0.9):
        box = NS(cls=np.float32(class_id), conf=np.float32(conf),
                 xyxy=np.array([[20, 30, 50, 50]]))
        return [NS(boxes=[box], speed={}, plot=lambda: self.frame.copy())]

    def process(self):
        self.node._images_cb(self.rgb, self.depth)
        return json.loads(self.node.det_pub.publish.call_args.args[0].data)

    def test_yolo_once_kcf_moves_loss_clears_and_next_pair_reacquires(self):
        self.kcf.updates = iter([(True, (25, 30, 30, 20)), (False, (0, 0, 0, 0))])
        self.assertEqual(self.process()[0]['label'], 'truck')
        self.assertEqual(self.model.call_count, 1)
        result = self.process()[0]
        self.assertEqual(self.model.call_count, 1)
        self.assertEqual(self.node.yolo_calls, 1)
        self.assertEqual(self.node.kcf_updates, 1)
        self.assertAlmostEqual(result['cx_3d'], -0.4)
        preview = self.node.debug_pub.publish.call_args.args[0]
        self.assertEqual(tuple(preview.data[49, 30]), (0, 255, 0))
        self.assertEqual(tuple(preview.data[40, 40]), (0, 0, 255))
        self.assertIs(preview.header, self.header)
        self.assertEqual(self.process(), [])
        self.assertIsNone(self.node.tracker)
        self.assertIsNone(self.node.tracked_label)
        self.assertEqual(self.model.call_count, 1)
        self.model.model.parameters.return_value = iter([NS(device='cpu')])
        self.assertEqual(self.process()[0]['label'], 'truck')
        self.assertEqual(self.model.call_count, 2)
        for frame in self.kcf.frames:
            self.assertFalse(frame.any())  # no annotations fed back into KCF
        self.assertFalse(self.frame.any())

    def test_original_xyz_math_and_json_fields_are_preserved(self):
        result = self.process()[0]
        self.assertEqual(set(result), {'label', 'confidence', 'cx_3d', 'cy_3d', 'cz_3d',
                                      'frame_id', 'stamp', 'source'})
        self.assertEqual((result['cx_3d'], result['cy_3d'], result['cz_3d']),
                         (-0.45, -0.2, 1.0))

    def test_invalid_depth_keeps_kcf_but_does_not_publish_xyz(self):
        self.depth.data[:] = 0
        self.assertEqual(self.process(), [])
        self.assertIsNotNone(self.node.tracker)
        self.kcf.updates = iter([(True, (25, 30, 30, 20))])
        self.assertEqual(self.process(), [])
        self.assertEqual(self.model.call_count, 1)
        preview = self.node.debug_pub.publish.call_args.args[0]
        self.assertEqual(tuple(preview.data[49, 30]), (0, 255, 0))

    def test_original_sync_validation_still_rejects_bad_depth(self):
        self.depth.encoding = '32FC1'
        self.node._images_cb(self.rgb, self.depth)
        self.model.assert_not_called()
        self.node.debug_pub.publish.assert_not_called()

    def test_empty_yolo_result_does_not_raise_or_publish_stale_target(self):
        self.model.return_value = []
        self.assertEqual(self.process(), [])
        self.assertIsNone(self.node.tracker)
        self.assertFalse(self.node.logger.warn.call_args_list[-1].args[0].startswith('YOLO error'))

    def test_only_allowed_classes_acquire_kcf(self):
        self.model.return_value = self.results(class_id=0)
        self.assertEqual(self.process(), [])
        self.assertIsNone(self.node.tracker)

    def test_kcf_init_false_leaves_yolo_enabled(self):
        self.kcf.init_result = False
        self.process()
        self.assertIsNone(self.node.tracker)
        self.model.model.parameters.return_value = iter([NS(device='cpu')])
        self.process()
        self.assertEqual(self.model.call_count, 2)

    def test_legacy_true_init_is_accepted(self):
        self.kcf.init_result = True
        self.process()
        self.assertIs(self.node.tracker, self.kcf)

    def test_bad_kcf_boxes_clear_the_target(self):
        self.process()
        for bbox in [(170, 10, 30, 20), (10, 10, 0, 20), (10, 10, float('nan'), 20)]:
            with self.subTest(bbox=bbox):
                self.node.tracker = self.kcf
                self.node.tracked_label, self.node.tracked_conf = 'truck', 0.9
                self.node.tracked_shape = self.frame.shape[:2]
                self.kcf.updates = iter([(True, bbox)])
                self.assertEqual(self.process(), [])
                self.assertIsNone(self.node.tracker)

    def test_kcf_exception_clears_the_target(self):
        self.process()
        self.kcf.updates = iter([RuntimeError('update failed')])
        self.assertEqual(self.process(), [])
        self.assertIsNone(self.node.tracker)

    def test_resolution_change_discards_old_tracker(self):
        self.process()
        self.node.tracked_shape = (240, 320)
        self.assertEqual(self.process(), [])
        self.assertIsNone(self.node.tracker)

    def test_two_targets_select_one_for_single_object_tracking(self):
        self.model.return_value[0].boxes += self.results(class_id=6)[0].boxes
        self.assertEqual(len(self.process()), 1)
        self.assertEqual(self.node.tracked_label, 'truck')

    def test_real_opencv_kcf_on_moving_texture_loss_and_return(self):
        from aloha_tracking_perception.tracking import make_kcf
        try:
            make_kcf(cv2)
        except RuntimeError:
            self.skipTest('This OpenCV does not have KCF')
        texture = np.random.default_rng(2026).integers(30, 255, (20, 30, 3), np.uint8)
        def position(x=None):
            self.frame[:] = 0
            if x is not None:
                self.frame[30:50, x:x+30] = texture
        with patch.object(self.detector, 'make_kcf', side_effect=lambda _: make_kcf(cv2)):
            position(20)
            self.process()
            for index in range(1, 11):
                position(20+index)
                self.process()
                self.assertIsNotNone(self.node.tracker)
            self.assertEqual(self.model.call_count, 1)
            position()
            self.assertEqual(self.process(), [])
            self.assertIsNone(self.node.tracker)
            position(20)
            self.model.model.parameters.return_value = iter([NS(device='cpu')])
            self.process()
            self.assertIsNotNone(self.node.tracker)
            self.assertEqual(self.model.call_count, 2)


if __name__ == '__main__':
    unittest.main()
