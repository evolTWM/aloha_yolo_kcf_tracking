"""State-machine regression tests; no ROS, camera or model download required."""
import unittest

import numpy as np

from aloha_tracking_perception.tracking import YoloKcfTracker, make_kcf


class FakeKcf:
    def __init__(self, updates=(), init_result=None):
        self.updates = iter(updates)
        self.init_result = init_result
        self.initialized_box = None

    def init(self, frame, bbox):
        self.initialized_box = bbox
        return self.init_result

    def update(self, frame):
        value = next(self.updates)
        if isinstance(value, Exception):
            raise value
        return value


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [('truck', 0.92, (20, 30, 30, 20))]

    def test_yolo_once_then_moving_kcf_then_loss_and_reacquisition(self):
        first = FakeKcf([(True, (25, 30, 30, 20)),
                         (True, (30, 30, 30, 20)), (False, (0, 0, 0, 0))])
        second = FakeKcf(init_result=True)
        instances = iter([first, second])
        tracker = YoloKcfTracker(lambda: next(instances), search_interval=5.0)
        self.assertEqual(tracker.step(self.frame, 0, self.detect).method, 'YOLO')
        for index in range(2):
            target = tracker.step(self.frame, 0.1 * (index + 1), self.detect)
            self.assertEqual(target.method, 'KCF')
            self.assertEqual(target.bbox[0], 25 + 5 * index)
            self.assertEqual(self.calls, 1)
        self.assertIsNone(tracker.step(self.frame, 0.3, self.detect))
        self.assertEqual(tracker.state, 'SEARCHING')
        self.assertIsNone(tracker.target)  # no stale box/XYZ on failure
        # Loss overrides the five-second empty-scene search throttle.
        self.assertEqual(tracker.step(self.frame, 0.4, self.detect).method, 'YOLO')
        self.assertEqual(self.calls, 2)
        self.assertEqual(tracker.yolo_calls, 2)
        self.assertEqual(tracker.kcf_updates, 3)

    def test_empty_scene_throttles_yolo_without_creating_tracker(self):
        def empty(frame):
            self.calls += 1
            return []
        tracker = YoloKcfTracker(lambda: self.fail('No detection to initialize'))
        for now in [0, 0.1, 0.49, 0.5, 0.9, 1.0]:
            self.assertIsNone(tracker.step(self.frame, now, empty))
        self.assertEqual(self.calls, 3)

    def test_reject_bad_kcf_boxes_even_if_update_reports_success(self):
        for bbox in [(161, 10, 30, 20), (150, 10, 30, 20),
                     (10, 10, float('nan'), 20), (10, 10, 0, 20)]:
            with self.subTest(bbox=bbox):
                tracker = YoloKcfTracker(lambda: FakeKcf([(True, bbox)]))
                tracker.step(self.frame, 0, self.detect)
                self.assertIsNone(tracker.step(self.frame, 1, self.detect))
                self.assertEqual(tracker.state, 'SEARCHING')

    def test_update_exception_clears_target(self):
        tracker = YoloKcfTracker(lambda: FakeKcf([RuntimeError('update error')]))
        tracker.step(self.frame, 0, self.detect)
        self.assertIsNone(tracker.step(self.frame, 1, self.detect))
        self.assertIn('update error', tracker.last_event)
        self.assertIsNone(tracker.target)

    def test_init_false_does_not_enter_tracking(self):
        tracker = YoloKcfTracker(lambda: FakeKcf(init_result=False))
        self.assertIsNone(tracker.step(self.frame, 0, self.detect))
        self.assertEqual(tracker.state, 'SEARCHING')

    def test_init_exception_does_not_enter_tracking(self):
        def unavailable():
            raise RuntimeError('no KCF')
        tracker = YoloKcfTracker(unavailable)
        self.assertIsNone(tracker.step(self.frame, 0, self.detect))
        self.assertIn('no KCF', tracker.last_event)

    def test_yolo_error_is_retried(self):
        tracker = YoloKcfTracker(lambda: FakeKcf())
        def failed(frame):
            raise RuntimeError('inference error')
        self.assertIsNone(tracker.step(self.frame, 0, failed))
        self.assertIn('inference error', tracker.last_event)
        self.assertIsNotNone(tracker.step(self.frame, 0.5, self.detect))

    def test_select_highest_confidence_target(self):
        tracker = YoloKcfTracker(lambda: FakeKcf())
        candidates = [('train', 0.6, (2, 2, 20, 20)),
                      ('truck', 0.9, (30, 30, 20, 20))]
        target = tracker.step(self.frame, 0, lambda frame: candidates)
        self.assertEqual(target.label, 'truck')
        self.assertEqual(target.detection_confidence, 0.9)

    def test_resolution_change_reacquires_instead_of_updating_old_kcf(self):
        tracker = YoloKcfTracker(lambda: FakeKcf())
        tracker.step(self.frame, 0, self.detect)
        bigger = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(tracker.step(bigger, 0.1, self.detect).method, 'YOLO')
        self.assertEqual(tracker.yolo_calls, 2)
        self.assertEqual(tracker.kcf_updates, 0)

    def test_make_kcf_modern_and_legacy(self):
        from types import SimpleNamespace
        for module in [SimpleNamespace(TrackerKCF_create=lambda: 'modern'),
                       SimpleNamespace(legacy=SimpleNamespace(TrackerKCF_create=lambda: 'legacy'))]:
            self.assertIn(make_kcf(module), ['modern', 'legacy'])
        with self.assertRaisesRegex(RuntimeError, 'KCF is unavailable'):
            make_kcf(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
