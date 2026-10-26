"""Optional real KCF smoke test on a deterministic synthetic moving texture."""
import unittest

import numpy as np

from aloha_tracking_perception.tracking import YoloKcfTracker, make_kcf

try:
    import cv2
    HAS_KCF = hasattr(cv2, 'TrackerKCF_create') or hasattr(getattr(cv2, 'legacy', None), 'TrackerKCF_create')
except ImportError:
    HAS_KCF = False


@unittest.skipUnless(HAS_KCF, 'OpenCV with KCF is required')
class OpenCvKcfTests(unittest.TestCase):
    def test_texture_moves_exits_and_returns(self):
        rng = np.random.default_rng(2026)
        texture = rng.integers(30, 255, size=(48, 64, 3), dtype=np.uint8)
        def frame_at(x=None):
            frame = np.zeros((240, 320, 3), np.uint8)
            if x is not None:
                frame[80:128, x:x + 64] = texture
            return frame
        detections = []
        def detect(frame):
            detections.append(True)
            return [('truck', 0.9, (80, 80, 64, 48))]
        tracker = YoloKcfTracker(lambda: make_kcf(cv2), search_interval=0)
        tracker.step(frame_at(80), 0, detect)
        for index in range(1, 21):
            x = 80 + index * 2
            target = tracker.step(frame_at(x), index / 30, detect)
            self.assertIsNotNone(target)
            self.assertEqual(target.method, 'KCF')
            self.assertLessEqual(abs(target.bbox[0] - x), 6)
        self.assertEqual(len(detections), 1)
        self.assertIsNone(tracker.step(frame_at(), 1, detect))
        self.assertEqual(tracker.state, 'SEARCHING')
        self.assertEqual(tracker.step(frame_at(80), 1.1, detect).method, 'YOLO')
        self.assertEqual(len(detections), 2)


if __name__ == '__main__':
    unittest.main()
