"""Tracking, geometry and filtering utilities; no ROS dependencies."""
from dataclasses import dataclass

import numpy as np


def make_kcf(cv2_module):
    if hasattr(cv2_module, 'TrackerKCF_create'):
        return cv2_module.TrackerKCF_create()
    legacy = getattr(cv2_module, 'legacy', None)
    if legacy is not None and hasattr(legacy, 'TrackerKCF_create'):
        return legacy.TrackerKCF_create()
    raise RuntimeError(
        'OpenCV KCF is unavailable. Check cv2.TrackerKCF_create or '
        'cv2.legacy.TrackerKCF_create in the ROS Python environment; '
        'do not change shared OpenCV installations without lab approval.'
    )


def clip_bbox(bbox, width, height):
    x, y, w, h = [float(v) for v in bbox]
    if not np.isfinite([x, y, w, h]).all() or w <= 1 or h <= 1:
        return None
    x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


@dataclass(frozen=True)
class Target:
    label: str
    detection_confidence: float
    bbox: tuple  # (x, y, width, height), in pixels
    method: str  # YOLO acquisition or KCF update; confidence is always from YOLO


class YoloKcfTracker:
    """Acquire one target, update KCF only, and reacquire after loss.

    detect(frame) returns (label, confidence, xywh) candidates. The highest
    confidence candidate wins; reacquisition does not guarantee object identity.
    Call step with a clean BGR frame, before drawing any annotations.
    """

    def __init__(self, tracker_factory, search_interval=0.5):
        if not np.isfinite(search_interval) or search_interval < 0:
            raise ValueError('search_interval must be finite and nonnegative')
        self.tracker_factory = tracker_factory
        self.search_interval = search_interval
        self.yolo_calls = 0
        self.kcf_updates = 0
        self._shape = None
        self.reset()

    @property
    def state(self):
        return 'TRACKING' if self._tracker is not None else 'SEARCHING'

    def reset(self, reason=''):
        self._tracker = None
        self.target = None
        self.last_event = reason
        self._last_search = None

    @staticmethod
    def _visible_box(bbox, width, height):
        x, y, w, h = [float(v) for v in bbox]
        clipped = clip_bbox(bbox, width, height)
        if clipped is None:
            return None
        # Do not keep tracking a tiny sliver outside the image.
        if not (0 <= x + w / 2 < width and 0 <= y + h / 2 < height):
            return None
        if clipped[2] * clipped[3] < 0.5 * w * h:
            return None
        return clipped

    def step(self, frame, now, detect):
        height, width = frame.shape[:2]
        if self._shape is not None and self._shape != (height, width):
            self.reset('Image size changed; searching again')
        self._shape = (height, width)

        if self._tracker is not None:
            self.kcf_updates += 1
            try:
                ok, bbox = self._tracker.update(frame)
                clipped = self._visible_box(bbox, width, height) if ok else None
            except Exception as exc:
                self.reset(f'KCF update failed: {exc}')
                return None
            if clipped is None:
                self.reset('KCF lost target; searching again')
                # Publish an empty result for the loss frame. YOLO resumes on
                # the next frame, without waiting for the search interval.
                return None
            self.target = Target(self.target.label,
                                 self.target.detection_confidence, clipped, 'KCF')
            return self.target

        if self._last_search is not None and now - self._last_search < self.search_interval:
            return None
        self._last_search = now
        self.yolo_calls += 1
        try:
            candidates = sorted(detect(frame), key=lambda item: item[1], reverse=True)
        except Exception as exc:
            self.last_event = f'YOLO failed: {exc}'
            return None
        for label, confidence, bbox in candidates:
            clipped = self._visible_box(bbox, width, height)
            if clipped is None:
                continue
            try:
                tracker = self.tracker_factory()
                initialized = tracker.init(frame, clipped)
                # Modern OpenCV returns None on success; legacy may return bool.
                if initialized is not None and not initialized:
                    self.last_event = 'KCF initialization failed; searching again'
                    continue
            except Exception as exc:
                self.last_event = f'KCF initialization failed: {exc}'
                continue
            self._tracker = tracker
            self.target = Target(label, confidence, clipped, 'YOLO')
            self.last_event = f'YOLO acquired {label}; KCF initialized'
            return self.target
        return None


def depth_at_target(depth_mm, bbox, depth_min=0.15, depth_max=3.0):
    """Robust visible-surface depth (meters), sampled at center of bbox.

    Returns (u, v, z) or None. This is NOT the object's geometric center.
    """
    height, width = depth_mm.shape
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return None
    x, y, w, h = clipped
    u = x + w // 2
    v = y + h // 2
    # Central patch avoids much of the background around the tracked box.
    half_w = max(3, min(12, w // 6))
    half_h = max(3, min(12, h // 6))
    roi = depth_mm[max(y, v-half_h):min(y+h, v+half_h+1),
                   max(x, u-half_w):min(x+w, u+half_w+1)]
    valid = roi[(roi > 0) & (roi >= depth_min * 1000) &
                (roi <= depth_max * 1000)]
    if valid.size < 6:
        return None
    med = np.median(valid)
    # Reject isolated depth values, but do not synthesize missing depth.
    good = valid[np.abs(valid.astype(np.float64) - med) <= 80.0]
    if good.size < 6:
        return None
    return (u, v, float(np.median(good)) * 0.001)


def backproject(u, v, z, intrinsics):
    fx, fy, cx, cy = intrinsics
    if not np.isfinite([fx, fy, cx, cy, z]).all() or fx <= 0 or fy <= 0 or z <= 0:
        return None
    return np.array([(u-cx)*z/fx, (v-cy)*z/fy, z], dtype=np.float64)


class ConstantVelocityKalman:
    """XYZ plus velocity. Filtering is for stability, not calibration accuracy."""
    def __init__(self, process_acceleration=1.0, measurement_sigma=0.015):
        self.x = None
        self.P = None
        self.accel = float(process_acceleration)
        self.meas_var = float(measurement_sigma) ** 2

    def reset(self):
        self.x = self.P = None

    def update(self, measurement, dt):
        z = np.asarray(measurement, dtype=np.float64).reshape(3)
        if not np.isfinite(z).all():
            return None
        if self.x is None:
            self.x = np.r_[z, np.zeros(3)]
            self.P = np.diag([0.02**2]*3 + [0.5**2]*3)
            return self.x[:3].copy()
        dt = float(np.clip(dt, 0.001, 0.5))
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        G = np.vstack((np.eye(3)*(dt*dt/2), np.eye(3)*dt))
        Q = (self.accel ** 2) * (G @ G.T)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        H = np.hstack((np.eye(3), np.zeros((3, 3))))
        R = np.eye(3) * self.meas_var
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x[:3].copy()
