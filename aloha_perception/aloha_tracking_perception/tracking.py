"""Pure tracking, geometry and filtering utilities for the ALOHA perception node."""
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
