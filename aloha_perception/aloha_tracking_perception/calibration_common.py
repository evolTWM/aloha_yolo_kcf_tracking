"""Shared, ROS-independent geometry and image handling for ALOHA calibration.

T_a_b maps coordinates in frame b into frame a. Lengths are metres.
"""
import cv2
import numpy as np


def transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def quat_matrix(q):
    q = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quat(R):
    # Rodrigues provides a stable axis-angle representation, including near pi.
    v = cv2.Rodrigues(np.asarray(R, dtype=float))[0].reshape(3)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([0., 0., 0., 1.])
    return np.r_[v * (np.sin(angle / 2) / angle), np.cos(angle / 2)]


def tf_matrix(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return transform(quat_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])


def decode_image(msg):
    channels = {'mono8': 1, 'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding.lower())
    if channels is None or msg.step < msg.width * channels:
        raise ValueError('Unsupported encoding or invalid row stride')
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if len(raw) < msg.height * msg.step:
        raise ValueError('Truncated image')
    a = raw[:msg.height * msg.step].reshape(msg.height, msg.step)
    a = a[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if msg.encoding.lower() in ('rgb8', 'rgba8'):
        return a[:, :, [2, 1, 0]].copy()
    return a[:, :, :3].copy()


class MarkerDetector:
    def __init__(self, dictionary='DICT_4X4_50'):
        if not hasattr(cv2.aruco, dictionary):
            raise ValueError(f'Unknown ArUco dictionary: {dictionary}')
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(d, params)

    def detect(self, image, K, D, size, max_error=2.0):
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.last_detected_ids = [] if ids is None else [int(i) for i in ids.flatten()]
        if ids is None:
            return {}
        obj = np.array([[-size/2, size/2, 0], [size/2, size/2, 0],
                        [size/2, -size/2, 0], [-size/2, -size/2, 0]], dtype=float)
        result = {}
        for pixels, mid in zip(corners, ids.flatten()):
            n, rotations, translations, _ = cv2.solvePnPGeneric(
                obj, pixels.reshape(4, 2), K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            candidates = []
            for rv, tv in zip(rotations[:n], translations[:n]):
                R = cv2.Rodrigues(rv)[0]
                if np.min((R @ obj.T + tv.reshape(3, 1))[2]) <= 0:
                    continue
                projected = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(4, 2)
                error = float(np.sqrt(np.mean(np.sum((projected-pixels.reshape(4, 2))**2, axis=1))))
                if np.isfinite(error) and error <= max_error:
                    candidates.append((error, transform(R, tv)))
            if candidates:
                error, T = min(candidates, key=lambda x: x[0])
                result[int(mid)] = (T, error, pixels.reshape(4, 2))
        return result


def consistency(robot, target, X, mode):
    implied = [(np.linalg.inv(A) @ X @ B if mode == 'eye-to-hand' else A @ X @ B)
               for A, B in zip(robot, target)]
    positions = np.array([T[:3, 3] for T in implied])
    translation_rms = float(np.sqrt(np.mean(np.sum((positions-positions.mean(axis=0))**2, axis=1))))
    # Closest proper rotation to the mean rotation (chordal mean).
    u, _, vt = np.linalg.svd(sum(T[:3, :3] for T in implied))
    mean_R = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    angles = [np.arccos(np.clip((np.trace(mean_R.T @ T[:3, :3])-1)/2, -1, 1)) for T in implied]
    return translation_rms, float(np.degrees(np.sqrt(np.mean(np.square(angles)))))


def solve_hand_eye(robot, target, mode, min_translation=0.10, min_rotation_deg=25.):
    if len(robot) != len(target) or len(robot) < 4:
        raise ValueError('At least four paired samples are required')
    positions = np.array([T[:3, 3] for T in robot])
    if np.linalg.norm(np.ptp(positions, axis=0)) < min_translation:
        raise ValueError('Insufficient translation diversity')
    relative_rotations = [cv2.Rodrigues(A[:3, :3].T @ B[:3, :3])[0].reshape(3)
                          for i, A in enumerate(robot) for B in robot[i+1:]]
    singular = np.linalg.svd(np.array(relative_rotations), compute_uv=False)
    if max(np.linalg.norm(v) for v in relative_rotations) < np.radians(min_rotation_deg):
        raise ValueError('Insufficient orientation diversity')
    if singular[1] < 0.1 * singular[0]:
        raise ValueError('Rotate around multiple axes; a single rotation axis cannot determine hand-eye calibration')
    poses = [np.linalg.inv(A) for A in robot] if mode == 'eye-to-hand' else robot
    candidates = []
    for name in ('TSAI', 'PARK', 'HORAUD', 'ANDREFF'):
        try:
            R, t = cv2.calibrateHandEye([A[:3, :3] for A in poses], [A[:3, 3] for A in poses],
                                      [B[:3, :3] for B in target], [B[:3, 3] for B in target],
                                      method=getattr(cv2, 'CALIB_HAND_EYE_' + name))
            X = transform(R, t)
            if not np.all(np.isfinite(X)) or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0:
                continue
            tr, rr = consistency(robot, target, X, mode)
            candidates.append(dict(solver=name, matrix=X, translation_rms_m=tr, rotation_rms_deg=rr))
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        raise ValueError('All hand-eye solvers failed')
    return sorted(candidates, key=lambda c: c['translation_rms_m'] + .01*c['rotation_rms_deg'])


def board_pose(corners, ids):
    """Four marker centres in TL, TR, BR, BL order; reconstruct one missing."""
    points = [corners.get(i) for i in ids]
    if sum(p is not None for p in points) < 3:
        return None
    if any(p is None for p in points):
        missing = next(i for i, p in enumerate(points) if p is None)
        points[missing] = points[(missing-1)%4] + points[(missing+1)%4] - points[(missing+2)%4]
    tl, tr, br, bl = points
    x = (tr-tl)+(br-bl)
    y = (tl-bl)+(tr-br)
    z = np.cross(x, y)
    if np.linalg.norm(x) < 1e-6 or np.linalg.norm(z) < 1e-8:
        return None
    x, z = x/np.linalg.norm(x), z/np.linalg.norm(z)
    return transform(np.column_stack([x, np.cross(z, x), z]), np.mean(points, axis=0))


def load_camera_calibration(path, reference_frame):
    """Load a fixed optical-to-reference transform without modifying ROS TF."""
    from pathlib import Path
    import yaml
    result = yaml.safe_load(Path(path).expanduser().read_text())
    if not isinstance(result, dict) or result.get('parent_frame') != reference_frame:
        raise ValueError('Calibration parent_frame must match the fusion reference_frame')
    optical = result.get('camera_optical_frame')
    if not isinstance(optical, str) or not optical:
        raise ValueError('Calibration needs a nonempty camera_optical_frame')
    T = np.asarray(result.get('T_parent_camera_optical'), dtype=float)
    if T.shape != (4,4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0,0,0,1], atol=1e-8):
        raise ValueError('Calibration must contain a finite homogeneous 4x4 matrix')
    R = T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1.,atol=1e-6):
        raise ValueError('Calibration rotation must be a proper rotation')
    return optical,T
