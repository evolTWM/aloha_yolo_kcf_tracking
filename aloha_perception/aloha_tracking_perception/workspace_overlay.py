"""Match filtered workspace coordinates to their source debug image.

No ROS or inference dependencies; the detector supplies images and headers.
"""
from collections import OrderedDict
import math
import cv2


class WorkspaceOverlay:
    def __init__(self, frame='table_workspace', max_frames=12, max_age=1.0):
        self.frame, self.max_frames, self.max_age = frame, max_frames, max_age
        self.frames = OrderedDict()

    def remember(self, stamp, image, header, centres):
        self.frames[stamp] = (image.copy(), header, set(centres))
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)

    def clear(self):
        self.frames.clear()

    def render(self, detections, now):
        # The current EMA intentionally supports one object only.
        if not isinstance(detections, list) or len(detections) != 1:
            self.clear()
            return None
        d = detections[0]
        if not isinstance(d, dict):
            return None
        stamp = d.get('stamp')
        values = [d.get(k) for k in ('x', 'y', 'z', 'stamp')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            return None
        if d.get('frame_id') != self.frame or d.get('units') != 'm' or d.get('smoothing') != 'ema':
            return None
        if not 0 <= now-stamp <= self.max_age:
            return None
        source = self.frames.pop(stamp, None)
        if source is None:
            return None
        image, header, centres = source
        if d.get('camera_frame_id') != header.frame_id:
            return None
        pixel = d.get('pixel_center')
        if not isinstance(pixel, list) or len(pixel) != 2 or any(type(v) is not int for v in pixel):
            return None
        u, v = pixel
        if (u,v) not in centres:
            return None
        # A fixed panel prevents text clipping near the image edges.
        # It is labeled as workspace XYZ; camera-frame XYZ is never substituted.
        text = f"Workspace EMA: X={d['x']:+.3f}  Y={d['y']:+.3f}  Z={d['z']:+.3f} m"
        cv2.rectangle(image, (0,0), (min(image.shape[1]-1,920),62), (25,25,25), -1)
        cv2.putText(image, text, (10,24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2)
        cv2.putText(image, 'Table-centred XY | Z relative to left base | filtered surface point',
                    (10,49), cv2.FONT_HERSHEY_SIMPLEX, .45, (230,230,230), 1)
        cv2.circle(image, (u,v), 6, (0,0,255), -1)
        return image, header
