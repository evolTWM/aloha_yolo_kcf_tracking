#!/usr/bin/env python3
"""Convert the existing high-camera YOLO JSON detections to left-base XYZ.

Reads only perception data and a saved calibration; never commands motion.
Camera optical XYZ must already be in metres. Does not use the table-board pose.
"""
import json
import math
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml

if __package__:
    from .calibration_common import load_camera_calibration
else:
    from calibration_common import load_camera_calibration

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = str(PACKAGE_DIR / 'calibration_candidate.yaml')


def default_workspace_parameters():
    # Bundled lab setup: use ROS parameters to override for another table/camera.
    config = yaml.safe_load((PACKAGE_DIR / 'lab_workspace.yaml').read_text())
    return config['yolo_to_robot']['ros__parameters']


def transform_detection(detection, matrix, source_frame, target_frame, now, max_age):
    if not isinstance(detection,dict):
        raise ValueError('Each detection must be an object')
    if detection.get('frame_id')!=source_frame:
        raise ValueError('Detection frame differs from calibration optical frame')
    values=[detection.get(k) for k in ('cx_3d','cy_3d','cz_3d','stamp','confidence')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('Require finite camera XYZ, stamp and confidence')
    x,y,z,stamp,confidence=values
    if z<=0 or not 0<=confidence<=1:
        raise ValueError('Invalid camera depth or confidence')
    age=now-stamp
    if age>max_age or age<-.1:
        raise ValueError(f'Detection is stale or clock differs (age {age:.3f}s)')
    result=matrix@np.array([x,y,z,1.])
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite transformed point')
    output = dict(label=detection.get('label',''),confidence=confidence,
                x=float(result[0]),y=float(result[1]),z=float(result[2]),
                frame_id=target_frame,units='m',stamp=stamp,
                camera_frame_id=source_frame,camera_position=dict(x=x,y=y,z=z),
                source=detection.get('source','realsense'))
    pixel = detection.get('pixel_center')
    if isinstance(pixel,list) and len(pixel)==2 and all(type(v) is int for v in pixel):
        output['pixel_center'] = pixel.copy()
    return output


def workspace_detection(filtered, origin_x, origin_y, frame):
    """Translate XY only; retain base XYZ and its frame for downstream users."""
    output = dict(
        label=filtered['label'], confidence=filtered['confidence'],
        x=filtered['x']-origin_x, y=filtered['y']-origin_y, z=filtered['z'],
        frame_id=frame, units='m', stamp=filtered['stamp'],
        smoothing=filtered['smoothing'], filter_alpha=filtered['filter_alpha'],
        base_frame_id=filtered['frame_id'],
        base_position={axis:filtered[axis] for axis in ('x','y','z')},
    )
    for key in ('pixel_center','camera_frame_id'):
        if key in filtered:
            output[key] = filtered[key]
    return output


class YoloToRobot(Node):
    def __init__(self):
        super().__init__('yolo_to_robot')
        workspace = default_workspace_parameters()
        for name,value in dict(calibration_file=DEFAULT_CALIBRATION,
                reference_frame='follower_left/base_link',input_topic='/aloha/perception/detections',
                output_topic='/aloha/perception/detections_base',max_detection_age=1.,
                workspace_origin_x=workspace['workspace_origin_x'],
                workspace_origin_y=workspace['workspace_origin_y'],
                workspace_frame=workspace['workspace_frame']).items():
            self.declare_parameter(name,value)
        self.reference=self.get_parameter('reference_frame').value
        self.source,self.matrix=load_camera_calibration(self.get_parameter('calibration_file').value,self.reference)
        self.max_age=self.get_parameter('max_detection_age').value
        if not math.isfinite(self.max_age) or self.max_age<=0:
            raise ValueError('max_detection_age must be positive and finite')
        incoming=self.get_parameter('input_topic').value
        outgoing=self.get_parameter('output_topic').value
        if self.resolve_topic_name(incoming)==self.resolve_topic_name(outgoing):
            raise ValueError('Input and output topics must differ')
        self.publisher=self.create_publisher(String,outgoing,10)
        self.filtered_pub = self.create_publisher(String, '/aloha/perception/detections_base_filtered', 10)
        self.workspace_pub = self.create_publisher(
            String, '/aloha/perception/detections_workspace_filtered', 10
        )
        self.workspace_origin_x = self.get_parameter('workspace_origin_x').value
        self.workspace_origin_y = self.get_parameter('workspace_origin_y').value
        self.workspace_frame = self.get_parameter('workspace_frame').value
        if not all(math.isfinite(v) for v in (self.workspace_origin_x,self.workspace_origin_y)):
            raise ValueError('Workspace XY origin must be finite')
        if not self.workspace_frame or self.workspace_frame == self.reference:
            raise ValueError('Workspace frame must be nonempty and distinct from base frame')
        self.filter_alpha = 0.25
        self.filter_reset_seconds = 1.0
        self.filtered_xyz = None
        self.filter_stamp = None
        self.subscription=self.create_subscription(String,incoming,self.callback,10)
        self.get_logger().info(f'{incoming} [{self.source}] -> {outgoing} [{self.reference}], metres. No robot motion.')
        self.get_logger().info('Using an estimated calibration; transformed detections are not validated grasp targets.')
        self.get_logger().info('Bundled calibration and workspace defaults describe the lab setup; configure your own camera mounting and table origin on other robots.')
        self.get_logger().info(
            f'Workspace output: /aloha/perception/detections_workspace_filtered; '
            f'XY origin=({self.workspace_origin_x:.6f}, {self.workspace_origin_y:.6f}) m in {self.reference}; '
            'Z unchanged. Workspace frame is a JSON convention; no TF is broadcast.'
        )

    def callback(self,msg):
        try:
            detections=json.loads(msg.data)
            if not isinstance(detections,list):
                raise ValueError('Expected the YOLO JSON list')
        except (ValueError,TypeError) as exc:
            self.get_logger().warning(str(exc),throttle_duration_sec=3.)
            self.publisher.publish(String(data='[]'))
            self.publish_filtered([])
            return
        now=self.get_clock().now().nanoseconds/1e9
        output=[]
        for detection in detections:
            try:
                output.append(transform_detection(detection,self.matrix,self.source,self.reference,now,self.max_age))
            except ValueError as exc:
                self.get_logger().warning(str(exc),throttle_duration_sec=3.)
        self.publisher.publish(String(data=json.dumps(output,allow_nan=False)))
        self.publish_filtered(output)
        if output:
            first=output[0]
            self.get_logger().info(f"{first['label']}: x={first['x']:.4f}, y={first['y']:.4f}, z={first['z']:.4f} m",throttle_duration_sec=2.0)

    def publish_filtered(self, output):
        # This simple filter handles exactly one object.
        # Empty or ambiguous detections reset its history.
        if len(output) != 1:
            self.filtered_xyz = None
            self.filter_stamp = None
            self.filtered_pub.publish(String(data='[]'))
            self.workspace_pub.publish(String(data='[]'))
            if len(output) > 1:
                self.get_logger().warning(
                    'Multiple detections: single-object filter reset',
                    throttle_duration_sec=3.0
                )
            return

        detection = output[0]
        stamp = detection['stamp']
        xyz = np.array(
            [detection['x'], detection['y'], detection['z']],
            dtype=float
        )

        # Ignore duplicate or out-of-order measurements.
        if self.filter_stamp is not None and stamp <= self.filter_stamp:
            return

        if (
            self.filtered_xyz is None
            or stamp - self.filter_stamp > self.filter_reset_seconds
        ):
            self.filtered_xyz = xyz.copy()
        else:
            alpha = self.filter_alpha
            self.filtered_xyz = (
                alpha * xyz + (1.0 - alpha) * self.filtered_xyz
            )

        self.filter_stamp = stamp

        filtered = detection.copy()
        filtered['raw_position'] = {
            'x': detection['x'],
            'y': detection['y'],
            'z': detection['z'],
        }
        filtered['x'] = float(self.filtered_xyz[0])
        filtered['y'] = float(self.filtered_xyz[1])
        filtered['z'] = float(self.filtered_xyz[2])
        filtered['smoothing'] = 'ema'
        filtered['filter_alpha'] = self.filter_alpha

        self.filtered_pub.publish(
            String(data=json.dumps([filtered], allow_nan=False))
        )
        workspace = workspace_detection(
            filtered, self.workspace_origin_x, self.workspace_origin_y, self.workspace_frame
        )
        self.workspace_pub.publish(
            String(data=json.dumps([workspace], allow_nan=False))
        )


def main():
    rclpy.init()
    node=None
    try:
        node=YoloToRobot()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
