# ALOHA Object Detection and 3D Localization

ROS 2 package for detecting objects using YOLO and estimating their 3D positions using an Intel RealSense depth camera.

The implementation detects **trucks and trains** with YOLO (`yolov8m.pt`), initializes **one KCF tracker** from the first eligible detection, and follows that object without running YOLO during successful tracking. It keeps the original synchronized RGB/aligned-depth callbacks, camera-relative XYZ calculation, and debug-image publisher.

> **Current status:** KCF is added directly to the original `detection.py`; the callback structure and XYZ output fields are preserved. **Kalman filtering is not enabled.** XYZ remains a depth measurement near the box center, not the geometric object center or a robot-base coordinate.

## Requirements

- Ubuntu with ROS 2 Humble and the lab's ALOHA setup
- Intel RealSense camera and the lab's RealSense ROS 2 driver
- Python dependencies already configured for ROS 2, OpenCV with KCF, NumPy, CV Bridge, and Ultralytics
- KCF must be available via `cv2.TrackerKCF_create` or `cv2.legacy.TrackerKCF_create` in the ROS Python environment
- `yolov8m.pt` available to the detector

**Package name:** `aloha_tracking_perception`  
**Detector executable:** `detector`

## 1. Clone and build

In a terminal with the ROS 2 and Interbotix environments available, clone the repository into the `src` directory of **your actual ROS 2 workspace**. Replace the paths below if your workspace is elsewhere.

```bash
git clone https://github.com/evolTWM/aloha_yolo_kcf_tracking.git
```

Move or clone the repository under your workspace's `src/` directory, then build from the workspace root:

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
cd /path/to/your/workspace
colcon build --packages-select aloha_tracking_perception
source /path/to/your/workspace/install/setup.bash
```

Confirm ROS 2 can find the installed package:

```bash
ros2 pkg prefix aloha_tracking_perception
ros2 pkg executables aloha_tracking_perception
```

> The directory holding `package.xml` must declare the package name `aloha_tracking_perception`. If the repository includes an extra wrapper directory, place or build the inner ROS package accordingly. Do not build a different copy of the package by accident.

## 2. Run the system (four terminals)

**In every terminal**, source the following three environments in this order. Replace the third path with your actual tracking workspace's `install/setup.bash` location:

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/setup.bash
```

### Terminal 1 — ALOHA bringup

After sourcing the three setup files:

```bash
ros2 launch aloha aloha_bringup.launch.py is_mobile:=false use_base:=false use_joystick_teleop:=false use_cameras:=false
```

Use this only when approved by the lab operator. Bringup may start robot hardware processes; it is not required solely to inspect an already-running camera stream. Keep this terminal open.

### Terminal 2 — RealSense camera

After sourcing the three setup files:

```bash
ros2 launch realsense2_camera rs_launch.py   camera_namespace:=cam_high   camera_name:=camera   serial_no:=_021222070323   enable_color:=true   enable_depth:=true   align_depth.enable:=true   enable_infra1:=false   enable_infra2:=false   enable_gyro:=false   enable_accel:=false   pointcloud.enable:=false   rgb_camera.color_profile:=640,480,15   depth_module.depth_profile:=640,480,15
```

This is the **lab-provided launch command**. Confirm that it exists on your workstation and publishes the expected `cam_high` topics. Do **not** start a second camera driver if the RealSense camera is already running. Keep this terminal open.

Check the image streams in another sourced shell if necessary:

```bash
ros2 topic info /cam_high/camera/color/image_raw
ros2 topic hz /cam_high/camera/color/image_raw
ros2 topic hz /cam_high/camera/aligned_depth_to_color/image_raw
```

Press `Ctrl+C` to stop each frequency check. The image topics must have an active publisher for detection to work.

### Terminal 3 — YOLO detector

After sourcing the three setup files:

```bash
ros2 run aloha_tracking_perception detector
```

The detector subscribes to the overhead camera's color, aligned depth, and camera-info topics and publishes detection measurements and an annotated image. Keep this terminal open.

### Terminal 4 — Image viewer

After sourcing the three setup files:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select:

```text
/aloha/perception/debug_image
```

The acquisition frame retains the original YOLO plot. Successful KCF updates draw a green box. The original red center point, pixel coordinates and XYZ annotations are drawn when depth is valid. The top line shows `SEARCHING`/`TRACKING`, `YOLO calls`, and `KCF updates`.

## 3. Read the XYZ output

In another sourced terminal, run:

```bash
ros2 topic echo /aloha/perception/detections
```

A typical detection includes:

```json
{
  "label": "truck",
  "confidence": 0.91,
  "cx_3d": 0.020,
  "cy_3d": -0.040,
  "cz_3d": 0.734,
  "frame_id": "camera_color_optical_frame",
  "stamp": 123.0,
  "source": "realsense"
}
```

*The numbers above are examples, not measured values.* The topic retains the original `std_msgs/String` JSON fields. This stage tracks at most one target and publishes `[]` when none can be reported. During KCF tracking, `confidence` is the last YOLO score, not a new KCF confidence estimate.

## Current implementation

1. When no tracker exists, the original YOLO block detects a target (`truck` or `train`) and initializes KCF from the first eligible box.
2. On following synchronized image pairs, KCF updates the box and supplies its center to the original depth/XYZ code; YOLO is skipped.
3. If KCF returns failure, throws an error, or reports an invalid/fully-outside box, the old target is cleared and `[]` is published for that frame. YOLO resumes on the next synchronized pair.
4. The center pixel `(u, v)` is calculated from the current box.
5. The original aligned-depth patch provides a median depth value near that pixel.
6. The original camera intrinsics convert the pixel and depth to camera-frame XYZ.
7. The original publishers send JSON to `/aloha/perception/detections` and images to `/aloha/perception/debug_image`.

The original `_images_cb`, `_get_depth`, `_backproject`, subscriptions, synchronization and JSON schema are preserved. Changes in `detection.py` are marked `# Added:` where practical. KCF uses the existing `make_kcf` and `clip_bbox` helpers; `tracking.py` is unchanged from the original version.

During successful tracking, no YOLO inference runs. During searching, YOLO still runs on every processed image pair, as in the original code. There are no new model/class/search-interval ROS parameters; the original model, class list and 0.30 threshold remain in the source.

## Stage-one acceptance test

1. Place one truck on the table. YOLO should acquire it and the terminal should log `KCF initialized`.
2. Move it slowly. The green box should follow, `KCF updates` should increase, and `YOLO calls` should stay fixed.
3. Move it fully out of view. After KCF reports failure, the terminal should log `KCF lost target`; the node clears the result and resumes YOLO on the next synchronized image pair.
4. Return the truck. YOLO should initialize a new tracker.

Because the original synchronization is retained, tracking still requires accepted RGB/depth pairs and matching camera info. Invalid depth pixels suppress XYZ but do not discard an initialized KCF tracker. If a depth stream stops, this version does not independently process RGB.

KCF can drift onto background while reporting success. This version does not periodically run YOLO to detect that drift, and reacquisition does not guarantee the same physical object's identity. Test with the actual truck and camera.

## Software checks

From the repository root:

```bash
PYTHONPATH=aloha_perception python3 -m unittest discover -s aloha_perception/test -v
```

Tests exercise the original detector with ROS/model stubs and real OpenCV, including acquisition, tracking, failure/reacquisition, the original XYZ/JSON output, and a moving synthetic texture with real KCF. OpenCV-dependent tests are skipped if OpenCV/KCF is unavailable. These checks do not validate ROS middleware, RealSense hardware or lab frame rate.

## Future work

- **Kalman filtering:** Smooth XYZ measurements and estimate target motion.
- **Validation:** Measure detection accuracy, coordinate error, frame rate, and end-to-end latency on the lab workstation.

## Notes

- XYZ coordinates are **in the camera optical frame**, not the robot base frame. Robot-coordinate use requires separate calibration.
- `16UC1` aligned depth is interpreted as millimeters by the current detector.
- This project only performs perception; it does not control the robot.
- Launch commands and workspace locations can differ between lab machines. Use the versions installed and approved for your workstation.
- To stop your test processes, press `Ctrl+C` in their terminals. Avoid stopping shared lab processes without coordination.
