# ALOHA Object Detection and 3D Localization

ROS 2 package for detecting objects using YOLO and estimating their 3D positions using an Intel RealSense depth camera.

The current implementation detects **trucks and trains** with YOLO (`yolov8m.pt`), reads synchronized RGB and aligned depth images, and estimates **camera-relative X, Y, Z coordinates in meters** using the camera intrinsics. It also publishes an annotated image showing the detection box, center point, depth, and coordinates.

> **Current status:** This is a YOLO-based detection pipeline. **KCF tracking and Kalman filtering are planned future additions; they are not implemented in the current version.** The reported XYZ point is a depth measurement near the center of the detected box, not the object's geometric center or a robot-base coordinate.

## Requirements

- Ubuntu with ROS 2 Humble and the lab's ALOHA setup
- Intel RealSense camera and the lab's RealSense ROS 2 driver
- Python dependencies already configured for ROS 2, OpenCV, NumPy, CV Bridge, and Ultralytics
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
source /path/to/your/workspace/install/setup.bash
```

### Terminal 1 — ALOHA bringup

After sourcing the three setup files:

```bash
ros2 launch aloha aloha_bringup.launch.py is_mobile:=false
```

Use this only when approved by the lab operator. Bringup may start robot hardware processes; it is not required solely to inspect an already-running camera stream. Keep this terminal open.

### Terminal 2 — RealSense camera

After sourcing the three setup files:

```bash
ros2 launch realsense_camera rs_launch.py
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

The viewer should show YOLO boxes with center points, measured depth, and XYZ coordinates **when the camera is publishing and a target has valid depth data**.

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

*The numbers above are examples, not measured values.* The topic uses a `std_msgs/String` containing a JSON array; it may contain multiple detections or `[]` when none can be reported.

## Current implementation

1. YOLO detects a target (`truck` or `train`) in the RGB image.
2. The center pixel `(u, v)` is calculated from the detection box.
3. The aligned depth image provides a median depth value near that pixel.
4. The camera intrinsics convert the pixel and depth to camera-frame XYZ.
5. The detector publishes JSON results to `/aloha/perception/detections` and an annotated image to `/aloha/perception/debug_image`.

YOLO currently runs on each processed image pair; there is **no persistent KCF object tracking yet**.

## Future work

- **KCF tracking:** Run YOLO to acquire a target, then follow that target across subsequent frames without running YOLO on every frame.
- **Kalman filtering:** Smooth XYZ measurements and estimate target motion.
- **Target reacquisition:** Re-run YOLO if tracking is lost.
- **Validation:** Measure detection accuracy, coordinate error, frame rate, and end-to-end latency on the lab workstation.

## Notes

- XYZ coordinates are **in the camera optical frame**, not the robot base frame. Robot-coordinate use requires separate calibration.
- `16UC1` aligned depth is interpreted as millimeters by the current detector.
- This project only performs perception; it does not control the robot.
- Launch commands and workspace locations can differ between lab machines. Use the versions installed and approved for your workstation.
- To stop your test processes, press `Ctrl+C` in their terminals. Avoid stopping shared lab processes without coordination.
