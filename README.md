# ALOHA Object Detection and 3D Localization

ROS 2 package for detecting objects using YOLO and estimating their 3D positions using an Intel RealSense depth camera.

The detector uses YOLO (`yolov8m.pt`) to acquire **one truck or train**, then follows it with **OpenCV KCF without running YOLO during successful tracking**. When KCF reports failure or an invalid/out-of-frame box, the detector clears the old target and resumes YOLO search. It publishes an annotated RGB image and estimates **camera-relative XYZ in meters** when recent aligned depth and matching intrinsics are available.

> **Current stage:** YOLO acquisition → KCF tracking → YOLO reacquisition. **Kalman filtering is not enabled.** An unused Kalman utility remains in `tracking.py` for a later stage. XYZ is a visible-surface depth measurement near the box center, not the geometric object center or a robot-base coordinate.

## Requirements

- Ubuntu with ROS 2 Humble and the lab's ALOHA setup
- Intel RealSense camera and the lab's RealSense ROS 2 driver
- Python dependencies already configured for ROS 2, OpenCV **with KCF**, NumPy, CV Bridge, and Ultralytics
- Either `cv2.TrackerKCF_create` or `cv2.legacy.TrackerKCF_create` in the **same Python environment used by ROS 2**
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
ros2 launch aloha aloha_bringup.launch.py is_mobile:=false use_base:=false use_joystick_teleop:=false use_cameras:=false
```

Use this only when approved by the lab operator. Bringup may start robot hardware processes; it is not required solely to inspect an already-running camera stream. Keep this terminal open.

### Terminal 2 — RealSense camera

After sourcing the three setup files:

```bash
ros2 launch realsense2_camera rs_launch.py   camera_namespace:=cam_high   camera_name:=camera   serial_no:=_021222070323   enable_color:=true   enable_depth:=true   align_depth.enable:=true   enable_infra1:=false   enable_infra2:=false   enable_gyro:=false   enable_accel:=false   pointcloud.enable:=false   rgb_camera.color_profile:=640,480,15   depth_module.depth_profile:=640,480,15
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
ros2 run aloha_tracking_perception detector --ros-args -p "target_classes:=[truck]"
```

Omit the parameter to allow both `truck` and `train`. Start only this detector for the test; a separately running old YOLO detector would publish conflicting debug images. The node fails at startup with a clear error if KCF is unavailable. Check the ROS Python environment without changing its packages:

```bash
python3 -c "import cv2; print(cv2.__version__); print('KCF:', hasattr(cv2, 'TrackerKCF_create') or hasattr(getattr(cv2, 'legacy', None), 'TrackerKCF_create'))"
```

The detector subscribes to color, aligned depth and camera info. RGB drives tracking independently; missing or stale depth does not freeze the box. Keep this terminal open.

### Terminal 4 — Image viewer

After sourcing the three setup files:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select:

```text
/aloha/perception/debug_image
```

The viewer shows:

- `SEARCHING`: YOLO searches every 0.5 seconds by default; RGB images continue publishing between searches.
- Yellow box + `[YOLO]`: the acquisition frame that initializes KCF.
- Green box + `[KCF]`: subsequent KCF updates, with a red center point and pixel `(u, v)`.
- `YOLO calls` and `KCF updates`: cumulative counters for verifying that YOLO stops during successful tracking.
- XYZ if valid, or an `XYZ unavailable` message while the 2D box keeps tracking.

The initial YOLO frame may be too brief to see at camera frame rate.

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
  "confidence_source": "last_yolo_detection",
  "tracking_method": "KCF",
  "u": 337,
  "v": 207,
  "cx_3d": 0.020,
  "cy_3d": -0.040,
  "cz_3d": 0.734,
  "frame_id": "camera_color_optical_frame",
  "stamp": 123.0,
  "source": "realsense"
}
```

*The numbers above are examples, not measured values.* The topic uses a `std_msgs/String` containing a JSON array with **at most one target**. It publishes `[]` when the target is lost **or** valid XYZ cannot be measured. `confidence` is the last YOLO acquisition score, not a KCF confidence score. `tracking_method` identifies whether the current box came from YOLO or KCF.

## Pipeline and stage-one acceptance test

1. Place the truck on the table. In `SEARCHING`, YOLO selects the highest-confidence allowed target and initializes KCF on the clean BGR image.
2. Move the truck slowly. The green box and red center point should follow. `KCF updates` increases while `YOLO calls` stays fixed.
3. Move the truck fully out of view. When KCF returns failure, throws an error, or reports an invalid/out-of-frame box, the node discards the box, publishes `[]`, and logs `KCF lost target` (or the failure reason).
4. YOLO resumes on the next RGB callback, then searches at the configured interval while no target is found. Return the truck to view; a new acquisition should restore the green KCF box.
5. Depth failures should not stop steps 1–4. XYZ uses a small center patch, valid depth range 0.15–3.0 m, and matching camera metadata. It is not filtered by Kalman.

**Limits:** KCF is a visual tracker, not a semantic detector. It can drift onto background or a similar object while reporting success. This stage does not periodically run YOLO to detect such drift. Reacquisition selects the most confident allowed object, so it does not preserve identity if several trucks are visible. KCF's fixed box scale also limits large distance/scale changes. Validate with the actual truck, lighting and camera; synthetic tests do not establish lab FPS or tracking reliability.

### Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `model` | `yolov8m.pt` | YOLO acquisition weights |
| `target_classes` | `[truck, train]` | Allowed model class names |
| `confidence` | `0.30` | Minimum YOLO acquisition confidence |
| `search_interval` | `0.5` seconds | Minimum interval between empty-scene YOLO attempts; bypassed after tracking loss |
| `max_depth_age` | `0.10` seconds | Maximum absolute RGB/depth timestamp difference for XYZ |

These parameters are set at node startup. Setting `search_interval:=0.0` searches every processed RGB frame **only while SEARCHING**. There is no periodic YOLO inference during successful KCF tracking.

### Software checks

From the repository root:

```bash
PYTHONPATH=aloha_perception python3 -m unittest discover -s aloha_perception/test -v
```

The state-machine tests do not require ROS or a YOLO model. Adapter tests use ROS message/publisher stubs and real OpenCV; the optional real-KCF test follows a deterministic moving texture, loses it on a blank frame, and reacquires it. OpenCV-dependent tests are skipped if their dependency is unavailable. No camera, ROS middleware, GPU or robot behavior is simulated as a hardware validation.

## Next stage

- Add Kalman filtering after the YOLO/KCF acceptance test passes on the lab workstation.
- Validate XYZ jitter, motion lag, coordinate error, processed frame rate and latency independently.
- Add drift checks if KCF falsely reports success after the target leaves; this needs separate validation and may require occasional YOLO verification.

## Notes

- XYZ coordinates are **in the camera optical frame**, not the robot base frame. Robot-coordinate use requires separate calibration.
- `16UC1` aligned depth is interpreted as millimeters by the current detector.
- This project only performs perception; it does not control the robot.
- Launch commands and workspace locations can differ between lab machines. Use the versions installed and approved for your workstation.
- To stop your test processes, press `Ctrl+C` in their terminals. Avoid stopping shared lab processes without coordination.
