# ALOHA YOLO-once / KCF tracker (independent test package)

**What it does:** YOLO identifies a truck, KCF tracks that original target in subsequent synchronized RGB/depth image pairs, calibrated RealSense intrinsics convert a valid surface-depth point to camera-frame XYZ, and a constant-velocity Kalman filter smooths output. After tracking fails, the detector retries YOLO. It does **not** command robot motion.

**Safety:** Do not run robot bringup, teleoperation, change camera USB ownership, or power-cycle shared hardware as part of this test. Use an existing approved RealSense camera driver. This implementation has not been tested on your lab's Ubuntu / ROS 2 workstation.

## Important measurement limitations

- `cx_3d, cy_3d, cz_3d` are a tracked **visible surface point** near the bounding-box center in the **camera optical frame**, not the geometric center of the truck, a grasp point, or robot/world coordinates.
- KCF is an appearance tracker, **not** a semantic classifier or identity guarantee. It can drift to nearby similar objects or stay falsely locked. No tracker-confidence estimate is available. Treat `confidence` as the YOLO confidence **from the initial detection**, not a new confidence score every frame.
- A Kalman filter reduces jitter, but smoothing adds latency and does not eliminate systematic depth/calibration error. Verify measured XYZ against known distances; do not use for arm motion without validated calibration, collision avoidance, and supervision.
- On failed tracking or invalid/jumpy depth, the detection payload is `[]` for that frame instead of invented/stale XYZ. If KCF erroneously reports success on the background, however, output can still be inaccurate. A production safety system needs independent object/identity validation.

## Contents

```text
aloha_perception/
  package.xml
  setup.py
  setup.cfg
  resource/aloha_perception
  aloha_perception/
    __init__.py
    detection.py       # YOLO -> KCF -> depth -> Kalman, no imshow
    tracking.py        # KCF factory, depth sampling, geometry, Kalman
    visualization.py   # optional separate-process window
```

## ROS environment and compatibility checks

Do not run `pip install` in the shared lab environment without approval. In a terminal on the robot workstation:

```bash
conda deactivate  # only if conda (base) is active
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
python3 -c "import cv2; print('OpenCV',cv2.__version__); print('KCF',hasattr(cv2,'TrackerKCF_create') or hasattr(getattr(cv2,'legacy',None),'TrackerKCF_create')); import numpy, rclpy, message_filters, cv_bridge, ultralytics; print('Python dependencies OK')"
```

If KCF is `False`, stop and consult the lab's environment owner. Many OpenCV builds need the *contrib* tracking module. Avoid blindly installing a second OpenCV wheel into the shared ROS Python environment. A pre-existing Ultralytics package and `yolov8s.pt` model must be available.

## Install as a separate, reversible workspace (RECOMMENDED)

The package deliberately uses the same ROS package name and `detector` executable as Anthony's repository, so **do not overwrite or mix it with the current lab workspaces**. Unzip and copy the inner `aloha_perception/` package directory into a new isolated workspace:

```bash
mkdir -p ~/aloha_tracking_ws/src
cp -r /path/to/unzipped/aloha_perception ~/aloha_tracking_ws/src/
cd ~/aloha_tracking_ws
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
colcon build --packages-select aloha_perception
 source ~/interbotix_ws/src/aloha_tracking_ws/install/setup.bash
ros2 pkg prefix aloha_perception
ros2 pkg executables aloha_perception
```

`ros2 pkg prefix` MUST show `~/aloha_tracking_ws/install/aloha_perception`; otherwise stop and fix overlay order. Use a fresh terminal with the same sourcing order for each command below.

## Camera verification (don't start another driver if one already owns it)

```bash
ros2 topic list | grep cam_high
ros2 topic hz /cam_high/camera/color/image_raw
# Ctrl+C once several frequency samples appear
ros2 topic hz /cam_high/camera/aligned_depth_to_color/image_raw
# Ctrl+C once several frequency samples appear
```

Both RGB and aligned depth must publish; expected optical frame IDs and RGB CameraInfo dimensions must match. This code expects depth encoding `16UC1`, interpreted as millimeters. The driver serial/configuration is not changed by this package.

## Start detector (terminal A)

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/setup.bash
ros2 run aloha_perception detector --ros-args \
  -p model_path:=yolov8s.pt \
  -p target_class:=truck \
  -p rgb_topic:=/cam_high/camera/color/image_raw \
  -p depth_topic:=/cam_high/camera/aligned_depth_to_color/image_raw \
  -p info_topic:=/cam_high/camera/color/camera_info
```

On first successful detection, logs should say `Truck acquired; ... switching to KCF`. There should be **no OpenCV window** from this process.

## Inspect XYZ (terminal B)

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/setup.bash
ros2 topic echo /aloha/perception/detections
```

Expected example output format (illustrative only):

```json
[{"label":"truck","confidence":0.83,"cx_3d":0.02,"cy_3d":-0.04,"cz_3d":0.73,"frame_id":"camera_color_optical_frame","stamp":123.0,"source":"realsense","tracking_state":"TRACKING"}]
```

`[]` means no valid measurement for a frame.

## View annotated images (terminal C)

Option A (prefer for initial test):

```bash
ros2 run rqt_image_view rqt_image_view
# Select /aloha/perception/debug_image
```

Option B (separate viewer process with cv2 on its own main thread):

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/setup.bash
ros2 run aloha_perception tracking_viewer
# Press q to close
```

`debug_image` is a sensor-data QoS publisher (best-effort, volatile, depth 5); it does not block on old frames. If a viewer freezes while XYZ continues changing, debug the viewer without changing perception.

## Reset target and re-identify

```bash
ros2 service call /realsense_detection/reset_target std_srvs/srv/Trigger '{}'
```

Use reset if truck appearance changes substantially or tracking drifts without reporting failure.

## Adjustable ROS parameters

- `confidence` (default 0.40): initial YOLO threshold
- `max_lost_frames` (3): consecutive KCF update failures before reacquisition
- `max_depth_m` (3.0): valid depth bound in meters
- `max_position_jump_m` (0.30): reject abrupt coordinate jumps in meters
- `synchronizer_slop` (0.06): max RGB/depth timestamp gap in seconds
- `debug_every_n_frames` (1): preview publishing decimation, e.g. `2` halves preview CPU/network usage

The filter currently uses a fixed measurement noise estimate of 0.015 m as a starting assumption. Tune it only using observed physical measurement noise. The output frequency is limited by incoming synchronized pairs and KCF processing; benchmark on the actual workstation.

## Benchmarks

```bash
ros2 topic hz /cam_high/camera/color/image_raw
ros2 topic hz /aloha/perception/detections
ros2 topic hz /aloha/perception/debug_image
```

Check whether the tracked bounding box continues following the actual truck, including lateral motion, partial occlusion, and disappearance. Measure stationary XYZ jitter and moving-object latency separately. No automatic position/identity validator is included.

## Cleanup

Press Ctrl+C in the detector and monitoring terminals, close the viewer, and leave other lab processes alone. Because it lives in `~/aloha_tracking_ws`, removing this test workspace later will not alter Anthony's or Vignesh's source trees.
