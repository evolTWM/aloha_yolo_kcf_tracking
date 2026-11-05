# ALOHA Object Detection and 3D Localization

>ROS 2 perception package for detecting toy vehicles with YOLO and estimating their 3D positions using synchronized Intel RealSense RGB and aligned depth images.
>
>The pipeline converts camera-relative measurements into the left robot base frame using a saved extrinsic calibration, applies an exponential moving average (EMA) filter, and publishes workspace coordinates with the table center as the XY origin. An annotated image displays the filtered workspace XYZ values in meters.
>
> **Current status**: YOLO detection, coordinate transformation, EMA smoothing, and the workspace overlay are implemented. KCF tracking and Kalman filtering are not used. They may be considered later if needed.
>
>The reported point represents a depth sample near the detection-box center; it is not necessarily the object’s geometric center or a validated grasp target.

## Requirements

- Ubuntu 22.04 with ROS 2 Humble
- A configured ALOHA/Interbotix workspace and RealSense ROS 2 driver
- An overhead RealSense camera publishing RGB, aligned depth, and CameraInfo
- Python dependencies: OpenCV, NumPy, PyYAML, CV Bridge, message_filters, PyTorch, and Ultralytics
- YOLO weights (`yolov8m.pt`) accessible to the detector

**ROS package:** `aloha_tracking_perception`
**Executables:** `detector`, `yolo_to_robot`

The bundled `calibration_candidate.yaml` and `lab_workspace.yaml` describe
the lab's camera placement and workspace origin. Other setups must supply
their own calibration and workspace origin.

## 1. Clone and build

These commands use the lab's workspace layout. Adjust the paths if your
Interbotix workspace is elsewhere.

```bash
mkdir -p ~/interbotix_ws/src/aloha_tracking_ws
cd ~/interbotix_ws/src/aloha_tracking_ws

git clone https://github.com/evolTWM/aloha_yolo_kcf_tracking.git

unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH

source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/local_setup.bash

/usr/bin/colcon build \
  --base-paths aloha_yolo_kcf_tracking/aloha_perception \
  --packages-select aloha_tracking_perception

source install/local_setup.bash

ros2 pkg prefix aloha_tracking_perception
ros2 pkg executables aloha_tracking_perception
```

The package prefix should point inside
`~/interbotix_ws/src/aloha_tracking_ws/install/`.

The commands below use `local_setup.bash` to source the workspaces in an
explicit order and avoid the lab's existing workspace-chain conflict.

Place `yolov8m.pt` in `~/interbotix_ws/src/aloha_tracking_ws/`, or change the
detector's `model` parameter to its actual path.

This assumes dependencies from the requirements section are installed; cloning and building alone does not install PyTorch or Ultralytics.

## 2. Run the system (four terminals)

### Terminal 1 - ALOHA bringup

```bash
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH

source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/local_setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/local_setup.bash

ros2 launch aloha aloha_bringup.launch.py
```

> Leave bringup running and wait for the arm drivers and overhead camera to initialize. If bringup is already running, use that instance.

### Terminal 2 - YOLO Detector

```bash
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH

source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/local_setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/local_setup.bash

ros2 run aloha_tracking_perception detector \
  --ros-args \
  -p model:="$HOME/interbotix_ws/src/aloha_tracking_ws/yolov8m.pt"
```

> The detector reads synchronized overhead RGB and aligned depth images and publishes camera-relative detections. Leave this terminal running.

### Terminal 3 - Coordinate Transformation & EMA Filter

```bash
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH

source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/local_setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/local_setup.bash

ros2 run aloha_tracking_perception yolo_to_robot
```

> This node loads the saved calibration, transforms detections into robot-base coordinates, applies EMA smoothing, and publishes table-centered workspace coordinates. Leave it running alongside the detector.

### Terminal 4 - Image viewer

```bash
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH

source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/local_setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/local_setup.bash

ros2 run rqt_image_view rqt_image_view
```

> Select /aloha/perception/debug_image without the /compressed suffix. With a valid detection, the overlay shows the filtered workspace XYZ values. XY is relative to the saved table center; Z remains relative to the left robot base

## 3. Read the coordinates

In a fifth terminal:

```bash
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH

source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/local_setup.bash
source ~/interbotix_ws/src/aloha_tracking_ws/install/local_setup.bash

ros2 topic echo /aloha/perception/detections_workspace_filtered \
  --field data
```

The topic publishes a `std_msgs/String` containing a JSON array.
Each detection includes its label, confidence, filtered `x`, `y`, `z`,
frame identifier, and source timestamp. Values are in meters.

|  |  |
|---|---|
| /aloha/perception/detections | Raw camera optical coordinates (cx_3d, cy_3d, cz_3d) |
| /aloha/perception/detections_base | Unfiltered robot-base XYZ |
| /aloha/perception/detections_base_filtered | EMA-filtered robot-base XYZ |
| /aloha/perception/detections_workspace_filtered | Filtered XYZ with a table-centered XY origin |

Workspace coordinates preserve the robot-base axis directions.
Only the XY origin is shifted; Z is not tabletop-relative.

Delete the entire repeated shared-memory/USB reset sequence. Those commands are machine-specific recovery actions, and the broad shared-memory deletion and controller reset should not be routine startup instructions.

1. **Clear ROS 2 Shared Memory & Restart Daemon:**
   ```bash
   sudo rm -rf /dev/shm/*
   ros2 daemon stop
   ros2 daemon start

2. **Hard-Reset USB Controllers & Verify Devices:**
   ```bash
   sudo bash -c '
   echo -n 0000:00:14.0 > /sys/bus/pci/drivers/xhci_hcd/unbind
   sleep 5
   echo -n 0000:00:14.0 > /sys/bus/pci/drivers/xhci_hcd/bind
   '
   sleep 10

   # Sanity check to confirm the USB devices are visible again
   lsusb | grep 0403
   ls -l /dev/ttyDXL_*
   ```

## Current implementation

1. YOLO detects cars, trucks, and trains in the RGB image.
2. A median of valid aligned-depth pixels near each detection-box center provides depth.
3. Camera intrinsics convert the pixel and depth into camera optical XYZ.

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

## Credits
* Dr. Mingyu Cai (REAL Lab Director)
* Sashwat's Work on Kinova: https://github.com/Shashy2305/KinovaARM
