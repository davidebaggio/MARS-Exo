# exo_head_slam

`exo_head_slam` is a ROS 2 `ament_python` package for a dual RGB-D SLAM pipeline using a head-mounted camera and an exoskeleton-mounted camera. The package preprocesses depth streams, masks dynamic objects, estimates the camera-to-camera extrinsic transform, tracks motion, and fuses both RGB-D streams into one volumetric map.

The project is topic-driven: RGB, depth, and camera calibration are expected to come from an upstream camera stack. This package does not acquire images from hardware or align depth to color.

## Pipeline Overview

The pipeline has five main stages:

1. Depth preprocessing for each camera stream.
2. Semantic masking to remove dynamic objects from RGB and depth.
3. Visual odometry / SLAM for global pose tracking.
4. Online extrinsic estimation between the head and exo camera frames.
5. TSDF-based volumetric fusion of both camera streams.

Data flow:

```text
Head RGB + depth -> depth_preprocessor -> semantic_masker ---------.
                                                                   |
Exo RGB + depth  -> depth_preprocessor -> semantic_masker -> VO/SLAM -> TF tree
                                            |                      |
                                            '-> extrinsic_solver --'

Masked head/exo RGB-D + TF tree -> nvblox_node -> mesh, pointcloud, costmap
```

The intended TF tree is:

```text
map -> odom -> exo_link -> head_link
                 |            |
                 v            v
           exo_camera_link  head_camera_link
```

`map -> odom` and `odom -> exo_link` come from the pose-tracking path. `exo_link -> head_link` is produced by the extrinsic solver. Static identity transforms connect each link frame to its camera link frame.

## Package Contents

Core ROS nodes live in `exo_head_slam/`:

| Executable | Source | Purpose |
|---|---|---|
| `depth_preprocessor` | `depth_preprocessor_node.py` | Filters and republishes depth images |
| `semantic_masker` | `semantic_masker_node.py` | Applies YOLO segmentation masks to RGB-D streams |
| `extrinsic_solver` | `extrinsic_solver_node.py` | Estimates and broadcasts the head/exo SE(3) transform |
| `fallback_vo` | `fallback_vo_node.py` | Python RGB-D visual odometry fallback |
| `pointcloud_publisher` | `pointcloud_publisher_node.py` | Optional debug `PointCloud2` publisher |
| `nvblox_node` | `nvblox_node.py` | Integrates both RGB-D streams into a TSDF map |

Launch files:

| File | Purpose |
|---|---|
| `launch/main_pipeline_launch.py` | Top-level pipeline launch |
| `launch/rtabmap_agents_launch.py` | Exo visual odometry + RTAB-Map SLAM launch |
| `launch/nvblox_fusion_launch.py` | Unified nvblox fusion launch |

Configuration files:

| File | Purpose |
|---|---|
| `config/head.yaml` | Head camera topics and per-node parameters |
| `config/exo.yaml` | Exo camera topics and per-node parameters |
| `config/common.yaml` | Shared extrinsic solver parameters |

## Requirements

Base runtime:

- ROS 2 with `colcon`
- Python 3
- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `nav_msgs`
- `tf2_ros`
- `cv_bridge`
- `message_filters`
- NumPy
- OpenCV
- SciPy
- PyTorch with CUDA support for nvblox and optional LightGlue acceleration

Optional runtime dependencies:

- `ultralytics` for YOLOv8 segmentation. If unavailable, the semantic masker publishes unmasked images.
- `lightglue` and SuperPoint dependencies for feature matching. If unavailable, matching falls back to ORB.
- `rtabmap_slam` for RTAB-Map SLAM. If unavailable, the top-level launch skips RTAB-Map.
- `nvblox_torch` for TSDF fusion.

Model files such as `yolov8n-seg.pt` are not tracked in git. Place them in the repository root or update `masker.model_path` in the YAML configuration.

## Build

From the repository root:

```bash
make build
source install/setup.bash
```

Equivalent manual build:

```bash
colcon build --packages-select exo_head_slam --symlink-install
source install/setup.bash
```

If using conda or a virtual environment, the generated ROS shim scripts may hardcode `/usr/bin/python3`. `make build` rewrites those shebangs to the active `python3`. If building manually, apply the same fix:

```bash
find install/exo_head_slam/lib/exo_head_slam -type f -executable \
  -exec sed -i "1s|^#!.*python.*|#!$(which python3)|" {} \;
```

## Run

Launch the full pipeline:

```bash
ros2 launch exo_head_slam main_pipeline_launch.py
```

Common launch arguments:

```bash
ros2 launch exo_head_slam main_pipeline_launch.py \
  use_sim_time:=true \
  publish_debug_pcl:=true \
  global_frame:=odom
```

Use `run.sh` to build, source the workspace, launch the pipeline, play a rosbag, and start RViz:

```bash
./run.sh
```

Pass a bag path explicitly:

```bash
./run.sh /path/to/bag/file.mcap
```

The script plays the bag in a loop with simulated time and opens `rviz/pipeline.rviz`.

## Input Topics

Default input topics are configured in `config/head.yaml` and `config/exo.yaml`.

Head camera:

- `/camera/head/color/image_raw`
- `/camera/head/aligned_depth_to_color/image_raw`
- `/camera/head/color/camera_info`

Exo camera:

- `/camera/exo/color/image_raw`
- `/camera/exo/aligned_depth_to_color/image_raw`
- `/camera/exo/color/camera_info`

Depth images are expected to be aligned to the color camera. Camera info must match the RGB-D stream used by each node.

## Output Topics

Filtered depth:

- `/head/filtered/depth_raw`
- `/exo/filtered/depth_raw`

Masked RGB-D:

- `/head/masked/image_raw`
- `/head/masked/depth_raw`
- `/exo/masked/image_raw`
- `/exo/masked/depth_raw`

Odometry and mapping:

- `/exo/odom`
- `/tf`
- `/tf_static`
- `/nvblox/mesh`
- `/nvblox/pointcloud`
- `/nvblox/costmap`

Optional debug point clouds:

- `/head/debug_pcl`
- `/exo/debug_pcl`

## Configuration

Most runtime behavior is controlled through YAML:

- Change camera topics in `config/head.yaml` and `config/exo.yaml`.
- Change depth filtering parameters under `depth_filter.*`.
- Change YOLO model path, confidence threshold, and dynamic classes under `masker.*`.
- Change feature matching and extrinsic solver thresholds in `config/common.yaml`.
- Change voxel size, integration distance, mesh update period, and costmap settings under `head_nvblox` / `exo_nvblox`.

The top-level launch loads the YAML files from the installed package share directory. Rebuild after changing configuration files:

```bash
make build
source install/setup.bash
```

## Node Behavior

### Depth Preprocessor

Each depth preprocessor subscribes to one depth image topic, converts depth to meters, removes invalid values, applies optional spatial filtering, applies optional temporal blending, and republishes a `32FC1` depth image.

### Semantic Masker

Each semantic masker synchronizes RGB and filtered depth messages, runs YOLO segmentation when available, masks configured dynamic classes, and republishes RGB-D messages with masked pixels zeroed. If YOLO or the model file is missing, it publishes empty masks.

### Extrinsic Solver

The extrinsic solver synchronizes the head and exo masked RGB-D streams, matches image features using LightGlue or ORB, deprojects matches into 3D using camera intrinsics, estimates an SE(3) transform with RANSAC, filters unstable estimates, and broadcasts the configured head/exo transform.

It can also write extrinsic metrics to CSV when `metrics_enabled` is true.

### Fallback Visual Odometry

The fallback VO node estimates frame-to-frame RGB-D motion from the exo stream using the configured matcher and publishes odometry and optionally TF. RTAB-Map consumes this odometry when `rtabmap_slam` is installed.

### NVBlox Fusion

The nvblox node synchronizes each masked RGB-D stream with camera info, looks up the pose for the incoming frame, integrates depth and color into a TSDF map, and periodically publishes mesh, point cloud, and costmap outputs.

## RViz

The repository includes an RViz configuration at:

```text
rviz/pipeline.rviz
```

The `run.sh` script launches it automatically. To open it manually:

```bash
rviz2 -d "$(ros2 pkg prefix exo_head_slam)/share/exo_head_slam/rviz/pipeline.rviz"
```

## Repository Notes

- `build/`, `install/`, and `log/` are generated by colcon.
- Large model files such as `*.pt` and `*.engine` should stay outside git.
- `pipeline.md` contains a more detailed architecture discussion.
- `vggt/`, `vggt-omega/`, and `eVGGT/` are research/vendor directories and are not required for the core ROS package entry points.
- There are currently no project tests or CI.

## Troubleshooting

If ROS nodes fail to import Python packages, rebuild with `make build` so generated shim shebangs use the active environment.

If masked images are identical to input images, check that `ultralytics` is installed and the configured YOLO model path exists.

If LightGlue is requested but unavailable, the matcher falls back to ORB. This is expected behavior.

If nvblox does not publish a map, check CUDA/PyTorch availability, `nvblox_torch` installation, camera info topics, and TF lookup failures in the node logs.

If transforms are missing, inspect:

```bash
ros2 run tf2_tools view_frames
ros2 topic echo /tf
ros2 topic echo /tf_static
```

If running from a bag, use `use_sim_time:=true` and ensure `/clock` is published by bag playback.
