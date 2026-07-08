# exo_head_slam VGGT Branch

This branch contains a ROS 2 `ament_python` RGB-D SLAM pipeline that uses the `facebookresearch/vggt` submodule for deep-learning-based camera alignment and dense depth completion. It is built around two synchronized RGB-D streams: a head-mounted camera and an exoskeleton-mounted camera.

The package assumes an upstream camera stack already publishes aligned RGB, depth, and camera calibration topics. It does not capture camera data or align depth to color.

## Pipeline Overview

The pipeline:

1. Filters head and exo depth streams.
2. Masks dynamic objects from RGB-D frames.
3. Runs RTAB-Map RGB-D odometry / mapping on the exo stream.
4. Runs a VGGT-based extrinsic solver on synchronized head/exo images.
5. Publishes combined depth images and point clouds for RViz visualization.

Data flow:

```text
Head RGB + depth -> depth_preprocessor -> semantic_masker ---------.
                                                                   |
Exo RGB + depth  -> depth_preprocessor -> semantic_masker -> RTAB-Map -> TF tree
                                            |                      |
                                            '-> VGGT solver -------'

VGGT solver -> /head/combined/depth_raw
VGGT solver -> /exo/combined/depth_raw
VGGT solver -> /vggt/combined_pointcloud
```

The expected TF structure is:

```text
map -> odom -> exo_link -> head_link
                 |
                 v
              vggt_world
```

## Package Contents

Core nodes:

| Executable | Source | Purpose |
|---|---|---|
| `depth_preprocessor` | `exo_head_slam/depth_preprocessor_node.py` | Filters depth images and republishes `32FC1` depth |
| `semantic_masker` | `exo_head_slam/semantic_masker_node.py` | Removes configured dynamic classes from RGB-D frames |
| `extrinsic_solver` | `exo_head_slam/extrinsic_solver_node.py` | Loads VGGT-1B, estimates `exo_link -> head_link`, publishes combined depth and point clouds |
| `pointcloud_publisher` | `exo_head_slam/pointcloud_publisher_node.py` | Optional debug `PointCloud2` output from combined depth |

Important files:

| Path | Purpose |
|---|---|
| `launch/main_pipeline_launch.py` | Top-level launch file |
| `launch/rtabmap_agents_launch.py` | RTAB-Map odometry / mapping launch |
| `config/head.yaml` | Head camera topics and masking/filtering parameters |
| `config/exo.yaml` | Exo camera topics and RTAB-Map parameters |
| `config/common.yaml` | VGGT solver parameters, confidence gates, metrics |
| `pipeline.md` | More detailed architecture notes |
| `plot_metrics.py` | Plots solver and depth-estimation metrics |
| `tests/test_depth_blend.py` | Minimal self-check for depth blending behavior |

## Dependencies

Base requirements:

- ROS 2 with `colcon`
- Python 3
- `rclpy`, `sensor_msgs`, `geometry_msgs`, `tf2_ros`, `cv_bridge`, `message_filters`
- NumPy, OpenCV, SciPy
- PyTorch
- `ultralytics` for YOLOv8 segmentation
- `rtabmap_slam` and `rtabmap_odom` for exo RGB-D odometry / mapping

VGGT-specific requirement:

- The `vggt` submodule from `https://github.com/facebookresearch/vggt.git`

Initialize submodules with:

```bash
git submodule update --init --recursive
```

`setup.sh` creates a conda environment, initializes the `vggt` submodule, installs Python dependencies, installs VGGT editable, and builds the ROS package.

## Setup

Using the provided setup script:

```bash
./setup.sh
conda activate exo_head_slam
source install/setup.bash
```

Manual build:

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install -e vggt
make build
source install/setup.bash
```

After `colcon build`, generated ROS shim scripts may hardcode `/usr/bin/python3`. `make build` rewrites them to the active `python3`, which is required when running inside conda or a virtual environment.

## Run

Launch the pipeline:

```bash
ros2 launch exo_head_slam main_pipeline_launch.py
```

Common launch arguments:

```bash
ros2 launch exo_head_slam main_pipeline_launch.py \
  use_sim_time:=true \
  publish_debug_pcl:=true \
  metrics_csv_path:=metrics/pipeline/run.csv \
  use_imu:=false
```

Run against the default bag and open RViz:

```bash
./run.sh
```

Run against a specific bag:

```bash
./run.sh /path/to/bag/file.mcap
```

`run.sh` builds the package, sources the workspace, fixes Python shebangs, starts the pipeline, plays the bag with `/clock`, writes metrics under `metrics/pipeline/`, and opens `rviz/pipeline.rviz`.

## Topics

Default inputs:

- `/camera/head/color/image_raw`
- `/camera/head/aligned_depth_to_color/image_raw`
- `/camera/head/color/camera_info`
- `/camera/exo/color/image_raw`
- `/camera/exo/aligned_depth_to_color/image_raw`
- `/camera/exo/color/camera_info`

Intermediate outputs:

- `/head/filtered/depth_raw`
- `/exo/filtered/depth_raw`
- `/head/masked/image_raw`
- `/head/masked/depth_raw`
- `/exo/masked/image_raw`
- `/exo/masked/depth_raw`

VGGT outputs:

- `/head/combined/depth_raw`
- `/exo/combined/depth_raw`
- `/vggt/combined_pointcloud`
- TF `exo_link -> head_link`
- TF `exo_link -> vggt_world`

Debug outputs:

- `/head/debug_pcl`
- `/exo/debug_pcl`

RTAB-Map outputs include odometry, `/map`, and RTAB-Map cloud/map topics depending on installed RTAB-Map packages and launch parameters.

## Configuration

Runtime behavior is YAML-driven:

- Camera topics and depth filter parameters: `config/head.yaml`, `config/exo.yaml`
- Semantic mask classes and confidence thresholds: `masker.*` parameters
- RTAB-Map parameters: `exo_rtabmap` in `config/exo.yaml`
- VGGT solver parameters: `extrinsic_solver` in `config/common.yaml`

Important VGGT parameters:

- `sliding_window_size`
- `tf_filter_alpha`
- `depth_conf_mode`
- `depth_conf_percentile`
- `depth_conf_absolute`
- `conf_gate_combine`
- `conf_filter_pointcloud`
- `depth_blend_error_threshold`
- `depth_blend_camera_weight`
- `metrics_enabled`
- `metrics_csv_path`

Rebuild after changing installed launch/config files:

```bash
make build
source install/setup.bash
```

## Metrics

When metrics are enabled, the extrinsic solver writes CSV rows with solver status, depth scale, depth error, confidence percentiles, and optional ground-truth transform error. Plot the latest metrics with:

```bash
python plot_metrics.py
```

## Tests

Run the minimal depth-blending self-check directly:

```bash
python tests/test_depth_blend.py
```

## Troubleshooting

If Python imports fail from ROS nodes, rebuild with `make build` and source `install/setup.bash`.

If the VGGT solver cannot start, check that the `vggt` submodule is initialized and installed, PyTorch is available, and the model weights can be loaded.

If masks are empty, check `ultralytics` and the YOLO model path in the YAML config.

If RTAB-Map is skipped, install/source `rtabmap_slam` and `rtabmap_odom`.

If running bags, use `use_sim_time:=true` and verify `/clock` is being published.
