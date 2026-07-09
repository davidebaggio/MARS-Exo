# exo_head_slam VGGT-Omega Branch

This branch contains a ROS 2 `ament_python` RGB-D SLAM pipeline that uses the `facebookresearch/vggt-omega` submodule for deep-learning-based camera alignment and dense depth completion. It is built around two synchronized RGB-D streams: a head-mounted camera and an exoskeleton-mounted camera.

The package assumes an upstream camera stack already publishes aligned RGB, depth, and camera calibration topics. It does not capture camera data or align depth to color.

## Pipeline Overview

The pipeline:

1. Filters head and exo depth streams.
2. Masks dynamic objects from RGB-D frames.
3. Runs ORB-SLAM3 RGB-D-inertial tracking on the exo stream.
4. Runs a VGGT-Omega extrinsic solver on synchronized head/exo images.
5. Accumulates a dense global exo cloud in the ORB odom frame and publishes the combined depth / debug point clouds for RViz.

Data flow:

```text
Head RGB + depth -> depth_preprocessor -> semantic_masker ---------.
                                                                   |
Exo RGB + depth  -> depth_preprocessor -> semantic_masker -> ORB-SLAM3 -> /exo/odom + TF
                                            |
                                            '-> Dense global cloud accumulator -> /orbslam/cloud_map

Exo RGB + depth  -> depth_preprocessor -> semantic_masker ---------.
                                                                   |
                                                                   '-> VGGT-Omega solver -

VGGT-Omega solver -> /head/combined/depth_raw
VGGT-Omega solver -> /exo/combined/depth_raw
VGGT-Omega solver -> /vggt/combined_pointcloud
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
| `extrinsic_solver` | `exo_head_slam/extrinsic_solver_node.py` | Loads VGGT-Omega, estimates `exo_link -> head_link`, publishes combined depth and point clouds |
| `pointcloud_publisher` | `exo_head_slam/pointcloud_publisher_node.py` | Optional debug `PointCloud2` output from combined depth |

Important files:

| Path | Purpose |
|---|---|
| `launch/main_pipeline_launch.py` | Top-level launch file |
| `launch/orbslam3_exo_launch.py` | ORB-SLAM3 odometry + dense global map launch |
| `config/head.yaml` | Head camera topics and masking/filtering parameters |
| `config/exo.yaml` | Exo camera topics and masking/filtering parameters |
| `config/orbslam3_exo.yaml` | ORB-SLAM3 settings template |
| `config/common.yaml` | VGGT-Omega solver parameters, confidence gates, metrics |
| `pipeline.md` | More detailed architecture notes |
| `plot_metrics.py` | Plots solver and depth-estimation metrics |
| `tests/test_depth_blend.py` | Minimal self-check for depth hole-fill behavior |

## Dependencies

Base requirements:

- ROS 2 with `colcon`
- Python 3
- `rclpy`, `sensor_msgs`, `geometry_msgs`, `tf2_ros`, `cv_bridge`, `message_filters`
- NumPy, OpenCV, SciPy
- PyTorch with CUDA
- `ultralytics` for YOLOv8 segmentation
- ORB-SLAM3 + Pangolin built under `third_party/ORB_SLAM3` and `third_party/Pangolin`

VGGT-Omega-specific requirements:

- The `vggt-omega` submodule from `https://github.com/facebookresearch/vggt-omega.git`
- A local `vggt_omega_1b_512.pt` checkpoint in the repository root

Initialize submodules with:

```bash
git submodule update --init --recursive
```

The solver searches for the vendored `vggt-omega/` directory from the package root or its parents, then loads `vggt_omega_1b_512.pt` from the repository root.

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
python -m pip install -e vggt-omega
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
  orbslam3_vocabulary_path:=third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  orbslam3_settings_path:=config/orbslam3_exo.yaml \
  dense_map_voxel_size:=0.03 \
  dense_map_max_points:=250000
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

VGGT-Omega outputs:

- `/head/combined/depth_raw`
- `/exo/combined/depth_raw`
- `/vggt/combined_pointcloud`
- TF `exo_link -> head_link`
- TF `exo_link -> vggt_world`

ORB-SLAM3 outputs:

- `/exo/odom`
- TF `odom -> exo_link`
- `/orbslam/cloud_map`

Debug outputs:

- `/head/debug_pcl`
- `/exo/debug_pcl`

## Configuration

Runtime behavior is YAML-driven:

- Camera topics and depth filter parameters: `config/head.yaml`, `config/exo.yaml`
- Semantic mask classes and confidence thresholds: `masker.*` parameters
- ORB-SLAM3 settings template: `config/orbslam3_exo.yaml`
- VGGT-Omega solver parameters: `extrinsic_solver` in `config/common.yaml`

Important solver parameters:

- `sliding_window_size`
- `tf_filter_alpha`
- `depth_conf_mode`
- `depth_conf_percentile`
- `depth_conf_absolute`
- `conf_gate_combine`
- `conf_filter_pointcloud`
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

If the VGGT-Omega solver cannot start, check that the `vggt-omega` submodule is initialized and installed, CUDA is available, and `vggt_omega_1b_512.pt` exists in the repository root.

If masks are empty, check `ultralytics` and the YOLO model path in the YAML config.

If ORB-SLAM3 fails to start, check `ORB_SLAM3_ROOT`, `third_party/ORB_SLAM3/lib/libORB_SLAM3.so`, and the vocabulary/settings paths passed to launch.

If running bags, use `use_sim_time:=true` and verify `/clock` is being published.
