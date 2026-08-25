# exo_head_slam VGGT-Omega Branch

This branch contains a ROS 2 `ament_python` RGB-D SLAM pipeline that uses the `facebookresearch/vggt-omega` submodule for deep-learning-based camera alignment and dense depth completion. It is built around two synchronized RGB-D streams: a head-mounted camera and an exoskeleton-mounted camera.

The package assumes an upstream camera stack already publishes aligned RGB, depth, and camera calibration topics. It does not capture camera data or align depth to color.

## Pipeline Overview

The pipeline:

1. Filters head and exo depth streams.
2. Masks dynamic objects from RGB-D frames.
3. Runs RTAB-Map RGB-D odometry / mapping on the exo stream.
4. Runs a VGGT-Omega extrinsic solver on synchronized head/exo images.
5. Publishes combined depth images and point clouds for RViz visualization.

Data flow:

```text
Head RGB + depth -> depth_preprocessor -> semantic_masker ---------.
                                                                   |
Exo RGB + depth  -> depth_preprocessor -> semantic_masker -> RTAB-Map -> TF tree
                                            |                      |
                                            '-> VGGT-Omega solver -'

VGGT-Omega solver -> /head/combined/depth_raw
VGGT-Omega solver -> /exo/combined/depth_raw
VGGT-Omega solver -> /vggt/combined_pointcloud
```

The expected TF structure is:

```text
map -> odom -> exo_link -> head_link
```

`/vggt/combined_pointcloud` is transformed into `exo_link` before publication,
so its geometry stays consistent when the VGGT sliding-window world gauge changes.

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
| `launch/rtabmap_agents_launch.py` | RTAB-Map odometry / mapping launch |
| `config/head.yaml` | Head camera topics and masking/filtering parameters |
| `config/exo.yaml` | Exo camera topics and RTAB-Map parameters |
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
- `rtabmap_slam` and `rtabmap_odom` for exo RGB-D odometry / mapping

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

`run.sh` builds the package, sources the workspace, fixes Python shebangs, starts the pipeline, plays the bag with `/clock`, writes metrics under `metrics/pipeline/`, and opens `rviz/pipeline.rviz`. When visible-cloud ground truth is present, it records the VGGT cloud, ground-truth cloud, and odometry without evaluating them live. On shutdown it prints the `cloud_map_evaluation_launch.py` command that computes the CSV offline.

When the bag contains `/ground_truth/global_map`, `run.sh` automatically enables
the simulated exoskeleton dataset profile: metric `32FC1` depth, isolated GT TF
topics, one-shot playback, extrinsic GT metrics, and trajectory/map evaluation.
Benchmark JSON and paired trajectory CSV files are written under `metrics/eval/`.
RViz also shows `/ground_truth/global_map` in green, aligned to the estimated
map at the first GT camera pose. GT odometry and a conflict-free `gt_*` TF tree
are enabled; visible cloud/map displays are available disabled.

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

The VGGT point cloud uses `exo_link` as its message frame.

Debug outputs:

- `/head/debug_pcl`
- `/exo/debug_pcl`

RTAB-Map outputs include odometry, `/map`, and RTAB-Map cloud/map topics depending on installed RTAB-Map packages and launch parameters.

## Configuration

Runtime behavior is YAML-driven:

- Camera topics and depth filter parameters: `config/head.yaml`, `config/exo.yaml`
- Semantic mask classes and confidence thresholds: `masker.*` parameters
- RTAB-Map parameters: `exo_rtabmap` in `config/exo.yaml`
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
python tests/test_benchmark_evaluator.py
```

## Troubleshooting

If Python imports fail from ROS nodes, rebuild with `make build` and source `install/setup.bash`.

If the VGGT-Omega solver cannot start, check that the `vggt-omega` submodule is initialized and installed, CUDA is available, and `vggt_omega_1b_512.pt` exists in the repository root.

If masks are empty, check `ultralytics` and the YOLO model path in the YAML config.

If RTAB-Map is skipped, install/source `rtabmap_slam` and `rtabmap_odom`.

If running bags, use `use_sim_time:=true` and verify `/clock` is being published.
