# GEMINI.md - exo_head_slam

## Project Overview
`exo_head_slam` is a multi-agent RGB-D SLAM pipeline for a wearable exoskeleton system equipped with a head-mounted camera. It synchronizes two distinct RGB-D streams and uses a deep geometric transformer (VGGT-1B) to estimate extrinsics and fill depth gaps, while RTAB-Map handles visual SLAM on the exoskeleton camera.

The system decouples **map building / loop closure** (handled by RTAB-Map running internally on the exo camera) from **inter-camera calibration** (handled by the VGGT-based Extrinsic Solver).

## Core Architecture

### 1. Depth Preprocessing (`depth_preprocessor_node.py`)
- Sanitizes raw depth topics: removes invalid values (NaN/Inf), applies spatial bilateral smoothing, and EMA temporal blending to stabilize frames.

### 2. Semantic Masking (`semantic_masker_node.py`)
- Filters dynamic objects from RGB and Depth streams using `ultralytics` (YOLOv8-seg). Pixels classified as dynamic are zeroed out to prevent "ghosting" in the map.

### 3. Pose Tracking & SLAM (`rtabmap_agents_launch.py`)
- `rgbd_odometry` runs on the **exo camera** stream and publishes dynamic `odom -> exo_link`.
- `map -> odom` is a static identity transform; `rtabmap_slam` consumes `/exo_rtabmap/odom` and does not publish map TF.
- Occupancy grid (`/map`) and `/exo_rtabmap/cloud_map` are published when `RGBD/CreateOccupancyGrid`/`Grid/FromDepth` are `true`.

### 4. Extrinsic Solver (`extrinsic_solver_node.py`)
- Estimates the spatial relationship between the head and exoskeleton cameras using **VGGT-1B** (`facebook/VGGT-1B`).
- Forwards a sliding window of masked head+exo image pairs through the VGGT aggregator, camera head, and depth head. Predicted depths are scale-aligned to metric depth via per-frame median ratio and merged with raw depth to fill holes.
- **VGGT depth-head confidence** (`depth_conf`, returned alongside `depth_map`, range `(1, +inf)`) gates:
  - the `np.where` raw+VGGT depth combine — raw-depth holes are filled with VGGT depth only where `depth_conf` is high (percentile or absolute threshold),
  - the published `/vggt/combined_pointcloud` — low-confidence points are dropped.
- Broadcasts `exo_link -> head_link` and `exo_link -> vggt_world` TFs after distance / Z-height / jump / EMA safety checks.
- Optional metrics CSV: scale, depth RMSE/MAE on confident pixels, conf percentiles (p50/p95), GT error (when `gt_parent_frame`/`gt_child_frame` set).

### 5. Debug PointCloud Publishers (`pointcloud_publisher_node.py`)
- Optional per-camera XYZRGB PointCloud2 in the local camera frame, gated by the `publish_debug_pcl` launch arg.

## Technical Stack
- **Robotics:** ROS 2 (Jazzy), `rclpy`, `tf2_ros`.
- **Vision:** OpenCV, `cv_bridge`, `ultralytics`.
- **Calibration:** `message_filters` (Approximate Time Sync), VGGT-1B transformer, scipy SE(3) math.
- **Accelerated Computing:** PyTorch (CUDA, bfloat16 on Ampere+).

## Building and Installation

### Dependencies
- `rtabmap_ros` (optional: when absent, the launch file silently skips RTAB-Map and you lose `map -> odom` TF).
- Python packages: `ultralytics>=8.0`, `scipy`, `torch`, `huggingface_hub`, `cv_bridge`.

### Build Process
```bash
colcon build --packages-select exo_head_slam --symlink-install
source install/setup.bash
```
(`make build` does both, plus the shebang fix described in `AGENTS.md`.)

### Model Files
- `yolov8n-seg.pt` — place in repository root (gitignored). Semantic masker falls back to an empty mask if missing.
- `facebook/VGGT-1B` — downloaded automatically by `huggingface_hub` on first run.

## Running the Pipeline

### Automated Launch (Bag Playback)
```bash
./run.sh /path/to/data.mcap
```
Builds, sources, launches `main_pipeline_launch.py` with `use_sim_time:=true`, plays the bag at 0.3x, opens RViz. Default bag: `data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap`.

### Manual Pipeline Launch
```bash
ros2 launch exo_head_slam main_pipeline_launch.py
```

### Debug Visualization
```bash
ros2 launch exo_head_slam main_pipeline_launch.py publish_debug_pcl:=true
```
Publishes `/head/debug_pcl` and `/exo/debug_pcl` (PointCloud2 in local camera frames) plus the VGGT `vggt_world`-frame `/vggt/combined_pointcloud`.

## Configuration
- **`head.yaml`** — depth_preprocessor and semantic_masker for the head camera.
- **`exo.yaml`** — depth_preprocessor, semantic_masker, and the shared exo RTAB-Map / RGB-D odometry parameter block.
- **`common.yaml`** — `extrinsic_solver` parameters: frame ids, sliding window, TF EMA, **VGGT confidence gating** (`depth_conf_mode`, `depth_conf_percentile`, `depth_conf_absolute`, `conf_gate_combine`, `conf_filter_pointcloud`), and metrics settings.

## Development Conventions
- **Nodes:** All ROS 2 nodes are Python classes in `exo_head_slam/`.
- **Utilities:** `exo_head_slam/utils/vision_utils.py` holds the `apply_semantic_mask` helper used by `semantic_masker_node`.
- **TFs:** static `map -> odom` comes from the launch file, dynamic `odom -> exo_link` comes from `rgbd_odometry`, static camera TFs publish `exo_link -> exo_camera_link` and `head_link -> head_camera_link`, and the extrinsic solver publishes `exo_link -> head_link` plus `exo_link -> vggt_world`.
- **Odometry source:** `rtabmap_odom/rgbd_odometry` publishes `odom -> exo_link`; the old static `odom -> exo_link` fallback was removed.
