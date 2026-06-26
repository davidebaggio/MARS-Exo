# GEMINI.md - exo_head_slam

## Project Overview
`exo_head_slam` is a multi-agent RGB-D SLAM and volumetric fusion pipeline designed for wearable exoskeleton systems equipped with head-mounted cameras. It synchronizes two distinct RGB-D streams to produce a single, high-fidelity 3D map in a shared coordinate system.

The system decouples **pose tracking** (handled by RTAB-Map) from **volumetric reconstruction** (handled by NVIDIA Isaac ROS NVBlox), using a custom **Extrinsic Solver** to dynamically link the coordinate frames of the two cameras.

## Core Architecture

### 1. Depth Preprocessing (`depth_preprocessor_node.py`)
- **Action:** Sanitizes raw depth topics.
- **Features:** Removes invalid values (NaN/Inf), applies spatial smoothing, and temporal blending to stabilize frames for downstream tracking and fusion.

### 2. Semantic Masking (`semantic_masker_node.py`)
- **Action:** Filters dynamic objects from RGB and Depth streams.
- **Technology:** Uses `ultralytics` (YOLOv8) to identify humans and moving machinery, zeroing out corresponding pixels to prevent "ghosting" in the 3D map.

### 3. Pose Tracking (`rtabmap_agents_launch.py`)
- **Action:** Localization of the primary agent.
- **Logic:** RTAB-Map runs on the **exo camera** stream to publish the `map -> odom -> exo_link` transform (using visual odometry via `fallback_vo` and SLAM via `rtabmap`). Dense mapping is disabled within RTAB-Map to conserve resources.

### 4. Extrinsic Solver (`extrinsic_solver_node.py`)
- **Action:** Estimates the spatial relationship between cameras.
- **Logic:** Matches features between head and exo views using **LightGlue** (GPU) or **ORB** (CPU). Deprojects matches to 3D and solves for the rigid transform via RANSAC and SVD.
- **Output:** Broadcasts the `exo_link -> head_link` TF (configured via `exo_frame_id` and `head_frame_id`).

### 5. Volumetric Fusion (`nvblox_fusion_launch.py`)
- **Action:** Global 3D mapping.
- **Logic:** A single NVBlox node integrates masked depth from both cameras using the unified TF tree to generate a global TSDF-based mesh and 2D costmap.

## Technical Stack
- **Robotics:** ROS 2 (Humble/Iron/Jazzy), `rclpy`, `tf2_ros`.
- **Vision:** OpenCV, `cv_bridge`, `ultralytics`.
- **Calibration:** `message_filters` (Approximate Time Sync), 3D-to-3D rigid transform estimation.
- **Accelerated Computing:** NVIDIA Isaac ROS NVBlox, PyTorch (LightGlue).

## Building and Installation

### Dependencies
Ensure the following are installed in your ROS 2 workspace:
- `rtabmap_ros`
- `isaac_ros_nvblox`
- Python packages: `ultralytics`, `opencv-python`, `torch` (for LightGlue).

### Build Process
```bash
colcon build --packages-select exo_head_slam --symlink-install
source install/setup.bash
```

### Model Files
The semantic masker requires a YOLOv8 segmentation model. By default, it expects `yolov8n-seg.pt` in the repository root. Ensure this file is present before launching the pipeline.

## Running the Pipeline

### Automated Launch (Bag Playback)
The project includes a `run.sh` script that automates building, sourcing, and launching with a specific ROS 2 bag:
```bash
./run.sh /path/to/data.mcap
```

### Manual Pipeline Launch
To start the pipeline without bag playback (e.g., for live sensor data):
```bash
ros2 launch exo_head_slam main_pipeline_launch.py
```

### Debugging & Visualization
To visualize the raw alignment of cameras in the `map` frame before volumetric fusion, you can enable the PointCloud publishers:
```bash
ros2 launch exo_head_slam main_pipeline_launch.py publish_debug_pcl:=true
```
This will publish colored PointCloud2 messages to `/head/debug_pcl` and `/exo/debug_pcl`, transformed into the global `map` frame.

## Configuration
System behavior is defined across three YAML files in the `config/` directory:
- **`head.yaml`:** Topics and parameters specific to the head-mounted camera.
- **`exo.yaml`:** Topics and parameters specific to the exoskeleton-mounted camera.
- **`common.yaml`:** Shared settings including matcher selection (`orb` vs `lightglue`) and TF frame names.

## Development Conventions
- **Nodes:** All ROS 2 nodes are implemented in Python within `exo_head_slam/`.
- **Utilities:** Math and vision helper functions are centralized in `exo_head_slam/utils/`.
- **TFs:** The system expects a standard `map -> odom -> exo_link` chain provided by the tracking backend, with static TFs defining `exo_link -> exo_camera_link` and `head_link -> head_camera_link`, and the extrinsic solver publishing `exo_link -> head_link`.
