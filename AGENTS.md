# AGENTS.md

## Project

ROS 2 ament_python package — multi-agent RGB-D SLAM pipeline for head + exoskeleton cameras. Master thesis project. No tests, no CI.

## Build & Run

```bash
make build
source install/setup.bash
```

**Or use `run.sh`** — builds, sources, launches the pipeline, and plays a rosbag at 0.3x loop. Accepts an optional bag path argument.

### Critical: Python shebang fix

After `colcon build`, the generated shim scripts in `install/exo_head_slam/lib/exo_head_slam/` hardcode `#!/usr/bin/python3`. If using a conda or venv, rewrite the shebangs to `$(which python3)` or nodes will run in the system Python and miss dependencies. Both `make build` and `run.sh` do this automatically.

## Nodes (entry points in setup.py)

| Executable | Source | Purpose |
|---|---|---|
| `depth_preprocessor` | `depth_preprocessor_node.py` | Spatial + temporal depth filtering |
| `semantic_masker` | `semantic_masker_node.py` | YOLOv8-seg dynamic object removal |
| `extrinsic_solver` | `extrinsic_solver_node.py` | VGGT-1B joint extrinsic + depth estimation, with depth-head confidence gating |
| `pointcloud_publisher` | `pointcloud_publisher_node.py` | Debug per-camera XYZRGB PointCloud2 from RGB-D (behind `publish_debug_pcl` flag) |

## Launch

- **`launch/main_pipeline_launch.py`** — top-level entry point. Starts the custom nodes (depth_preprocessor x2, semantic_masker x2, extrinsic_solver, pointcloud_publisher x2 debug) + static TF publishers.
- **`launch/rtabmap_agents_launch.py`** — conditionally included by the top launch only if `rtabmap_slam` is found. Launches a single `rtabmap_slam` instance on the **exo** camera running **internal visual odometry** (no separate odom node). Publishes `map -> odom -> exo_link` TFs. Silently skipped if `rtabmap_slam` is missing.

## Configuration

All runtime parameters live in YAML, not in code:
- `config/head.yaml` — head camera pipeline (depth_preprocessor, semantic_masker)
- `config/exo.yaml` — exo camera pipeline (depth_preprocessor, semantic_masker, exo_rtabmap)
- `config/common.yaml` — extrinsic solver params (VGGT conf gating, metrics)

## Optional Dependencies

- `ultralytics>=8.0` (requirements.txt) — semantic masker falls back to empty mask if missing
- `torch` — required by the VGGT extrinsic solver (not in requirements.txt because of CUDA build specifics)
- `huggingface_hub` — required to download `facebook/VGGT-1B`
- `scipy` — extrinsic solver SE(3) math + GT error metrics
- `rtabmap_slam` ROS 2 package — exo camera SLAM, optional (pipeline runs without it; you lose `map -> odom` TF)

## Pipeline Data Flow

```
Head RGB + depth → depth_preprocessor → semantic_masker ─┐
                                                          ├─ extrinsic_solver (VGGT-1B) → TF exo_link→head_link, vggt_world
Exo RGB + depth  → depth_preprocessor → semantic_masker ─┘                                  ↘ combined depth fills, /vggt/combined_pointcloud

RTAB-Map (exo, internal VO) → TF map→odom→exo_link  → exo_rtabmap/cloud_map + /map (optional)
```

VGGT depth-head confidence (`depth_conf`, range `(1, +inf)`) gates:
- The `np.where` raw+VGGT depth combine — only fill raw-depth holes where VGGT is confident
- The published `/vggt/combined_pointcloud` — percentile- or absolute-threshold-filtered

## Gotchas

- `vggt/` is a vendored copy of the VGGT repo at the package root; the extrinsic solver walks up to 6 parent dirs to find it. Clone it there or its absence raises `RuntimeError`.
- `model.depth_head(...)` returns `(depth_map, depth_conf)`; the project does NOT use `model.point_head`.
- `build/`, `install/`, `log/` are colcon artifacts, gitignored
- `*.pt` and `*.engine` model files are gitignored — `yolov8n-seg.pt` must be placed in repo root manually
- Bag playback defaults to `data/rosbag2_2026_06_11-15_34_13/rosbag2_2026_06_11-15_34_13_0.mcap` (run.sh), played at 0.3x
- RGB and depth must already be published and aligned by an upstream camera stack; this package does not capture or align them