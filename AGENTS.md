# AGENTS.md

## Project

ROS 2 ament_python package — multi-agent RGB-D SLAM pipeline for head + exoskeleton cameras. Master thesis project. No tests, no CI.

## Build & Run

```bash
make build
source install/setup.bash
```

**Or use `run.sh`** — builds, sources, launches the pipeline, and plays a rosbag at 0.1x loop. Accepts an optional bag path argument.

### Critical: Python shebang fix

After `colcon build`, the generated shim scripts in `install/exo_head_slam/lib/exo_head_slam/` hardcode `#!/usr/bin/python3`. If using a conda or venv, rewrite the shebangs to `$(which python3)` or nodes will run in the system Python and miss dependencies. Both `make build` and `run.sh` do this automatically.

## Nodes (entry points in setup.py)

| Executable | Source | Purpose |
|---|---|---|
| `depth_preprocessor` | `depth_preprocessor_node.py` | Spatial + temporal depth filtering |
| `semantic_masker` | `semantic_masker_node.py` | YOLOv8-seg dynamic object removal |
| `extrinsic_solver` | `extrinsic_solver_node.py` | 3D-to-3D SE(3) calibration (ORB or LightGlue) |
| `nvblox_node` | `nvblox_node.py` | TSDF fusion using nvblox_torch (mesh + pointcloud + costmap) |

## Launch

- **`launch/main_pipeline_launch.py`** — top-level entry point. Starts all 3 custom nodes + static TF publishers.
- RTAB-Map (single head instance, VO mode) is **optional** — the launch file checks if `rtabmap_slam` package exists and skips it silently if not found.

## Configuration

All runtime parameters live in YAML, not in code:
- `config/head.yaml` — head camera pipeline
- `config/exo.yaml` — exo camera pipeline
- `config/common.yaml` — extrinsic solver params (matcher_type, etc.)

Changing topics, filter params, or matcher type requires only YAML edits.

## Optional Dependencies

- `ultralytics>=8.0` (requirements.txt) — semantic masker falls back to empty mask if missing
- `lightglue` + `superpoint` — extrinsic solver falls back to ORB if missing
- `rtabmap_slam` ROS 2 package — head camera VO/SLAM, optional
- `nvblox_torch` (pip) — used by custom nvblox_node for TSDF fusion

## Pipeline Data Flow

```
Head RGB + depth → depth_preprocessor → semantic_masker → rtabmap (head VO, optional)
                                                     ↘
Exo RGB + depth  → depth_preprocessor → semantic_masker → nvblox_node (TSDF fusion)

semantic_masker (head+exo) → extrinsic_solver → TF head→exo
rtabmap → TF map→head
nvblox_node uses TF tree (map→head, map→head→exo) for fused TSDF
```

## Gotchas

- `build/`, `install/`, `log/` are colcon artifacts, gitignored
- `*.pt` and `*.engine` model files are gitignored — `yolov8n-seg.pt` must be placed in repo root manually
- Bag playback defaults to `$HOME/master_thesis/SLAM3R/data/exo/rosbag2_2026_05_06-16_48_23/rosbag2_2026_05_06-16_48_23_0.mcap`
- RGB and depth must already be published and aligned by an upstream camera stack; this package does not capture or align them
