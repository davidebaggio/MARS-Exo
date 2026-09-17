# AGENTS.md

## Project

ROS 2 `ament_python` package comparing LightGlue + RANSAC extrinsic
calibration against VGGT-Omega on identical RGB-D datasets and metrics.

## Build and test

```bash
make build
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

`make build` fixes generated Python shebangs and builds the Fast-CDR shim.

## Runtime

```bash
./run.sh [--rate RATE] [--imu] [--debug-pcl] [--loop] \
  [--headless] [--no-build] [bag_path]
./run_all_verified_evaluations.sh
```

Supported inputs: native dual-camera bags, exoskeleton ground-truth bags, and
converted TUM RGB-D bags under `data/`. Playback is one-shot by default.

## Nodes

| Executable | Purpose |
|---|---|
| `depth_preprocessor` | Metric depth normalization/filtering |
| `semantic_masker` | Batched YOLO dynamic-object masking |
| `sequence_pair_adapter` | Adjacent-frame TUM pairing |
| `ground_truth_adapter` | Evaluation-only dynamic camera GT composition |
| `extrinsic_solver` | LightGlue matching, RANSAC, TF, combined cloud |
| `pointcloud_publisher` | Optional debug clouds |
| `benchmark_evaluator` | RTAB trajectory/map metrics |
| `cloud_map_evaluator` | Offline visible-cloud metrics |

## Configuration

- `config/head.yaml`: head depth preprocessing
- `config/exo.yaml`: exo preprocessing and RTAB-Map
- `config/common.yaml`: masking, LightGlue/RANSAC, cloud evaluation

RTAB-Map configuration must remain identical to `vggt-omega` for fair
comparison. It consumes raw exo RGB plus filtered exo depth.

## Data flow

```text
head/exo RGB-D -> preprocessing -> semantic masking -> LightGlue/RANSAC
                                                   -> exo_link->head_link TF
                                                   -> /lightglue/combined_pointcloud
exo raw RGB + filtered depth -> RTAB odometry/map
GT bags -> extrinsic, cloud, trajectory, and map evaluators
```

Exoskeleton bag TF stays isolated from the live tree. The adapter composes
timestamped camera GT and exo-camera odometry; filenames never supply pitch.

## Gotchas

- LightGlue is required when configured; missing LightGlue/CUDA is fatal.
- Missing `yolov8n-seg.pt` disables masking but does not stop the pipeline.
- TUM pairs disable fixed-rig bounds, smoothing, jump rejection, and throttling.
- Original ROS 1 TUM bags require `data/TUM/add_pointclouds_to_bagfile.py`.
- Generated `build/`, `install/`, `log/`, model, bag, and metric files stay ignored.
