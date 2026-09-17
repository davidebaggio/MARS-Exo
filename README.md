# exo_head_slam

ROS 2 `ament_python` pipeline for comparing LightGlue + 3D RANSAC extrinsic
calibration with the VGGT-Omega implementation on the same RGB-D datasets and
evaluation protocol.

## Pipeline

```text
head RGB-D -> depth preprocessing -> semantic masking --.
                                                        +-> LightGlue + RANSAC
exo RGB-D  -> depth preprocessing -> semantic masking --'      |
                                                               +-> exo_link -> head_link
exo RGB + filtered depth -> RTAB-Map odometry + mapping
```

LightGlue matches SuperPoint features, valid matched depths are deprojected,
and deterministic RANSAC estimates `exo_link <- head_link`. Successful
estimates also publish a combined raw RGB-D cloud in `exo_link`.

RTAB-Map uses only raw exo RGB and filtered exo depth. Its configuration is
kept identical to `vggt-omega`; RTAB trajectory/map metrics therefore measure
the shared SLAM backend, while extrinsic and combined-cloud metrics compare the
two calibration systems.

## Build

```bash
make build
source install/setup.bash
```

`make build` also builds the Fast-CDR compatibility shim required by older
ROS Jazzy installations and rewrites generated Python shebangs to the active
conda/virtual-environment interpreter.

LightGlue is required when `matcher_type: lightglue`; the node fails instead
of silently falling back to ORB. Install dependencies with `./setup.sh`.
Place optional `yolov8n-seg.pt` in the repository root. Missing YOLO weights
disable semantic masking but do not stop the pipeline.

Required ROS packages:

- `rtabmap_slam`, `rtabmap_odom`, `rtabmap_util`
- `sensor_msgs_py`, `rtabmap_msgs`, `std_srvs`
- optional `imu_filter_madgwick` when using `--imu`

## Run

```bash
./run.sh [--rate RATE] [--metrics-dir DIR] [--eval-dir DIR] \
  [--imu] [--debug-pcl] [--loop] [--headless] [--no-build] [bag_path]
```

Playback is one-shot unless `--loop` is passed. Supported ROS 2 datasets:

- native dual-camera recordings under `data/rosbag2_*`;
- exoskeleton ground-truth datasets under `data/`, including moving cameras;
- converted TUM bags under `data/TUM/*_cloud.bag`.

TUM inputs are converted with:

```bash
python data/TUM/add_pointclouds_to_bagfile.py \
  --groundtruth data/TUM/groundtruth.txt input.bag output_cloud.bag
```

The sequence adapter turns adjacent TUM frames into synchronized
head/exoskeleton pairs. Fixed-rig bounds, temporal smoothing, jump rejection,
and throttling are disabled for these dynamic pairs.

Run every supported ROS 2 dataset once:

```bash
./run_all_verified_evaluations.sh [--rate RATE]
```

Batch outputs go to `metrics/final_evals/lightglue/`. Default single-run
outputs go to `metrics/lightglue/`, avoiding VGGT result collisions.

## Evaluation outputs

Solver CSV and summary:

- 2D/3D matches, RANSAC inliers, ratio, RMSE, status;
- estimated transform and fresh-transform flag;
- ground-truth translation/rotation error when available;
- compute time.

Combined-cloud evaluation:

- accuracy/completeness mean and RMSE;
- Chamfer distance;
- F-score at 2, 5, and 10 cm;
- per-frame CSV plus accumulated global-map CSV/NPZ and plots.

RTAB benchmark JSON:

- odometry and optimized-trajectory SE(3) ATE/RPE;
- Sim(3) diagnostic and scale error;
- tracking coverage, lost samples, segments, duration;
- estimated-map accuracy/completeness, Chamfer, and F-scores.

Raw recordings without ground truth still produce solver/runtime metrics.

## Main interfaces

| Executable | Purpose |
|---|---|
| `depth_preprocessor` | Normalize/filter metric depth |
| `semantic_masker` | Batched dual-camera YOLO masking |
| `sequence_pair_adapter` | Convert single-camera sequences into adjacent pairs |
| `ground_truth_adapter` | Compose timestamped camera GT for evaluation only |
| `extrinsic_solver` | LightGlue matching, RANSAC, TF and combined cloud |
| `benchmark_evaluator` | RTAB trajectory/map evaluation |
| `cloud_map_evaluator` | Offline visible-cloud evaluation |

Important topics:

- `/lightglue/combined_pointcloud`
- `/exo_rtabmap/odom`
- `/exo_rtabmap/cloud_map`
- `/ground_truth/visible_cloud`
- `/ground_truth/visible_map`

For exoskeleton bags, recorded `/tf` and `/tf_static` are isolated from the
live TF tree. Camera pitch is never inferred from filenames; LightGlue predicts
online, while bag transforms provide evaluation GT only. Robot joints are not
republished in RViz.

Runtime parameters live in `config/head.yaml`, `config/exo.yaml`, and
`config/common.yaml`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The environment variable avoids unrelated system ROS pytest plugins when their
optional Python dependencies are absent.
