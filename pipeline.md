# Pipeline architecture

## Online graph

```text
single-camera TUM RGB-D -- sequence_pair_adapter --.
head RGB-D ------------ depth_preprocessor --------+-- semantic_masker
exo RGB-D ------------- depth_preprocessor --------'         |
                                                             +-- LightGlue
                                                             +-- 3D RANSAC
                                                             +-- exo_link->head_link
                                                             '-- combined cloud

exo raw RGB + filtered depth -- rgbd_odometry -- RTAB-Map -- map
```

The sequence adapter is enabled only for single-camera datasets. It publishes
overlapping adjacent pairs: previous frame as head, current frame as exo.

## Frames

LightGlue deprojects matches in each optical frame, converts them to
`head_link` and `exo_link`, then solves `exo_link <- head_link`.
Fixed-rig datasets use EMA and physical/jump gates. TUM adjacent pairs use the
fresh estimate without fixed-rig validation or smoothing.

Successful estimates publish `/lightglue/combined_pointcloud`: masked sensor
depth from both cameras transformed into `exo_link`. Rejected estimates never
publish a stale evaluation cloud.

RTAB-Map independently owns `map -> odom -> exo_link`. Its input and tuning
match `vggt-omega` exactly.

## Evaluation

- Solver CSV: matches, inliers, RMSE, status, transform, GT error, compute time.
- Visible cloud: per-frame and accumulated accuracy, completeness, Chamfer,
  F-score at 2/5/10 cm.
- RTAB: raw odometry and optimized trajectory ATE/RPE/coverage; SE(3) primary,
  Sim(3) diagnostic; map cloud metrics.
- Non-GT recordings produce solver/runtime metrics only.

Cloud inputs are recorded during online processing, then evaluated offline so
KD-tree work does not alter solver timing.

Exoskeleton `/tf` and `/tf_static` are remapped to evaluation-only topics. A
small adapter composes dynamic exo-to-head GT and world-to-exo odometry without
publishing recorded robot joints into RViz or supplying camera pitch to LightGlue.
