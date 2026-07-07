# Pipeline Problems & Solutions

## Contents
- [Infrastructure & Build](#infrastructure--build)
- [Depth Preprocessor](#depth-preprocessor)
- [Semantic Masker](#semantic-masker)
- [Fallback VO](#fallback-vo)
- [Extrinsic Solver (VGGT)](#extrinsic-solver-vggt)
- [PointCloud Publisher](#pointcloud-publisher)
- [Launch & Config](#launch--config)
- [Data Flow & Integration](#data-flow--integration)
- [Monitoring & Metrics](#monitoring--metrics)

---

## Infrastructure & Build

### Python shebang hardcoded to `/usr/bin/python3`
- **Problem:** `colcon build` generates shim scripts with `#!/usr/bin/python3`. If running in a conda/venv, nodes launch with system Python and fail to import dependencies (`ultralytics`, `torch`, `nvblox_torch`).
- **Solution:** `make build` and `run.sh` already run `sed -i "1s|^#!.*python.*|#!$(which python3)|"`. Ensure this always runs after every build. Add a ROS 2 `ComposableNode`-style approach to avoid shim scripts entirely.

### Large DL model files not tracked in git
- **Problem:** `yolov8n-seg.pt` must be manually placed in repo root. If missing, semantic_masker silently emits empty masks — no error, no warning that model file is absent.
- **Solution:** Check `model_path` at node init and log a fatal error if file doesn't exist, rather than falling through silently. Add a `make download-models` target.

### DDS SHM transport disabled globally
- **Problem:** `disable_shm.xml` forces UDP-only transport by disabling built-in transports. Increases latency and bandwidth for large messages (depth images, point clouds) compared to SHM.
- **Solution:** Use SHM with proper QoS configuration instead of blanket disabling. Only disable SHM if inter-process shared memory is known to cause crashes on the target system.

---

## Depth Preprocessor

### `expected_frame_id` parameter declared but never read
- **Problem:** `head.yaml` and `exo.yaml` both define `expected_frame_id` (e.g. `head`, `exo`), but the node never checks it. Bad data from misconfigured topics goes undetected.
- **Solution:** Read `expected_frame_id` param and validate `msg.header.frame_id` matches on each callback. Log warning on mismatch.

### `to_meters()` scale logic is ambiguous
- **Problem:** The method checks `encoding in ['16UC1', '16uc1']` and `depth_unit_scale` independently, with a fallback `0.001` for 16UC1 when scale is 1.0. If a camera publishes 16UC1 in meters with scale=1.0, values get incorrectly multiplied by 0.001.
- **Solution:** Simplify: if encoding is 16UC1, always apply `depth_unit_scale` (default 0.001 for mm-to-m). If encoding is 32FC1, use scale only if user explicitly sets it ≠ 1.0.

### Spatial filter converts to uint16 mm, losing precision
- **Problem:** `spatial_filter` casts `depth * 1000` to `np.uint16`, clamping values > 65.535 m and losing sub-mm precision before median blur.
- **Solution:** Apply median blur directly on float32 depth with a mask. Only quantize if absolutely required by cv2 API; use `cv2.medianBlur` on float if OpenCV ≥ 4.5 supports it, or implement custom median.

### Temporal filter does nothing on first frame
- **Problem:** `previous_filtered` is `None` on first callback, so `temporal_filter` returns current depth unchanged. First frame inherits no temporal smoothing.
- **Solution:** Initialize `previous_filtered` to zeros and warm up with a few frames before enabling temporal filter, or accumulate into a running average immediately.

### No max-depth clipping
- **Problem:** Depth values are only clamped for NaN/Inf. Extremely large valid values (e.g. 1000 m from sensor noise) pass through and corrupt downstream operations (point clouds, scale alignment).
- **Solution:** Add configurable `max_range_m` parameter and clip depth before publishing. Default should match the sensor's valid range (e.g. 10 m for Intel RealSense).

### BEST_EFFORT QoS on subscriber may drop frames
- **Problem:** Depth subscriber uses `BEST_EFFORT` reliability. If the upstream publisher uses `RELIABLE`, mismatched QoS causes connection failure.
- **Solution:** Match the upstream publisher's QoS. Use `SensorDataQoS()` from `rclpy.qos` which provides a sensible default for sensor streams.

### Single generic `Exception` catch
- **Problem:** All errors are caught as `Exception`, logged, and swallowed. A corrupted message or persistent error causes silent data loss every frame.
- **Solution:** Categorize errors: log and skip single-frame errors; raise or shutdown for persistent/recoverable failures (e.g., subscriber disconnected).

---

## Semantic Masker

### YOLO model path resolution is fragile
- **Problem:** Relative model paths are resolved by walking up 2 parent dirs from `__file__`. If the package is installed (not symlink-install), the resolved path points into `install/`, not the repo root where `yolov8n-seg.pt` lives.
- **Solution:** Use `ament_index_python.get_package_share_directory('exo_head_slam')` to find the package prefix. Check existence at init and error early.

### YOLO `predict()` runs on every frame with no rate limiting
- **Problem:** YOLOv8 inference at full camera framerate (15–30 Hz) saturates GPU. The `callback` blocks the rclpy spin loop during inference, dropping subsequent frames.
- **Solution:** Throttle inference to a configurable max rate (e.g. 3–5 Hz). For skipped frames, reuse the last valid mask. Or run inference in a separate thread with a queue.

### `queue_size=2` on ApproximateTimeSynchronizer is too small
- **Problem:** The synchronizer drops messages aggressively with only 2 slots per topic. Burst traffic or a slow YOLO callback causes frequent drops, starving downstream nodes.
- **Solution:** Increase `queue_size` to at least 10–20 to absorb timing jitter. The synchronizer already limits output via slop — the queue just needs enough capacity.

### RELIABLE publisher QoS with no depth can cause backpressure
- **Problem:** Output topics use `RELIABLE` QoS. If a subscriber processes slowly, the publisher blocks the node (rclpy is synchronous), halting the callback.
- **Solution:** Use `BEST_EFFORT` for high-bandwidth depth outputs. Let downstream nodes miss frames rather than backpressure the whole pipeline.

### Box-based fallback mask when YOLO masks are unavailable
- **Problem:** If `result.masks` is None, the code falls back to zeroing entire bounding boxes. This removes large rectangular areas, destroying far more geometry than necessary.
- **Solution:** Log warning and skip masking frame if instance masks unavailable, rather than using destructive box masking. Add config option for this behavior.

### Instance mask resampling with INTER_LINEAR blends boundaries
- **Problem:** Resizing the binary instance mask with `INTER_LINEAR` creates fractional values. The `> 0.5` threshold hides the issue but can cause partial pixel masking at object edges.
- **Solution:** Use `INTER_NEAREST` for mask resampling to preserve hard binary edges.

### `expected_frame_id` declared in YAML but never validated
- **Problem:** Same as depth preprocessor. Frame ID drift goes undetected.
- **Solution:** Validate `msg.header.frame_id` matches `expected_frame_id` at init and periodically.

---

## Fallback VO

### Only first 100 matches used for 3D estimation
- **Problem:** `pts_prev[:100]` caps correspondences at 100, discarding useful matches. In scenes with many features, this wastes data and degrades accuracy.
- **Solution:** Use all valid matches after filtering by max distance or confidence. Cap by computational budget only if needed, with a configurable limit.

### Accumulated pose drifts unbounded
- **Problem:** `T_odom_cam = T_odom_cam @ T_prev_curr` concatenates incremental transforms. Any error accumulates monotonically with no correction or loop closure.
- **Solution:** This is expected for pure VO (the RTAB-Map SLAM node is meant to correct drift). Document that fallback VO is dead-reckoning only. Add a reset mechanism triggered by large pose jumps.

### No reset on prolonged tracking failure
- **Problem:** After sustained tracking failure (RANSAC returns None or < 8 inliers), the node continues publishing the stale pose indefinitely.
- **Solution:** Publish a covariance warning or set covariances to infinity after N consecutive failures. Optionally reset pose and reinitialize when tracking recovers.

### `get_3d_point` uses integer rounding of pixel coordinates
- **Problem:** `int(round(pt[0]))` discards sub-pixel precision from matched keypoints. This matters at close range where a few pixels translates to centimeters.
- **Solution:** Use bilinear interpolation of depth at the sub-pixel location instead of nearest-neighbor.

### No camera info cache
- **Problem:** `K` matrix is read from every callback message. If camera info is published at lower rate than RGB/depth, the synchronizer may fire without info, or info may be stale.
- **Solution:** Cache the last received camera info and only update when a new one arrives. Fall back to cached copy if info is not synchronized.

### LightGlue fallback prints error every frame
- **Problem:** If LightGlue isn't available, a warn log fires on every callback when using `lightglue` matcher type, because the check is in `__init__` but the matcher object logs on creation. Actually this happens once at init, not every frame — no issue.
- **Solution:** N/A — keep as is, just verify log level won't flood.

---

## Extrinsic Solver (VGGT)

### VGGT-1B inference is very slow and GPU memory intensive
- **Problem:** VGGT-1B has ~1B parameters. Inference takes multiple seconds per frame even on high-end GPUs. The `min_solver_interval` (0.2s default) prevents running every frame, but the backlog from the `ApproximateTimeSynchronizer` queue (size 5) fills instantly.
- **Solution:** Use a dedicated thread for VGGT inference with frame skipping. Separate the synchronizer callback (lightweight) from the inference (heavy). Drop frames from the queue if inference falls behind by more than N frames.

### `torch.cuda.empty_cache()` called every callback
- **Problem:** `empty_cache()` is a blocking synchronization call that stalls the GPU pipeline and hurts performance. Called after every solve attempt.
- **Solution:** Remove `empty_cache()` calls. PyTorch manages memory internally. Call only if an OOM error is caught and recovery is needed.

### Scale alignment assumes uniform scale across both cameras
- **Problem:** `scale = (scale_head + scale_exo) / 2` averages the two camera scales. If one camera has little valid depth data (e.g. looking at sky), its median is noisy and corrupts the other camera's scale.
- **Solution:** Weight scale by number of valid depth points each camera contributes. Or compute scale from whichever camera has more valid data and use the other as a sanity check.

### Depth post-processing uses nearest-fill for cropped regions
- **Problem:** When padded/cropped during preprocessing, the depth map is filled by repeating edge rows (`canvas[:crop_y, :] = pred_depth_np[0, :]`). This creates artificial depth values at the top and bottom of the image.
- **Solution:** Instead of padding, resize the image to maintain aspect ratio without cropping, or pad with zeros and mask out padded regions in downstream processing.

### TF lookup timeout (0.1s) is too short under load
- **Problem:** `lookup_transform` with `timeout=0.1s` may fail when TF tree is busy or when transforms arrive slightly delayed relative to image timestamps.
- **Solution:** Increase timeout to 0.5s and add a retry with backoff. Also cache the last known optical→link transforms and re-use them if lookup fails.

### EMA quaternion interpolation is not geodesically correct
- **Problem:** Linear interpolation of quaternions (`(1-α)*q + α*q_new`) followed by normalization approximates SLERP but can introduce angular velocity artifacts, especially for large rotations.
- **Solution:** Use `scipy.spatial.transform.Rotation` SLERP for properly geodesic EMA smoothing, or keep the approximation but document the limitation.

### Distance bounds [0.2, 2.2] and Z bounds [0.1, 1.5] are hardcoded
- **Problem:** Camera-to-exo translation constraints are hardcoded in the solver. If the physical mounting changes, the code must be modified.
- **Solution:** Make all sanity-check thresholds configurable YAML parameters (`min_translation`, `max_translation`, `min_z`, `max_z`, etc.).

### Transform broadcast at every callback even without new solution
- **Problem:** The node always broadcasts the last known transforms at the end of `solve_callback`, even if the solver didn't produce new output. This publishes stale transforms with a new timestamp, misleading downstream consumers.
- **Solution:** Only broadcast with a new timestamp if the solver actually produced an update. Use a timer to republish at a lower rate for TF tree liveness.

### Hardcoded point cloud downsample factor of 4
- **Problem:** `downsample_factor = 4` is not configurable. In high-resolution images this produces sparse point clouds; in low-resolution it may remove too many points.
- **Solution:** Make downsample factor a configurable parameter with a default of 4.

### Clearing image buffer on exception loses context
- **Problem:** On any exception, `self.image_buffer.clear()` discards the entire sliding window. The next successful callback must rebuild from scratch, causing a gap in tracking continuity.
- **Solution:** Only clear the last failed frame pair from the buffer. Keep previous successful frames to maintain continuity.

---

## PointCloud Publisher

### Points published in camera frame despite `global_frame` param
- **Problem:** The node declares `global_frame` parameter and stores it, but never uses it — `pcl_msg.header.frame_id` is set to `depth_msg.header.frame_id` (camera optical frame). The parameter name is misleading.
- **Solution:** Either look up the TF transform from camera frame to `global_frame` and transform points, or rename the parameter to reflect actual behavior. Currently the debug PCLs are in camera frame, making cross-stream comparison in RViz impossible.

### Downsampled K matrix incorrectly scaled
- **Problem:** `K = K / self.downsample_factor` divides all elements of K (including fx, fy, cx, cy) by the factor, but sets `K[2,2] = 1.0`. This is correct only if the image is downsampled by exactly that factor AND the principal point scales accordingly. Integer rounding in width/height may cause slight misalignment.
- **Solution:** Compute K from the original camera info with proper scaling: `K[0,0] /= factor`, `K[1,1] /= factor`, `K[0,2] /= factor`, `K[1,2] /= factor`. Verify against OpenCV's `cv2.getOptimalNewCameraMatrix`.

### Queue size 200 encourages unbounded memory growth
- **Problem:** Synchronizer queue of 200 messages per topic at 640×480×3 bytes = ~184 MB per topic, × 3 topics = 552 MB. If sync fails, memory balloons.
- **Solution:** Reduce queue size to 10–20. Use `BEST_EFFORT` QoS for depth to prevent backpressure.

---

## Launch & Config

### `use_sim_time` passed inconsistently across nodes
- **Problem:** `main_pipeline_launch.py` passes `common_params = {'use_sim_time': use_sim_time}` as a sub-dict alongside config YAML. Not all config files declare `use_sim_time`, and ros2 parameter override behavior depends on merge order.
- **Solution:** Pass `use_sim_time` as a launch argument via `--ros-args -p use_sim_time:=true` on each Node declaration, or use `force_override` semantics.

### Static TF publishers create dummy frame aliases
- **Problem:** `static_transform_publisher` publishes `exo_link → exo_camera_link` and `head_link → head_camera_link` with identity transforms. If the bag already publishes these TFs with different transforms, the static publisher overwrites them (last writer wins in TF2).
- **Solution:** Remove static TF publishers if the bag provides correct transforms. Add a launch argument to conditionally enable them only when bag lacks the TFs.

### RTAB-Map optional check uses `get_package_share_directory` at parse time
- **Problem:** The `try/except` for `rtabmap_slam` runs during launch file parsing, not at runtime. If `rtabmap_slam` is installed but nodes fail, the whole launch fails silently.
- **Solution:** Use `launch.conditions.IfCondition` with a boolean launch argument `use_rtabmap` that defaults to checking package availability at parse time but can be overridden.

### Pipeline.md describes old ORB/LightGlue extrinsic solver, but code uses VGGT
- **Problem:** Documentation is out of sync with implementation. The pipeline data flow diagram in `pipeline.md` references VGGT differently than the actual data connections.
- **Solution:** Update `pipeline.md` to accurately reflect the VGGT-based extrinsic solver's output topics and control flow. Keep `GEMINI.md` and `AGENTS.md` in sync.

---

## Data Flow & Integration

### No timestamp synchronization across camera pairs in extrinsic solver
- **Problem:** The extrinsic solver's `ApproximateTimeSynchronizer` with slop=0.05s synchronizes head RGB, head depth, exo RGB, exo depth. If the cameras are not hardware-synced, the time offset between the two cameras can exceed the slop, causing frequent dropped quad-tuples.
- **Solution:** Increase slop to 0.1–0.2s. Add a timer-based soft sync fallback: if no quad arrives within N ms, process the latest frame from each topic independently.

### Single-threaded rclpy spin blocks on heavy callbacks
- **Problem:** YOLO inference, VGGT inference, and NVBlox integration are all CPU/GPU-bound and block the ROS 2 callback group. One slow node delays all others.
- **Solution:** Use `MultiThreadedExecutor` with separate callback groups for I/O-bound vs. CPU-bound work. Move heavy inference to dedicated threads with message queues.

### No node health monitoring or watchdog
- **Problem:** If any node crashes or hangs, the pipeline continues silently with missing data. For example, if the head depth preprocessor dies, the head semantic masker receives nothing but the exo chain continues working.
- **Solution:** Add a health monitor node that subscribes to `/alive` heartbeats from each node and kills the launch group on timeout. Use `rclpy`'s `on_shutdown` hooks for clean death reporting.

### Depth chain is serial and single points of failure
- **Problem:** `depth_preprocessor → semantic_masker → (extrinsic_solver / rtabmap)`. If depth_preprocessor dies, the masker gets no input and the entire downstream chain stalls with no fallback.
- **Solution:** Add a simple topic-based watchdog that detects missing input and either restarts the dead node or publishes a warning diagnostic. Configure `on_shutdown` to restart via the launch system.

### Bag playback at 0.3x with `--clock` may cause TF timing issues
- **Problem:** Bag replay at reduced rate with `/clock` topic causes ROS Time to advance slower than wall clock. TF lookups with timeout durations (in wall seconds) may timeout prematurely relative to the slowed time.
- **Solution:** Use `rclpy.time.Time()` comparisons in all TF lookups and set timeouts conservatively (e.g., 1.0s sim time). Verify that all nodes handle `use_sim_time` correctly.

---

## Monitoring & Metrics

### CSV I/O in critical callback path
- **Problem:** `log_metrics()` opens, appends to, and closes a CSV file on every solver callback. File I/O blocks the ROS 2 callback, adding unpredictable latency and risking data loss if the node crashes between writes.
- **Solution:** Buffer metrics in a deque and flush to disk asynchronously on a timer (e.g., every 5 seconds or every 100 entries). Use `csv.writer` in append mode so partial writes are safe.

### No recording of per-frame latency or throughput
- **Problem:** Metrics track extrinsic solver status and depth accuracy but not end-to-end latency, frame rates, or queue depths. Hard to diagnose performance bottlenecks.
- **Solution:** Add `diagnostic_updater` to each node publishing processing time, input rate, output rate, and queue depth. Collect in a central diagnostics topic.

### Ground-truth TF lookup uses latest available transform
- **Problem:** `self.tf_buffer.lookup_transform(self.gt_parent_frame, self.gt_child_frame, rclpy.time.Time())` with `Time()` (0 = latest) grabs the most recent transform, which may not correspond to the frame's actual timestamp. This invalidates the error metrics.
- **Solution:** Use the stamp of the synchronized message pair for the GT lookup. If timestamped GT is unavailable, skip GT metrics for that frame.

