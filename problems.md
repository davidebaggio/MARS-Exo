1. Preprocessing (Depth Preprocessor & Semantic Masker)

Problem: Blocking Inference in SemanticMaskerNode
The SemanticMaskerNode performs YOLOv8 inference directly inside the callback (triggered
by ApproximateTimeSynchronizer). Since ROS 2 Python nodes are typically single-threaded
by default, this blocks the entire executor. If inference takes 50ms and the camera is
30fps (33ms), the node will fall behind, drop messages, and increase latency for the
entire pipeline.

- Solution: Use a MultithreadedExecutor with a ReentrantCallbackGroup or move the
  inference to a separate worker thread/process using a producer-consumer pattern with a
  queue to keep the communication layer responsive.

Problem: Synchronization "Slop" in SemanticMaskerNode
The ApproximateTimeSynchronizer uses a slop of 0.05s (50ms). If the camera streams have
jitter or are slightly out of sync, the masker will pair RGB and Depth frames that don't
perfectly align. This causes "halos" or misaligned masks, especially during fast motion,
leading to "ghost" artifacts in the volumetric fusion.

- Solution: Reduce slop to match the camera period (e.g., 0.033 for 30fps) and use
  message_filters.Cache to ensure the most recent messages are available for matching.

Problem: Floating Point Depth Units in DepthPreprocessorNode
The to_meters function assumes that if np.max(depth_m) > 100.0, the units are
millimeters. This is a fragile heuristic. Some sensors might publish depth in meters as
float32 but include high-value noise or outliers (e.g., sky, reflections) that trip this
condition, causing the entire image to be scaled down incorrectly.

- Solution: Explicitly define the input units via a ROS parameter or use the encoding
  field in the sensor_msgs/Image message to determine the bit-depth and scaling
  requirements.

---

2. Localization & Tracking (RTAB-Map)

Problem: Single-Camera Odometry Vulnerability
In rtabmap_agents_launch.py, visual odometry only runs on the Exo camera. If the Exo
camera is obscured (e.g., by the user's arm or a wall), the map -> odom transform will
fail, causing the entire pipeline to stop updating. The Head camera data is "blindly"
attached to this chain via the extrinsic solver.

- Solution: Implement multi-camera odometry or a fallback mechanism. RTAB-Map can
  subscribe to multiple RGB-D streams, or a secondary odometry node can run on the Head
  camera to provide a redundant transform if the Exo camera loses tracking.

Problem: TF "Map-to-Odom" Jumpiness
The SLAM node (exo_rtabmap) publishes the map -> odom transform. If loop closures occur
or the optimizer shifts the map, this transform can jump. Since NVBlox and the Extrinsic
Solver rely on stable TFs for integration, these jumps can cause double-mapping or
misalignment in the 3D volume.

- Solution: Enable TF filtering or use a smoother (like robot_localization) to fuse
  RTAB-Map's output with an IMU if available, or increase the RTAB-Map optimization
  frequency to make corrections more incremental.

---

3. Spatial Synchronization (Extrinsic Solver)

Problem: Heavy Computation and "Static" TF Expectation
The ExtrinsicSolverNode uses LightGlue on every synchronized frame. LightGlue is powerful
but computationally expensive. Furthermore, it broadcasts the exo_link -> head_link
transform on every frame. If the solver fails for a few frames (due to lack of overlap or
motion blur), the TF chain breaks.

- Solution: Implement a TF Buffer/Filter. Only update the transform when confidence is
  high (low RANSAC error) and use a Kalman Filter or EMA (Exponential Moving Average) to
  maintain a stable transform even when the solver temporarily fails.

Problem: Optical vs. Link Frame Confusion
The solver matches features in the Optical frame (Z-forward) but computes the transform
between Link frames (X-forward). It performs manual matrix multiplications (T_h_link_opt)
to bridge this. Any discrepancy in the URDF/Static TF for these frames will lead to a
systematic rotation error in the alignment.

- Solution: Use tf2_ros to look up the exact camera_optical -> camera_link transform at
  the specific message timestamp instead of assuming identity or hardcoded values.

Problem: Lack of Convergence Check
The solver uses compute_transform_ransac and immediately broadcasts. It doesn't check if
the Head and Exo cameras are even looking at the same scene. If they aren't, it will
produce a "hallucinated" transform that places the Head camera in an impossible location.

- Solution: Add a Geometric Consistency Check. Verify the resulting transform against
  the last known "good" extrinsic (if the cameras are relatively fixed) and ensure the
  number of inliers and the RANSAC error are within strict thresholds before
  broadcasting.
