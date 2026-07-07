import rclpy
import csv
import os
import sys

import numpy as np
import cv2
import torch
import message_filters
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

# Locate vggt directory dynamically
_current_dir = os.path.dirname(os.path.abspath(__file__))
vggt_path = None
for _ in range(6):
    candidate = os.path.join(_current_dir, 'vggt')
    if os.path.isdir(candidate):
        vggt_path = candidate
        break
    _current_dir = os.path.dirname(_current_dir)

if vggt_path is None:
    raise RuntimeError(
        "Could not locate the vendored 'vggt/' directory. "
        "Clone VGGT into the package root or its parents."
    )

if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class ExtrinsicSolverNode(Node):
    def __init__(self):
        super().__init__('extrinsic_solver_node')

        # Declare parameters
        self.declare_parameter('head_rgb_topic', '/head/masked/image_raw')
        self.declare_parameter('head_depth_topic', '/head/masked/depth_raw')
        self.declare_parameter('exo_rgb_topic', '/exo/masked/image_raw')
        self.declare_parameter('exo_depth_topic', '/exo/masked/depth_raw')
        self.declare_parameter('head_frame_id', 'head_link')
        self.declare_parameter('exo_frame_id', 'exo_link')
        self.declare_parameter('tf_filter_alpha', 0.5)
        self.declare_parameter('sliding_window_size', 2)
        self.declare_parameter('min_solver_interval', 0.2)
        self.declare_parameter('max_trans_jump', 0.3)
        self.declare_parameter('max_rot_jump', 0.5)

        # VGGT depth-head confidence gating
        self.declare_parameter('depth_conf_mode', 'percentile')       # "percentile" | "absolute"
        self.declare_parameter('depth_conf_percentile', 25.0)         # drop bottom N% when mode=percentile
        self.declare_parameter('depth_conf_absolute', 1.5)            # conf threshold when mode=absolute
        self.declare_parameter('conf_gate_combine', True)              # gate raw+VGGT depth combine
        self.declare_parameter('conf_filter_pointcloud', True)        # percentile filter /vggt/combined_pointcloud

        self.declare_parameter('metrics_enabled', False)
        self.declare_parameter('metrics_csv_path', 'extrinsic_metrics.csv')
        self.declare_parameter('gt_parent_frame', '')
        self.declare_parameter('gt_child_frame', '')

        # Get values
        self.head_rgb_topic = self.get_parameter('head_rgb_topic').value
        self.head_depth_topic = self.get_parameter('head_depth_topic').value
        self.exo_rgb_topic = self.get_parameter('exo_rgb_topic').value
        self.exo_depth_topic = self.get_parameter('exo_depth_topic').value
        self.head_frame_id = self.get_parameter('head_frame_id').value
        self.exo_frame_id = self.get_parameter('exo_frame_id').value
        self.tf_filter_alpha = float(self.get_parameter('tf_filter_alpha').value)
        self.sliding_window_size = int(self.get_parameter('sliding_window_size').value)
        self.min_solver_interval = float(self.get_parameter('min_solver_interval').value)
        self.max_trans_jump = float(self.get_parameter('max_trans_jump').value)
        self.max_rot_jump = float(self.get_parameter('max_rot_jump').value)

        self.depth_conf_mode = str(self.get_parameter('depth_conf_mode').value).lower()
        self.depth_conf_percentile = float(self.get_parameter('depth_conf_percentile').value)
        self.depth_conf_absolute = float(self.get_parameter('depth_conf_absolute').value)
        self.conf_gate_combine = bool(self.get_parameter('conf_gate_combine').value)
        self.conf_filter_pointcloud = bool(self.get_parameter('conf_filter_pointcloud').value)

        self.metrics_enabled = self.get_parameter('metrics_enabled').value
        self.metrics_csv_path = self.get_parameter('metrics_csv_path').value
        self.gt_parent_frame = self.get_parameter('gt_parent_frame').value
        self.gt_child_frame = self.get_parameter('gt_child_frame').value

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.current_t = np.array([-0.3, 0.0, 0.5])
        self.current_q = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_vggt_world_t = np.array([-0.3, 0.0, 0.5])
        self.current_vggt_world_q = np.array([0.0, 0.0, 0.0, 1.0])
        self.image_buffer = []
        self.last_solver_time = None

        # Publishers
        self.head_combined_depth_pub = self.create_publisher(Image, '/head/combined/depth_raw', 10)
        self.exo_combined_depth_pub = self.create_publisher(Image, '/exo/combined/depth_raw', 10)
        self.vggt_pcl_pub = self.create_publisher(PointCloud2, '/vggt/combined_pointcloud', 10)

        # Initialize VGGT-1B on CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            self.autocast_ctx = torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            self.dtype = torch.float32
            self.autocast_ctx = torch.cpu.amp.autocast(enabled=False)

        self.get_logger().info(f"Loading VGGT-1B on {self.device}...")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.model.eval()
        self.get_logger().info("VGGT-1B loaded successfully.")

        # Subscriptions
        self.head_rgb_sub = message_filters.Subscriber(self, Image, self.head_rgb_topic)
        self.head_depth_sub = message_filters.Subscriber(self, Image, self.head_depth_topic)
        self.exo_rgb_sub = message_filters.Subscriber(self, Image, self.exo_rgb_topic)
        self.exo_depth_sub = message_filters.Subscriber(self, Image, self.exo_depth_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.head_rgb_sub, self.head_depth_sub, self.exo_rgb_sub, self.exo_depth_sub],
            queue_size=5,
            slop=0.05
        )
        self.ts.registerCallback(self.solve_callback)
        self.get_logger().info(
            f"Extrinsic Solver Node (VGGT) online. "
            f"conf_mode={self.depth_conf_mode} gate_combine={self.conf_gate_combine} "
            f"filter_pcl={self.conf_filter_pointcloud}"
        )

    def preprocess_cv2_image(self, cv_img, target_size=518):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_img.shape

        new_width = target_size
        new_height = int(round(h * (target_size / w) / 14) * 14)

        resized = cv2.resize(rgb_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        crop_y = 0
        if new_height > target_size:
            crop_y = (new_height - target_size) // 2
            resized = resized[crop_y : crop_y + target_size, :]

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor, h, w, new_height, crop_y

    def postprocess_depth(self, pred_arr, orig_h, orig_w, new_height, crop_y, target_size=518):
        if new_height > target_size:
            canvas = np.zeros((new_height, target_size), dtype=np.float32)
            canvas[crop_y : crop_y + target_size, :] = pred_arr
            if crop_y > 0:
                canvas[:crop_y, :] = pred_arr[0, :]
                canvas[crop_y + target_size :, :] = pred_arr[-1, :]
            arr_resized = canvas
        else:
            arr_resized = pred_arr

        arr_orig = cv2.resize(arr_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return arr_orig

    def tf_to_matrix(self, tf):
        mat = np.eye(4)
        q = [tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w]
        from scipy.spatial.transform import Rotation as R
        mat[:3, :3] = R.from_quat(q).as_matrix()
        mat[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y,
                      tf.transform.translation.z]
        return mat

    def _conf_threshold(self, conf_np: np.ndarray) -> float:
        if self.depth_conf_mode == "absolute":
            return self.depth_conf_absolute
        # percentile filter against current frame's conf values
        return float(np.percentile(conf_np, self.depth_conf_percentile))

    def _conf_mask(self, conf_np: np.ndarray) -> np.ndarray:
        return conf_np >= self._conf_threshold(conf_np)

    def solve_callback(self, h_rgb: Image, h_depth: Image, e_rgb: Image, e_depth: Image):
        from scipy.spatial.transform import Rotation as R

        stamp_sec = h_rgb.header.stamp.sec + h_rgb.header.stamp.nanosec * 1e-9

        run_solver = (self.last_solver_time is None or
                      (stamp_sec - self.last_solver_time) >= self.min_solver_interval)

        if run_solver:
            self.get_logger().info("Solver: received synced quad. Estimating extrinsic & depth with VGGT...")
            scale = 1.0
            head_depth_rmse = head_depth_mae = None
            exo_depth_rmse = exo_depth_mae = None
            head_conf_stats = None
            exo_conf_stats = None
            conf_thr = None
            try:
                head_img = self.bridge.imgmsg_to_cv2(h_rgb, 'bgr8')
                head_dep = self.bridge.imgmsg_to_cv2(h_depth, 'passthrough')
                exo_img = self.bridge.imgmsg_to_cv2(e_rgb, 'bgr8')
                exo_dep = self.bridge.imgmsg_to_cv2(e_depth, 'passthrough')

                h_tensor, h_orig_h, h_orig_w, h_new_h, h_crop_y = self.preprocess_cv2_image(head_img)
                e_tensor, e_orig_h, e_orig_w, e_new_h, e_crop_y = self.preprocess_cv2_image(exo_img)

                self.image_buffer.append((h_tensor, e_tensor))
                if len(self.image_buffer) > self.sliding_window_size:
                    self.image_buffer.pop(0)

                all_tensors = []
                for hb, eb in self.image_buffer:
                    all_tensors.extend([hb, eb])
                images = torch.stack(all_tensors).unsqueeze(0).to(self.device)
                M = len(self.image_buffer)

                with torch.no_grad(), self.autocast_ctx:
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                    depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
                    point_map_by_unprojection = unproject_depth_map_to_point_map(
                        depth_map.squeeze(0).float(),
                        extrinsic.squeeze(0).float(),
                        intrinsic.squeeze(0).float()
                    )

                E_head = extrinsic[0, 2 * M - 2].cpu().float().numpy()
                E_exo = extrinsic[0, 2 * M - 1].cpu().float().numpy()

                h_pred_depth = depth_map[0, 2 * M - 2, ..., 0].cpu().float().numpy()
                e_pred_depth = depth_map[0, 2 * M - 1, ..., 0].cpu().float().numpy()
                h_conf = depth_conf[0, 2 * M - 2].cpu().float().numpy()
                e_conf = depth_conf[0, 2 * M - 1].cpu().float().numpy()

                self.get_logger().info(
                    f"VGGT: depth_map={depth_map.shape} depth_conf={depth_conf.shape} "
                    f"point_map={point_map_by_unprojection.shape}"
                )

                # Postprocess depth maps back to original dimensions
                h_pred_orig = self.postprocess_depth(h_pred_depth, h_orig_h, h_orig_w, h_new_h, h_crop_y)
                e_pred_orig = self.postprocess_depth(e_pred_depth, e_orig_h, e_orig_w, e_new_h, e_crop_y)
                # Postprocess depth_conf back to original dimensions for combine gating
                h_conf_orig = self.postprocess_depth(h_conf, h_orig_h, h_orig_w, h_new_h, h_crop_y)
                e_conf_orig = self.postprocess_depth(e_conf, e_orig_h, e_orig_w, e_new_h, e_crop_y)

                # Scale alignment logic
                scale_head = scale_exo = None
                h_valid = (head_dep > 0.1) & (head_dep < 10.0) & (~np.isnan(head_dep)) & (~np.isinf(head_dep))
                if np.sum(h_valid) > 10:
                    scale_head = np.median(head_dep[h_valid] / h_pred_orig[h_valid])

                e_valid = (exo_dep > 0.1) & (exo_dep < 10.0) & (~np.isnan(exo_dep)) & (~np.isinf(exo_dep))
                if np.sum(e_valid) > 10:
                    scale_exo = np.median(exo_dep[e_valid] / e_pred_orig[e_valid])

                if scale_head is not None and scale_exo is not None:
                    scale = (scale_head + scale_exo) / 2.0
                elif scale_head is not None:
                    scale = scale_head
                elif scale_exo is not None:
                    scale = scale_exo
                else:
                    scale = 1.0
                    self.get_logger().warn("No valid depth points found for scale alignment. Using scale = 1.0.")

                h_pred_orig_scaled = h_pred_orig * scale
                e_pred_orig_scaled = e_pred_orig * scale

                E_head_scaled = E_head.copy()
                E_head_scaled[:3, 3] *= scale
                E_exo_scaled = E_exo.copy()
                E_exo_scaled[:3, 3] *= scale

                # Conf masks (per-frame) for the depth combine gate
                h_conf_mask_h = self._conf_mask(h_conf_orig)
                e_conf_mask_e = self._conf_mask(e_conf_orig)
                conf_thr = self._conf_threshold(np.concatenate([h_conf_orig.ravel(), e_conf_orig.ravel()]))

                # Combine depth maps: raw where valid, VGGT-scaled where invalid,
                # optionally restricting VGGT fill to high-confidence pixels only.
                h_use_vggt = ~h_valid
                if self.conf_gate_combine:
                    h_use_vggt = h_use_vggt & h_conf_mask_h
                e_use_vggt = ~e_valid
                if self.conf_gate_combine:
                    e_use_vggt = e_use_vggt & e_conf_mask_e

                h_combined = np.where(h_use_vggt, h_pred_orig_scaled, head_dep)
                h_combined = np.nan_to_num(h_combined, nan=0.0, posinf=0.0, neginf=0.0)
                h_combined[h_combined < 0.0] = 0.0

                e_combined = np.where(e_use_vggt, e_pred_orig_scaled, exo_dep)
                e_combined = np.nan_to_num(e_combined, nan=0.0, posinf=0.0, neginf=0.0)
                e_combined[e_combined < 0.0] = 0.0

                h_msg = self.bridge.cv2_to_imgmsg(h_combined.astype(np.float32), encoding='32FC1')
                h_msg.header = h_depth.header
                self.head_combined_depth_pub.publish(h_msg)

                e_msg = self.bridge.cv2_to_imgmsg(e_combined.astype(np.float32), encoding='32FC1')
                e_msg.header = e_depth.header
                self.exo_combined_depth_pub.publish(e_msg)

                # Depth estimation metrics (VGGT-scaled vs raw metric) on confident pixels
                h_valid_metric = h_valid & h_conf_mask_h
                e_valid_metric = e_valid & e_conf_mask_e
                if np.sum(h_valid_metric) > 0:
                    h_diff = h_pred_orig_scaled[h_valid_metric] - head_dep[h_valid_metric]
                    head_depth_rmse = float(np.sqrt(np.mean(h_diff ** 2)))
                    head_depth_mae = float(np.mean(np.abs(h_diff)))
                if np.sum(e_valid_metric) > 0:
                    e_diff = e_pred_orig_scaled[e_valid_metric] - exo_dep[e_valid_metric]
                    exo_depth_rmse = float(np.sqrt(np.mean(e_diff ** 2)))
                    exo_depth_mae = float(np.mean(np.abs(e_diff)))

                if self.metrics_enabled:
                    head_conf_stats = (
                        float(np.percentile(h_conf_orig, 50)),
                        float(np.percentile(h_conf_orig, 95)),
                    )
                    exo_conf_stats = (
                        float(np.percentile(e_conf_orig, 50)),
                        float(np.percentile(e_conf_orig, 95)),
                    )

                # Publish VGGT Point Cloud map (conf-filtered if enabled)
                h_pts = point_map_by_unprojection[2 * M - 2]
                e_pts = point_map_by_unprojection[2 * M - 1]
                h_color = (h_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                e_color = (e_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                self.publish_vggt_pointcloud(
                    h_pts, e_pts, h_color, e_color,
                    h_pred_depth, e_pred_depth, h_conf, e_conf,
                    scale, e_rgb.header.stamp
                )

                # Compute relative extrinsic camera pose (in optical frames)
                T_head = np.eye(4)
                T_head[:3, :] = E_head_scaled
                T_exo = np.eye(4)
                T_exo[:3, :] = E_exo_scaled

                T_exo_opt_from_head_opt = T_exo @ np.linalg.inv(T_head)

                try:
                    t_h_opt_link = self.tf_buffer.lookup_transform(
                        h_rgb.header.frame_id, self.head_frame_id,
                        h_rgb.header.stamp,
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    t_e_link_opt = self.tf_buffer.lookup_transform(
                        self.exo_frame_id, e_rgb.header.frame_id,
                        e_rgb.header.stamp,
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    T_head_opt_from_head_link = self.tf_to_matrix(t_h_opt_link)
                    T_exo_link_from_exo_opt = self.tf_to_matrix(t_e_link_opt)

                    T_exo_link_from_head_link = (
                        T_exo_link_from_exo_opt @ T_exo_opt_from_head_opt @ T_head_opt_from_head_link
                    )

                    new_t = T_exo_link_from_head_link[:3, 3]
                    new_q = R.from_matrix(T_exo_link_from_head_link[:3, :3]).as_quat()

                    T_exo_link_from_vggt_world = T_exo_link_from_exo_opt @ T_exo
                    new_vggt_t = T_exo_link_from_vggt_world[:3, 3]
                    new_vggt_q = R.from_matrix(T_exo_link_from_vggt_world[:3, :3]).as_quat()

                    valid_tf = True
                    dist = np.linalg.norm(new_t)
                    if not (0.2 <= dist <= 2.2):
                        self.get_logger().warn(f"Rejecting transform: distance {dist:.2f} m out of bounds [0.2, 2.2]")
                        valid_tf = False
                        self._log_metrics(stamp_sec, 'OUT_OF_BOUNDS', scale=scale,
                                          head_depth_rmse=head_depth_rmse, head_depth_mae=head_depth_mae,
                                          exo_depth_rmse=exo_depth_rmse, exo_depth_mae=exo_depth_mae,
                                          head_conf_stats=head_conf_stats, exo_conf_stats=exo_conf_stats,
                                          conf_thr=conf_thr, new_t=new_t, new_q=new_q)
                    elif not (0.1 <= new_t[2] <= 1.5):
                        self.get_logger().warn(f"Rejecting transform: relative Z height {new_t[2]:.2f} m out of bounds [0.1, 1.5]")
                        valid_tf = False
                        self._log_metrics(stamp_sec, 'OUT_OF_BOUNDS', scale=scale,
                                          head_depth_rmse=head_depth_rmse, head_depth_mae=head_depth_mae,
                                          exo_depth_rmse=exo_depth_rmse, exo_depth_mae=exo_depth_mae,
                                          head_conf_stats=head_conf_stats, exo_conf_stats=exo_conf_stats,
                                          conf_thr=conf_thr, new_t=new_t, new_q=new_q)

                    if valid_tf and self.last_solver_time is not None:
                        trans_jump = np.linalg.norm(new_t - self.current_t)
                        if trans_jump > self.max_trans_jump:
                            self.get_logger().warn(f"Rejecting transform due to translation jump: {trans_jump:.3f} m > {self.max_trans_jump} m")
                            valid_tf = False
                            self._log_metrics(stamp_sec, 'TRANSLATION_JUMP', scale=scale,
                                              head_depth_rmse=head_depth_rmse, head_depth_mae=head_depth_mae,
                                              exo_depth_rmse=exo_depth_rmse, exo_depth_mae=exo_depth_mae,
                                              head_conf_stats=head_conf_stats, exo_conf_stats=exo_conf_stats,
                                              conf_thr=conf_thr, new_t=new_t, new_q=new_q)

                    if valid_tf and self.last_solver_time is not None:
                        dot_product = min(1.0, max(0.0, abs(np.dot(self.current_q, new_q))))
                        rot_jump = 2.0 * np.arccos(dot_product)
                        if rot_jump > self.max_rot_jump:
                            self.get_logger().warn(f"Rejecting transform due to rotation jump: {rot_jump:.3f} rad > {self.max_rot_jump} rad")
                            valid_tf = False
                            self._log_metrics(stamp_sec, 'ROTATION_JUMP', scale=scale,
                                              head_depth_rmse=head_depth_rmse, head_depth_mae=head_depth_mae,
                                              exo_depth_rmse=exo_depth_rmse, exo_depth_mae=exo_depth_mae,
                                              head_conf_stats=head_conf_stats, exo_conf_stats=exo_conf_stats,
                                              conf_thr=conf_thr, new_t=new_t, new_q=new_q)

                    if valid_tf:
                        if self.last_solver_time is None:
                            self.current_t = new_t
                            self.current_q = new_q
                            self.current_vggt_world_t = new_vggt_t
                            self.current_vggt_world_q = new_vggt_q
                        else:
                            self.current_t = (1.0 - self.tf_filter_alpha) * self.current_t + self.tf_filter_alpha * new_t
                            if np.dot(self.current_q, new_q) < 0:
                                new_q = -new_q
                            self.current_q = (1.0 - self.tf_filter_alpha) * self.current_q + self.tf_filter_alpha * new_q
                            self.current_q /= np.linalg.norm(self.current_q)

                            self.current_vggt_world_t = (1.0 - self.tf_filter_alpha) * self.current_vggt_world_t + self.tf_filter_alpha * new_vggt_t
                            if np.dot(self.current_vggt_world_q, new_vggt_q) < 0:
                                new_vggt_q = -new_vggt_q
                            self.current_vggt_world_q = (1.0 - self.tf_filter_alpha) * self.current_vggt_world_q + self.tf_filter_alpha * new_vggt_q
                            self.current_vggt_world_q /= np.linalg.norm(self.current_vggt_world_q)

                        self.get_logger().info(
                            f"Updated TF: t=[{self.current_t[0]:.3f}, {self.current_t[1]:.3f}, {self.current_t[2]:.3f}] "
                            f"q=[{self.current_q[0]:.3f}, {self.current_q[1]:.3f}, {self.current_q[2]:.3f}, {self.current_q[3]:.3f}]"
                        )
                        self.last_solver_time = stamp_sec
                        self._log_metrics(stamp_sec, 'SUCCESS', scale=scale,
                                          head_depth_rmse=head_depth_rmse, head_depth_mae=head_depth_mae,
                                          exo_depth_rmse=exo_depth_rmse, exo_depth_mae=exo_depth_mae,
                                          head_conf_stats=head_conf_stats, exo_conf_stats=exo_conf_stats,
                                          conf_thr=conf_thr, new_t=self.current_t, new_q=self.current_q)

                except Exception as e:
                    self.get_logger().warn(f"TF Lookup failed: {str(e)}")
                    self._log_metrics(stamp_sec, 'TF_LOOKUP_ERROR', scale=scale,
                                      head_depth_rmse=head_depth_rmse, head_depth_mae=head_depth_mae,
                                      exo_depth_rmse=exo_depth_rmse, exo_depth_mae=exo_depth_mae,
                                      head_conf_stats=head_conf_stats, exo_conf_stats=exo_conf_stats,
                                      conf_thr=conf_thr)

            except Exception as e:
                self.get_logger().error(f"VGGT extrinsic solver callback failed: {str(e)}")
                self.image_buffer.clear()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                self._log_metrics(stamp_sec, 'SOLVER_ERROR', scale=scale,
                                  head_depth_rmse=head_depth_rmse, head_depth_mae=head_depth_mae,
                                  exo_depth_rmse=exo_depth_rmse, exo_depth_mae=exo_depth_mae,
                                  head_conf_stats=head_conf_stats, exo_conf_stats=exo_conf_stats,
                                  conf_thr=conf_thr)

        # Always broadcast the last known good transforms to keep TF tree active
        if self.current_t is not None:
            self.broadcast_transform(e_rgb.header.stamp)
        if self.current_vggt_world_t is not None:
            self.broadcast_vggt_world_transform(e_rgb.header.stamp)

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def publish_vggt_pointcloud(self, h_pts, e_pts, h_color, e_color,
                                h_pred_depth, e_pred_depth, h_conf, e_conf,
                                scale, stamp):
        h_pts_metric = h_pts * scale
        e_pts_metric = e_pts * scale
        h_depth_scaled = h_pred_depth * scale
        e_depth_scaled = e_pred_depth * scale

        h_valid = (h_depth_scaled > 0.1) & (h_depth_scaled < 6.0)
        e_valid = (e_depth_scaled > 0.1) & (e_depth_scaled < 6.0)

        if self.conf_filter_pointcloud:
            h_valid = h_valid & self._conf_mask(h_conf)
            e_valid = e_valid & self._conf_mask(e_conf)

        h_pts_valid = h_pts_metric[h_valid]
        h_color_valid = h_color[h_valid]
        e_pts_valid = e_pts_metric[e_valid]
        e_color_valid = e_color[e_valid]

        if len(h_pts_valid) == 0 and len(e_pts_valid) == 0:
            self.get_logger().warn("VGGT PCL: no valid points after conf filter; skipping publish.")
            return

        all_pts = np.vstack([h_pts_valid, e_pts_valid])
        all_color = np.vstack([h_color_valid, e_color_valid])

        # Downsample to avoid ROS network bottlenecks
        downsample_factor = 4
        all_pts = all_pts[::downsample_factor]
        all_color = all_color[::downsample_factor]

        num_points = len(all_pts)
        if num_points == 0:
            return

        data = np.zeros(num_points, dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('rgb', np.uint32)
        ])
        data['x'] = all_pts[:, 0]
        data['y'] = all_pts[:, 1]
        data['z'] = all_pts[:, 2]

        rgb_packed = (all_color[:, 0].astype(np.uint32) << 16) | \
                     (all_color[:, 1].astype(np.uint32) << 8) | \
                     (all_color[:, 2].astype(np.uint32))
        data['rgb'] = rgb_packed

        pcl_msg = PointCloud2()
        pcl_msg.header.frame_id = 'vggt_world'
        pcl_msg.header.stamp = stamp
        pcl_msg.height = 1
        pcl_msg.width = num_points
        pcl_msg.is_dense = False
        pcl_msg.is_bigendian = False
        pcl_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        pcl_msg.point_step = 16
        pcl_msg.row_step = pcl_msg.point_step * num_points
        pcl_msg.data = data.tobytes()

        self.vggt_pcl_pub.publish(pcl_msg)
        self.get_logger().info(f"Published VGGT world point cloud: {num_points} points")

    def broadcast_transform(self, stamp):
        if self.current_t is None or self.current_q is None:
            return

        t_msg = TransformStamped()
        t_msg.header.stamp = stamp
        t_msg.header.frame_id = self.exo_frame_id
        t_msg.child_frame_id = self.head_frame_id

        t_msg.transform.translation.x = float(self.current_t[0])
        t_msg.transform.translation.y = float(self.current_t[1])
        t_msg.transform.translation.z = float(self.current_t[2])

        t_msg.transform.rotation.x = float(self.current_q[0])
        t_msg.transform.rotation.y = float(self.current_q[1])
        t_msg.transform.rotation.z = float(self.current_q[2])
        t_msg.transform.rotation.w = float(self.current_q[3])

        self.tf_broadcaster.sendTransform(t_msg)

    def broadcast_vggt_world_transform(self, stamp):
        if self.current_vggt_world_t is None or self.current_vggt_world_q is None:
            return

        t_msg = TransformStamped()
        t_msg.header.stamp = stamp
        t_msg.header.frame_id = self.exo_frame_id
        t_msg.child_frame_id = 'vggt_world'

        t_msg.transform.translation.x = float(self.current_vggt_world_t[0])
        t_msg.transform.translation.y = float(self.current_vggt_world_t[1])
        t_msg.transform.translation.z = float(self.current_vggt_world_t[2])

        t_msg.transform.rotation.x = float(self.current_vggt_world_q[0])
        t_msg.transform.rotation.y = float(self.current_vggt_world_q[1])
        t_msg.transform.rotation.z = float(self.current_vggt_world_q[2])
        t_msg.transform.rotation.w = float(self.current_vggt_world_q[3])

        self.tf_broadcaster.sendTransform(t_msg)

    def _log_metrics(self, stamp_sec, status, scale=1.0,
                     head_depth_rmse=None, head_depth_mae=None,
                     exo_depth_rmse=None, exo_depth_mae=None,
                     head_conf_stats=None, exo_conf_stats=None,
                     conf_thr=None,
                     new_t=None, new_q=None):
        if not self.metrics_enabled:
            return

        t_x, t_y, t_z = 0.0, 0.0, 0.0
        if new_t is not None:
            t_x, t_y, t_z = float(new_t[0]), float(new_t[1]), float(new_t[2])

        gt_t_x = gt_t_y = gt_t_z = None
        error_t = error_r_deg = None

        h_conf_p50 = h_conf_p95 = None
        if head_conf_stats is not None:
            h_conf_p50, h_conf_p95 = head_conf_stats
        e_conf_p50 = e_conf_p95 = None
        if exo_conf_stats is not None:
            e_conf_p50, e_conf_p95 = exo_conf_stats

        if self.gt_parent_frame and self.gt_child_frame and new_t is not None and new_q is not None:
            try:
                gt_tf = self.tf_buffer.lookup_transform(
                    self.gt_parent_frame, self.gt_child_frame, rclpy.time.Time()
                )
                gt_mat = self.tf_to_matrix(gt_tf)
                gt_t = gt_mat[:3, 3]
                gt_r = gt_mat[:3, :3]

                gt_t_x, gt_t_y, gt_t_z = float(gt_t[0]), float(gt_t[1]), float(gt_t[2])
                error_t = float(np.linalg.norm(new_t - gt_t))

                from scipy.spatial.transform import Rotation as R
                est_r = R.from_quat(new_q).as_matrix()
                R_diff = est_r.T @ gt_r
                trace = np.trace(R_diff)
                cos_theta = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
                error_r_deg = float(np.degrees(np.arccos(cos_theta)))

            except Exception as e:
                self.get_logger().warn(f"Failed to lookup GT transform for metrics: {str(e)}")

        dirname = os.path.dirname(self.metrics_csv_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        file_exists = os.path.exists(self.metrics_csv_path) and os.path.getsize(self.metrics_csv_path) > 0
        header = [
            'timestamp', 'status', 'scale',
            'head_depth_rmse', 'head_depth_mae', 'exo_depth_rmse', 'exo_depth_mae',
            'head_conf_p50', 'head_conf_p95', 'exo_conf_p50', 'exo_conf_p95', 'conf_thr',
            't_x', 't_y', 't_z', 'gt_t_x', 'gt_t_y', 'gt_t_z', 'error_t', 'error_r_deg'
        ]
        try:
            with open(self.metrics_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow([
                    stamp_sec, status, scale,
                    head_depth_rmse, head_depth_mae, exo_depth_rmse, exo_depth_mae,
                    h_conf_p50, h_conf_p95, e_conf_p50, e_conf_p95, conf_thr,
                    t_x, t_y, t_z, gt_t_x, gt_t_y, gt_t_z, error_t, error_r_deg
                ])
        except Exception as e:
            self.get_logger().error(f"Failed to write metrics to CSV: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()