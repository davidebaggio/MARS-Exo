import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
import message_filters
import tf2_ros
import uuid
import torch
from scipy.spatial.transform import Rotation as R

from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor


class NvbloxNode(Node):
    def __init__(self):
        super().__init__('nvblox_node')
        self.instance_id = str(uuid.uuid4())[:8]

        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('voxel_size_m', 0.05)
        self.declare_parameter('max_integration_distance_m', 5.0)
        self.declare_parameter('mesh_update_period', 10)
        self.declare_parameter('costmap_resolution', 0.1)
        self.declare_parameter('costmap_height_min', 0.0)
        self.declare_parameter('costmap_height_max', 1.0)
        self.declare_parameter('head_depth_topic', '/head/masked/depth_raw')
        self.declare_parameter('head_rgb_topic', '/head/masked/image_raw')
        self.declare_parameter('head_camera_info_topic', '/camera/head/color/camera_info')
        self.declare_parameter('exo_depth_topic', '/exo/masked/depth_raw')
        self.declare_parameter('exo_rgb_topic', '/exo/masked/image_raw')
        self.declare_parameter('exo_camera_info_topic', '/camera/exo/color/camera_info')

        self.global_frame = self.get_parameter('global_frame').value
        self.voxel_size_m = float(self.get_parameter('voxel_size_m').value)
        self.max_integration_distance_m = float(self.get_parameter('max_integration_distance_m').value)
        self.mesh_update_period = int(self.get_parameter('mesh_update_period').value)
        self.costmap_resolution = float(self.get_parameter('costmap_resolution').value)
        self.costmap_height_min = float(self.get_parameter('costmap_height_min').value)
        self.costmap_height_max = float(self.get_parameter('costmap_height_max').value)

        self.head_depth_topic = self.get_parameter('head_depth_topic').value
        self.head_rgb_topic = self.get_parameter('head_rgb_topic').value
        self.head_camera_info_topic = self.get_parameter('head_camera_info_topic').value
        self.exo_depth_topic = self.get_parameter('exo_depth_topic').value
        self.exo_rgb_topic = self.get_parameter('exo_rgb_topic').value
        self.exo_camera_info_topic = self.get_parameter('exo_camera_info_topic').value

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        params = MapperParams()
        proj_params = params.get_projective_integrator_params()
        proj_params.projective_integrator_max_integration_distance_m = self.max_integration_distance_m
        params.set_projective_integrator_params(proj_params)

        self.mapper = Mapper(
            voxel_sizes_m=self.voxel_size_m,
            integrator_types=ProjectiveIntegratorType.TSDF,
            mapper_parameters=params
        )

        self.head_sensor = None
        self.exo_sensor = None
        self.frame_count = 0
        self.mesh_vertices = None
        self.mesh_colors = None
        self._last_poses = {}

        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # Often needed for high-bw streams
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.head_depth_sub = message_filters.Subscriber(self, Image, self.head_depth_topic, qos_profile=qos)
        self.head_rgb_sub = message_filters.Subscriber(self, Image, self.head_rgb_topic, qos_profile=qos)
        self.head_info_sub = message_filters.Subscriber(self, CameraInfo, self.head_camera_info_topic, qos_profile=qos)

        self.head_ts = message_filters.ApproximateTimeSynchronizer(
            [self.head_depth_sub, self.head_rgb_sub, self.head_info_sub],
            queue_size=100,
            slop=0.1
        )
        self.head_ts.registerCallback(self.head_callback)

        self.exo_depth_sub = message_filters.Subscriber(self, Image, self.exo_depth_topic, qos_profile=qos)
        self.exo_rgb_sub = message_filters.Subscriber(self, Image, self.exo_rgb_topic, qos_profile=qos)
        self.exo_info_sub = message_filters.Subscriber(self, CameraInfo, self.exo_camera_info_topic, qos_profile=qos)

        self.exo_ts = message_filters.ApproximateTimeSynchronizer(
            [self.exo_depth_sub, self.exo_rgb_sub, self.exo_info_sub],
            queue_size=100,
            slop=0.1
        )
        self.exo_ts.registerCallback(self.exo_callback)

        self.mesh_pub = self.create_publisher(Marker, '/nvblox/mesh', pub_qos)
        self.pcl_pub = self.create_publisher(PointCloud2, '/nvblox/pointcloud', pub_qos)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/nvblox/costmap', pub_qos)

        self.get_logger().info(f'[{self.instance_id}] NVBlox node online.')
        self.get_logger().info(f'  -> Voxel size: {self.voxel_size_m}m')
        self.get_logger().info(f'  -> Max integration distance: {self.max_integration_distance_m}m')
        self.get_logger().info(f'  -> Mesh update every {self.mesh_update_period} frames')
        self.get_logger().info(f'  -> Head depth: {self.head_depth_topic}')
        self.get_logger().info(f'  -> Exo depth: {self.exo_depth_topic}')

    def _create_sensor(self, info_msg: CameraInfo) -> Sensor:
        K = np.array(info_msg.k).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        width, height = info_msg.width, info_msg.height
        return Sensor.from_camera(fu=fx, fv=fy, cu=cx, cv=cy, width=width, height=height)

    def _get_pose(self, frame_id: str, stamp) -> torch.Tensor:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                frame_id,
                stamp,
                rclpy.duration.Duration(seconds=0.5)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            if frame_id in self._last_poses:
                self.get_logger().warn(
                    f'TF lookup failed for {frame_id}, using cached pose: {str(e)}',
                    throttle_duration_sec=10.0)
                return self._last_poses[frame_id]
            self.get_logger().warn(
                f'TF lookup failed for {frame_id}, no cached pose: {str(e)}',
                throttle_duration_sec=10.0)
            return None
        except Exception as e:
            if frame_id in self._last_poses:
                self.get_logger().warn(
                    f'Unexpected TF error for {frame_id}, using cached pose: {str(e)}',
                    throttle_duration_sec=10.0)
                return self._last_poses[frame_id]
            self.get_logger().error(f'Unexpected TF error: {str(e)}')
            return None

        q = [transform.transform.rotation.x, transform.transform.rotation.y,
             transform.transform.rotation.z, transform.transform.rotation.w]
        t = [transform.transform.translation.x, transform.transform.translation.y,
             transform.transform.translation.z]

        rot = R.from_quat(q).as_matrix()
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot
        pose[:3, 3] = t
        pose_tensor = torch.from_numpy(pose)
        self._last_poses[frame_id] = pose_tensor
        return pose_tensor

    def _integrate_frame(self, depth_msg: Image, rgb_msg: Image, info_msg: CameraInfo, sensor_attr: str):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1').copy()
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8').copy()

            depth_cv = np.nan_to_num(depth_cv, nan=0.0, posinf=0.0, neginf=0.0)

            sensor = getattr(self, sensor_attr)
            if sensor is None:
                sensor = self._create_sensor(info_msg)
                setattr(self, sensor_attr, sensor)

            pose = self._get_pose(depth_msg.header.frame_id, depth_msg.header.stamp)
            if pose is None:
                return

            depth_gpu = torch.from_numpy(depth_cv).float().cuda()
            rgb_gpu = torch.from_numpy(rgb_cv).byte().cuda()

            self.mapper.add_depth_frame(depth_gpu, pose, sensor)
            self.mapper.add_color_frame(rgb_gpu, pose, sensor)

            self.frame_count += 1
            if self.frame_count % self.mesh_update_period == 0:
                self._update_and_publish()

        except Exception as e:
            self.get_logger().error(f'[{self.instance_id}] Integration error: {str(e)}')

    def head_callback(self, depth_msg: Image, rgb_msg: Image, info_msg: CameraInfo):
        self._integrate_frame(depth_msg, rgb_msg, info_msg, 'head_sensor')

    def exo_callback(self, depth_msg: Image, rgb_msg: Image, info_msg: CameraInfo):
        self._integrate_frame(depth_msg, rgb_msg, info_msg, 'exo_sensor')

    def _update_and_publish(self):
        try:
            self.mapper.update_color_mesh()
            mesh = self.mapper.get_color_mesh()

            vertices = mesh.vertices().cpu().numpy()
            triangles = mesh.triangles().cpu().numpy()
            colors = mesh.vertex_colors().cpu().numpy()

            if len(vertices) == 0:
                return

            self.mesh_vertices = vertices
            self.mesh_colors = colors

            self._publish_mesh(vertices, triangles, colors)
            self._publish_pointcloud(vertices, colors)
            self._publish_costmap(vertices)

            if self.frame_count % (self.mesh_update_period * 10) == 0:
                self.get_logger().info(f'[{self.instance_id}] Mesh: {len(vertices)} vertices, {len(triangles)} triangles')

        except Exception as e:
            self.get_logger().error(f'[{self.instance_id}] Mesh update error: {str(e)}')

    def _publish_mesh(self, vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'nvblox_mesh'
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        points = []
        mesh_colors = []
        for tri in triangles:
            for idx in tri:
                if idx < len(vertices):
                    v = vertices[idx]
                    points.append(Point(x=float(v[0]), y=float(v[1]), z=float(v[2])))
                    if idx < len(colors):
                        c = colors[idx]
                        mesh_colors.append(ColorRGBA(r=float(c[0])/255.0, g=float(c[1])/255.0, b=float(c[2])/255.0, a=1.0))
                    else:
                        mesh_colors.append(ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))

        marker.points = points
        marker.colors = mesh_colors
        self.mesh_pub.publish(marker)

    def _publish_pointcloud(self, vertices: np.ndarray, colors: np.ndarray):
        num_points = len(vertices)
        data = np.zeros(num_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        data['x'] = vertices[:, 0]
        data['y'] = vertices[:, 1]
        data['z'] = vertices[:, 2]

        rgb_packed = (colors[:, 0].astype(np.uint32) << 16) | \
                     (colors[:, 1].astype(np.uint32) << 8) | \
                     (colors[:, 2].astype(np.uint32))
        data['rgb'] = rgb_packed

        pcl_msg = PointCloud2()
        pcl_msg.header.frame_id = self.global_frame
        pcl_msg.header.stamp = self.get_clock().now().to_msg()
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
        self.pcl_pub.publish(pcl_msg)

    def _publish_costmap(self, vertices: np.ndarray):
        mask = (vertices[:, 2] >= self.costmap_height_min) & (vertices[:, 2] <= self.costmap_height_max)
        filtered = vertices[mask]

        if len(filtered) == 0:
            return

        min_x, min_y = filtered[:, 0].min(), filtered[:, 1].min()
        max_x, max_y = filtered[:, 0].max(), filtered[:, 1].max()

        width = int((max_x - min_x) / self.costmap_resolution) + 1
        height = int((max_y - min_y) / self.costmap_resolution) + 1

        if width <= 0 or height <= 0 or width > 10000 or height > 10000:
            return

        grid = np.zeros((height, width), dtype=np.int8)
        grid[:, :] = -1

        for v in filtered:
            col = int((v[0] - min_x) / self.costmap_resolution)
            row = int((v[1] - min_y) / self.costmap_resolution)
            if 0 <= col < width and 0 <= row < height:
                grid[row, col] = 100

        costmap = OccupancyGrid()
        costmap.header.frame_id = self.global_frame
        costmap.header.stamp = self.get_clock().now().to_msg()
        costmap.info.resolution = self.costmap_resolution
        costmap.info.width = width
        costmap.info.height = height
        costmap.info.origin.position.x = float(min_x)
        costmap.info.origin.position.y = float(min_y)
        costmap.info.origin.position.z = 0.0
        costmap.info.origin.orientation.w = 1.0
        costmap.data = grid.flatten().tolist()
        self.costmap_pub.publish(costmap)


def main(args=None):
    rclpy.init(args=args)
    node = NvbloxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
