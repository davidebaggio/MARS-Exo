import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String
import numpy as np
from collections import deque
from exo_head_slam.utils.imu_utils import (
    estimate_gravity,
    detect_motion,
    estimate_gyro_bias,
)


class IMUBuffer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.accel = deque(maxlen=window_size)
        self.gyro = deque(maxlen=window_size)
        self.stamps = deque(maxlen=window_size)
        self.last_angular_velocity = np.zeros(3)

    def add(self, msg: Imu):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        self.accel.append(np.array([ax, ay, az]))
        self.gyro.append(np.array([gx, gy, gz]))
        self.stamps.append(msg.header.stamp)
        self.last_angular_velocity = np.array([gx, gy, gz])

    def get_recent_accel(self, n: int = 50) -> np.ndarray:
        if len(self.accel) == 0:
            return np.array([]).reshape(0, 3)
        n = min(n, len(self.accel))
        return np.array(list(self.accel)[-n:])

    def get_recent_gyro(self, n: int = 50) -> np.ndarray:
        if len(self.gyro) == 0:
            return np.array([]).reshape(0, 3)
        n = min(n, len(self.gyro))
        return np.array(list(self.gyro)[-n:])

    def latest_stamp(self):
        return self.stamps[-1] if self.stamps else None


class IMUIntegratorNode(Node):
    def __init__(self):
        super().__init__('imu_integrator_node')

        self.declare_parameter('head_imu_topic', '/camera/head/imu')
        self.declare_parameter('exo_imu_topic', '/camera/exo/imu')
        self.declare_parameter('window_size', 100)
        self.declare_parameter('gravity_alpha', 0.1)
        self.declare_parameter('motion_gyro_threshold', 0.3)
        self.declare_parameter('motion_accel_var_threshold', 0.5)
        self.declare_parameter('bias_estimation_samples', 100)
        self.declare_parameter('process_rate_hz', 10.0)

        window_size = int(self.get_parameter('window_size').value)
        self.gravity_alpha = float(self.get_parameter('gravity_alpha').value)
        self.motion_gyro_threshold = float(self.get_parameter('motion_gyro_threshold').value)
        self.motion_accel_var_threshold = float(self.get_parameter('motion_accel_var_threshold').value)
        self.bias_estimation_samples = int(self.get_parameter('bias_estimation_samples').value)

        self.head_buffer = IMUBuffer(window_size)
        self.exo_buffer = IMUBuffer(window_size)

        self.head_gravity = np.array([0.0, 0.0, 1.0])
        self.exo_gravity = np.array([0.0, 0.0, 1.0])
        self.head_bias = np.zeros(3)
        self.exo_bias = np.zeros(3)
        self.head_bias_samples = []
        self.exo_bias_samples = []
        self.head_bias_initialized = False
        self.exo_bias_initialized = False

        self.head_imu_sub = self.create_subscription(
            Imu, self.get_parameter('head_imu_topic').value,
            lambda msg: self.imu_callback(msg, 'head'),
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10))
        self.exo_imu_sub = self.create_subscription(
            Imu, self.get_parameter('exo_imu_topic').value,
            lambda msg: self.imu_callback(msg, 'exo'),
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10))

        self.head_gravity_pub = self.create_publisher(Vector3Stamped, '/imu/head/gravity', 10)
        self.exo_gravity_pub = self.create_publisher(Vector3Stamped, '/imu/exo/gravity', 10)
        self.head_gyro_pub = self.create_publisher(Vector3Stamped, '/imu/head/gyro_filtered', 10)
        self.exo_gyro_pub = self.create_publisher(Vector3Stamped, '/imu/exo/gyro_filtered', 10)
        self.head_motion_pub = self.create_publisher(String, '/imu/head/motion_state', 10)
        self.exo_motion_pub = self.create_publisher(String, '/imu/exo/motion_state', 10)
        self.head_bias_pub = self.create_publisher(Vector3Stamped, '/imu/head/gyro_bias', 10)
        self.exo_bias_pub = self.create_publisher(Vector3Stamped, '/imu/exo/gyro_bias', 10)

        rate = float(self.get_parameter('process_rate_hz').value)
        self.create_timer(1.0 / rate, self.process_imu)

        self.get_logger().info('IMU integrator node online.')

    def imu_callback(self, msg: Imu, camera: str):
        buf = self.head_buffer if camera == 'head' else self.exo_buffer
        buf.add(msg)

    def process_imu(self):
        for camera in ['head', 'exo']:
            buf = self.head_buffer if camera == 'head' else self.exo_buffer
            stamp = buf.latest_stamp()
            if stamp is None:
                continue

            accel = buf.get_recent_accel(50)
            gyro = buf.get_recent_gyro(50)

            gravity = estimate_gravity(accel)
            self._update_gravity(camera, gravity)

            moving = detect_motion(
                gyro, accel,
                self.motion_gyro_threshold,
                self.motion_accel_var_threshold,
            )

            self._update_bias(camera, gyro, moving)
            self._publish_all(camera, stamp, gravity, gyro, moving)

    def _update_gravity(self, camera: str, gravity: np.ndarray):
        if camera == 'head':
            self.head_gravity = ((1.0 - self.gravity_alpha) * self.head_gravity +
                                  self.gravity_alpha * gravity)
            self.head_gravity /= np.linalg.norm(self.head_gravity)
        else:
            self.exo_gravity = ((1.0 - self.gravity_alpha) * self.exo_gravity +
                                 self.gravity_alpha * gravity)
            self.exo_gravity /= np.linalg.norm(self.exo_gravity)

    def _update_bias(self, camera: str, gyro: np.ndarray, moving: bool):
        samples = self.head_bias_samples if camera == 'head' else self.exo_bias_samples
        initialized = self.head_bias_initialized if camera == 'head' else self.exo_bias_initialized
        if not moving and len(gyro) > 0:
            samples.append(np.mean(gyro[-5:], axis=0))
            if len(samples) >= self.bias_estimation_samples and not initialized:
                bias = estimate_gyro_bias(np.array(samples))
                if camera == 'head':
                    self.head_bias = bias
                    self.head_bias_initialized = True
                    self.head_bias_samples = []
                else:
                    self.exo_bias = bias
                    self.exo_bias_initialized = True
                    self.exo_bias_samples = []
                self.get_logger().info(f'{camera} gyro bias estimated: {bias}')

    def _publish_all(self, camera: str, stamp, gravity: np.ndarray, gyro: np.ndarray, moving: bool):
        sec = stamp.sec
        nanosec = stamp.nanosec

        def make_vector(x, y, z):
            v = Vector3Stamped()
            v.header.stamp.sec = sec
            v.header.stamp.nanosec = nanosec
            v.vector.x = float(x)
            v.vector.y = float(y)
            v.vector.z = float(z)
            return v

        if camera == 'head':
            g_msg = make_vector(self.head_gravity[0], self.head_gravity[1], self.head_gravity[2])
            g_msg.header.frame_id = 'head_color_optical_frame'
            self.head_gravity_pub.publish(g_msg)

            if len(gyro) > 0:
                gyro_mean = np.mean(gyro, axis=0)
                gyro_msg = make_vector(gyro_mean[0], gyro_mean[1], gyro_mean[2])
                gyro_msg.header.frame_id = 'head_color_optical_frame'
                self.head_gyro_pub.publish(gyro_msg)

            motion_msg = String()
            motion_msg.data = 'stationary' if not moving else 'moving'
            self.head_motion_pub.publish(motion_msg)

            bias_msg = make_vector(self.head_bias[0], self.head_bias[1], self.head_bias[2])
            bias_msg.header.frame_id = 'head_color_optical_frame'
            self.head_bias_pub.publish(bias_msg)
        else:
            g_msg = make_vector(self.exo_gravity[0], self.exo_gravity[1], self.exo_gravity[2])
            g_msg.header.frame_id = 'exo_color_optical_frame'
            self.exo_gravity_pub.publish(g_msg)

            if len(gyro) > 0:
                gyro_mean = np.mean(gyro, axis=0)
                gyro_msg = make_vector(gyro_mean[0], gyro_mean[1], gyro_mean[2])
                gyro_msg.header.frame_id = 'exo_color_optical_frame'
                self.exo_gyro_pub.publish(gyro_msg)

            motion_msg = String()
            motion_msg.data = 'stationary' if not moving else 'moving'
            self.exo_motion_pub.publish(motion_msg)

            bias_msg = make_vector(self.exo_bias[0], self.exo_bias[1], self.exo_bias[2])
            bias_msg.header.frame_id = 'exo_color_optical_frame'
            self.exo_bias_pub.publish(bias_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IMUIntegratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
