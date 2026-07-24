#include <algorithm>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#include <System.h>

namespace
{
struct ImuSample
{
  double stamp = 0.0;
  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;
};

static Eigen::Matrix4d transform_to_matrix(const geometry_msgs::msg::TransformStamped & tf)
{
  const auto & q = tf.transform.rotation;
  const auto & t = tf.transform.translation;

  Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
  quat.normalize();

  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  matrix.topLeftCorner<3, 3>() = quat.toRotationMatrix();
  matrix(0, 3) = t.x;
  matrix(1, 3) = t.y;
  matrix(2, 3) = t.z;
  return matrix;
}

static geometry_msgs::msg::TransformStamped matrix_to_transform(
  const Eigen::Matrix4d & matrix, const std::string & parent, const std::string & child,
  const builtin_interfaces::msg::Time & stamp)
{
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = stamp;
  tf.header.frame_id = parent;
  tf.child_frame_id = child;

  const Eigen::Matrix3d rot = matrix.topLeftCorner<3, 3>();
  const Eigen::Quaterniond quat(rot);
  tf.transform.translation.x = matrix(0, 3);
  tf.transform.translation.y = matrix(1, 3);
  tf.transform.translation.z = matrix(2, 3);
  tf.transform.rotation.x = quat.x();
  tf.transform.rotation.y = quat.y();
  tf.transform.rotation.z = quat.z();
  tf.transform.rotation.w = quat.w();
  return tf;
}

static cv::Mat depth_message_to_cv(const sensor_msgs::msg::Image & msg, double depth_scale)
{
  auto depth = cv_bridge::toCvCopy(msg, msg.encoding)->image;
  if (msg.encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    cv::Mat depth32f;
    depth.convertTo(depth32f, CV_32F, depth_scale);
    return depth32f;
  }
  if (msg.encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
    throw std::runtime_error("Unsupported depth encoding: " + msg.encoding);
  }
  if (depth.type() != CV_32FC1) {
    cv::Mat depth32f;
    depth.convertTo(depth32f, CV_32F);
    return depth32f;
  }
  return depth;
}
}  // namespace

class Orbslam3RgbdImuNode : public rclcpp::Node
{
public:
  Orbslam3RgbdImuNode()
  : Node("orbslam3_rgbd_imu_node")
  {
    declare_parameter<std::string>("vocabulary_path", "");
    declare_parameter<std::string>("settings_path", "");
    declare_parameter<std::string>("rgb_topic", "/exo/masked/image_raw");
    declare_parameter<std::string>("depth_topic", "/exo/masked/depth_raw");
    declare_parameter<std::string>("imu_topic", "/camera/exo/imu");
    declare_parameter<std::string>("camera_info_topic", "/camera/exo/color/camera_info");
    declare_parameter<std::string>("odom_frame_id", "odom");
    declare_parameter<std::string>("base_frame_id", "exo_link");
    declare_parameter<std::string>("camera_frame_id", "");
    declare_parameter<bool>("publish_tf", true);
    declare_parameter<double>("depth_scale", 1.0);
    declare_parameter<double>("imu_timeout_sec", 15.0);
    declare_parameter<std::string>("slam_mode", "IMU_RGBD");

    vocabulary_path_ = get_parameter("vocabulary_path").as_string();
    settings_path_ = get_parameter("settings_path").as_string();
    rgb_topic_ = get_parameter("rgb_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();
    odom_frame_id_ = get_parameter("odom_frame_id").as_string();
    base_frame_id_ = get_parameter("base_frame_id").as_string();
    camera_frame_id_ = get_parameter("camera_frame_id").as_string();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    depth_scale_ = get_parameter("depth_scale").as_double();
    imu_timeout_sec_ = get_parameter("imu_timeout_sec").as_double();
    slam_mode_ = get_parameter("slam_mode").as_string();

    if (vocabulary_path_.empty() || settings_path_.empty()) {
      throw std::runtime_error("vocabulary_path and settings_path are required");
    }
    if (!std::filesystem::exists(vocabulary_path_)) {
      throw std::runtime_error("ORB vocabulary not found: " + vocabulary_path_);
    }
    if (!std::filesystem::exists(settings_path_)) {
      throw std::runtime_error("ORB settings not found: " + settings_path_);
    }

    auto orb_settings = load_and_convert_settings(settings_path_);
    generated_settings_path_ = write_orb_settings_file(orb_settings);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_buffer_->setUsingDedicatedThread(true);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      *tf_buffer_, get_node_base_interface(), get_node_logging_interface(),
      get_node_parameters_interface(), get_node_topics_interface(), true);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/exo/odom", 10);

    using ApproxPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;
    rgb_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
      this, rgb_topic_, rmw_qos_profile_sensor_data);
    depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
      this, depth_topic_, rmw_qos_profile_sensor_data);
    info_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(
      this, camera_info_topic_, rmw_qos_profile_sensor_data);
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxPolicy>>(
      ApproxPolicy(10), *rgb_sub_, *depth_sub_, *info_sub_);
    sync_->registerCallback(
      std::bind(&Orbslam3RgbdImuNode::rgbd_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&Orbslam3RgbdImuNode::imu_callback, this, std::placeholders::_1));

    const auto sensor_mode = (slam_mode_ == "RGBD")
      ? ORB_SLAM3::System::RGBD
      : ORB_SLAM3::System::IMU_RGBD;
    slam_ = std::make_unique<ORB_SLAM3::System>(
      vocabulary_path_, generated_settings_path_, sensor_mode, false);

    startup_wall_time_ = std::chrono::steady_clock::now();
    startup_timer_ = create_wall_timer(
      std::chrono::seconds(1), std::bind(&Orbslam3RgbdImuNode::startup_watchdog, this));

    RCLCPP_INFO(get_logger(), "ORB-SLAM3 %s node ready", slam_mode_.c_str());
  }

  ~Orbslam3RgbdImuNode() override
  {
    if (slam_) {
      slam_->Shutdown();
    }
  }

private:
  struct OrbSettings
  {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double bf = 0.0;
    double th_depth = 40.0;
    int width = 0;
    int height = 0;
    int fps = 30;
    double depth_map_factor = 1.0;
    Eigen::Matrix4d t_b_c1 = Eigen::Matrix4d::Identity();
    double imu_noise_gyro = 1e-2;
    double imu_noise_acc = 1e-1;
    double imu_gyro_walk = 1e-6;
    double imu_acc_walk = 1e-4;
    int imu_frequency = 200;
    int orb_features = 1250;
    double orb_scale_factor = 1.2;
    int orb_levels = 8;
    int orb_ini_fast = 20;
    int orb_min_fast = 7;
  };

  OrbSettings load_and_convert_settings(const std::string & path)
  {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      throw std::runtime_error("Failed to open ORB settings: " + path);
    }

    auto read_double = [&fs](const char * key) {
        cv::FileNode node = fs[key];
        if (node.empty()) {
          throw std::runtime_error(std::string("Missing settings key: ") + key);
        }
        return static_cast<double>(node);
      };
    auto read_int = [&fs](const char * key) {
        cv::FileNode node = fs[key];
        if (node.empty()) {
          throw std::runtime_error(std::string("Missing settings key: ") + key);
        }
        return static_cast<int>(node);
      };

    OrbSettings settings;
    settings.fx = read_double("Camera.fx");
    settings.fy = read_double("Camera.fy");
    settings.cx = read_double("Camera.cx");
    settings.cy = read_double("Camera.cy");
    settings.bf = read_double("Camera.bf");
    settings.th_depth = read_double("ThDepth");
    settings.width = read_int("Camera.width");
    settings.height = read_int("Camera.height");
    settings.fps = read_int("Camera.fps");
    settings.depth_map_factor = read_double("RGBD.DepthMapFactor");
    settings.imu_frequency = read_int("IMU.Frequency");
    settings.imu_noise_gyro = read_double("IMU.NoiseGyro");
    settings.imu_noise_acc = read_double("IMU.NoiseAcc");
    settings.imu_gyro_walk = read_double("IMU.GyroWalk");
    settings.imu_acc_walk = read_double("IMU.AccWalk");
    settings.orb_features = read_int("ORBextractor.nFeatures");
    settings.orb_scale_factor = read_double("ORBextractor.scaleFactor");
    settings.orb_levels = read_int("ORBextractor.nLevels");
    settings.orb_ini_fast = read_int("ORBextractor.iniThFAST");
    settings.orb_min_fast = read_int("ORBextractor.minThFAST");

    cv::FileNode imu_tbc = fs["IMU.T_b_c1"];
    if (imu_tbc.empty() || !imu_tbc.isMap()) {
      throw std::runtime_error("Missing settings key: IMU.T_b_c1");
    }
    cv::Mat tbc;
    imu_tbc >> tbc;
    if (tbc.empty() || tbc.rows != 4 || tbc.cols != 4) {
      throw std::runtime_error("IMU.T_b_c1 must be a 4x4 matrix");
    }
    const bool use_float = tbc.type() == CV_32F || tbc.type() == CV_32FC1;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        settings.t_b_c1(r, c) = use_float ? static_cast<double>(tbc.at<float>(r, c))
                                          : tbc.at<double>(r, c);
      }
    }
    return settings;
  }

  std::string write_orb_settings_file(const OrbSettings & settings)
  {
    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path = temp_dir / ("orbslam3_ros2_generated_" + suffix + ".yaml");
    std::ofstream out(path);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to write generated ORB settings: " + path.string());
    }

    auto fp = [&out](double v) { out << std::fixed << v; };
    out << "%YAML:1.0\n";
    out << "File.version: \"1.0\"\n";
    out << "Camera.type: \"PinHole\"\n";
    out << "Camera1.fx: "; fp(settings.fx); out << "\n";
    out << "Camera1.fy: "; fp(settings.fy); out << "\n";
    out << "Camera1.cx: "; fp(settings.cx); out << "\n";
    out << "Camera1.cy: "; fp(settings.cy); out << "\n";
    out << "Camera1.k1: 0.0\n";
    out << "Camera1.k2: 0.0\n";
    out << "Camera1.p1: 0.0\n";
    out << "Camera1.p2: 0.0\n";
    out << "Camera.width: " << settings.width << "\n";
    out << "Camera.height: " << settings.height << "\n";
    out << "Camera.fps: " << settings.fps << "\n";
    out << "Camera.RGB: 0\n";
    out << "Stereo.ThDepth: "; fp(settings.th_depth); out << "\n";
    out << "Stereo.b: "; fp(settings.bf / std::max(settings.fx, 1e-6)); out << "\n";
    out << "RGBD.DepthMapFactor: "; fp(settings.depth_map_factor); out << "\n";
    out << "IMU.T_b_c1: !!opencv-matrix\n";
    out << "   rows: 4\n   cols: 4\n   dt: f\n   data: [";
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out << static_cast<float>(settings.t_b_c1(r, c));
        if (!(r == 3 && c == 3)) {
          out << ", ";
        }
      }
      if (r < 3) {
        out << "\n         ";
      }
    }
    out << "]\n";
    out << "IMU.InsertKFsWhenLost: 0\n";
    out << "IMU.NoiseGyro: "; fp(settings.imu_noise_gyro); out << "\n";
    out << "IMU.NoiseAcc: "; fp(settings.imu_noise_acc); out << "\n";
    out << "IMU.GyroWalk: "; fp(settings.imu_gyro_walk); out << "\n";
    out << "IMU.AccWalk: "; fp(settings.imu_acc_walk); out << "\n";
    out << "IMU.Frequency: "; fp(settings.imu_frequency); out << "\n";
    out << "ORBextractor.nFeatures: " << settings.orb_features << "\n";
    out << "ORBextractor.scaleFactor: "; fp(settings.orb_scale_factor); out << "\n";
    out << "ORBextractor.nLevels: " << settings.orb_levels << "\n";
    out << "ORBextractor.iniThFAST: " << settings.orb_ini_fast << "\n";
    out << "ORBextractor.minThFAST: " << settings.orb_min_fast << "\n";
    out << "Viewer.KeyFrameSize: 0.05\n";
    out << "Viewer.KeyFrameLineWidth: 1.0\n";
    out << "Viewer.GraphLineWidth: 0.9\n";
    out << "Viewer.PointSize: 2.0\n";
    out << "Viewer.CameraSize: 0.08\n";
    out << "Viewer.CameraLineWidth: 3.0\n";
    out << "Viewer.ViewpointX: 0.0\n";
    out << "Viewer.ViewpointY: -0.7\n";
    out << "Viewer.ViewpointZ: -3.5\n";
    out << "Viewer.ViewpointF: 500.0\n";
    out.close();
    return path.string();
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_seen_ = true;

    ImuSample sample;
    sample.stamp = rclcpp::Time(msg->header.stamp).seconds();
    sample.ax = static_cast<float>(msg->linear_acceleration.x);
    sample.ay = static_cast<float>(msg->linear_acceleration.y);
    sample.az = static_cast<float>(msg->linear_acceleration.z);
    sample.gx = static_cast<float>(msg->angular_velocity.x);
    sample.gy = static_cast<float>(msg->angular_velocity.y);
    sample.gz = static_cast<float>(msg->angular_velocity.z);
    imu_buffer_.push_back(sample);

    const double cutoff = sample.stamp - 5.0;
    while (!imu_buffer_.empty() && imu_buffer_.front().stamp < cutoff) {
      imu_buffer_.pop_front();
    }
  }

  std::vector<ORB_SLAM3::IMU::Point> collect_imu_samples(double stamp)
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    std::vector<ORB_SLAM3::IMU::Point> samples;
    while (!imu_buffer_.empty() && imu_buffer_.front().stamp <= stamp) {
      const auto sample = imu_buffer_.front();
      samples.emplace_back(
        sample.ax, sample.ay, sample.az, sample.gx, sample.gy, sample.gz, sample.stamp);
      imu_buffer_.pop_front();
    }
    return samples;
  }

  void rgbd_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg)
  {
    if (!rgbd_seen_) {
      rgbd_seen_ = true;
      startup_wall_time_ = std::chrono::steady_clock::now();
    }

    const double stamp = rclcpp::Time(depth_msg->header.stamp).seconds();
    std::vector<ORB_SLAM3::IMU::Point> imu_samples;

    if (slam_mode_ != "RGBD") {
      if (!imu_seen_) {
        throttle_warn("Waiting for IMU samples on " + imu_topic_ + " before feeding ORB-SLAM3");
        return;
      }
      imu_samples = collect_imu_samples(stamp);
      if (imu_samples.empty()) {
        throttle_warn("No IMU samples available for current frame; skipping ORB-SLAM3 update");
        return;
      }
    }

    cv::Mat rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8)->image.clone();
    cv::Mat depth = depth_message_to_cv(*depth_msg, depth_scale_);

    const Sophus::SE3f t_cw = slam_->TrackRGBD(rgb, depth, stamp, imu_samples);
    const auto tracking_state = slam_->GetTrackingState();
    if (tracking_state != ORB_SLAM3::Tracking::OK &&
        tracking_state != ORB_SLAM3::Tracking::OK_KLT) {
      throttle_warn("ORB-SLAM3 tracking lost for current frame");
      return;
    }

    // Compute camera→base static TF (exo_color_optical_frame → exo_link)
    std::string camera_frame = camera_frame_id_;
    if (camera_frame.empty()) {
      camera_frame = !info_msg->header.frame_id.empty() ? info_msg->header.frame_id : depth_msg->header.frame_id;
    }
    Eigen::Matrix4d t_cam_base = Eigen::Matrix4d::Identity();
    try {
      const auto tf = tf_buffer_->lookupTransform(
        camera_frame, base_frame_id_, tf2_ros::fromMsg(rgb_msg->header.stamp), tf2::durationFromSec(0.1));
      t_cam_base = transform_to_matrix(tf);
    } catch (const std::exception & ex) {
      throttle_warn(std::string("Waiting for TF ") + camera_frame + " -> " + base_frame_id_ + ": " + ex.what());
      return;
    }

    // Full odom→exo_link pose
    const Eigen::Matrix4d t_w_c = t_cw.inverse().matrix().cast<double>();
    const Eigen::Matrix4d t_w_base = t_w_c * t_cam_base;

    // Frame-to-frame continuity
    if (pose_initialized_) {
      const Eigen::Vector3d dp = t_w_base.block<3, 1>(0, 3) - last_pose_.block<3, 1>(0, 3);
      const double dist = dp.norm();
      const Eigen::Quaterniond q_new(t_w_base.topLeftCorner<3, 3>());
      const Eigen::Quaterniond q_old(last_pose_.topLeftCorner<3, 3>());
      const double dot = std::abs(q_new.dot(q_old));
      const double angle = 2.0 * std::acos(std::clamp(dot, 0.0, 1.0));
      if (dist > 2.0 || angle > 1.0) {
        throttle_warn(
          std::string("Large frame-to-frame change: t=") + std::to_string(dist) +
          "m angle=" + std::to_string(angle * 180.0 / M_PI) + "deg");
        last_pose_ = t_w_base;
        return;
      }
    }
    last_pose_ = t_w_base;
    pose_initialized_ = true;

    // Publish odometry + TF
    const Eigen::Quaterniond quat(t_w_base.topLeftCorner<3, 3>());
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = rgb_msg->header.stamp;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = base_frame_id_;
    odom.pose.pose.position.x = t_w_base(0, 3);
    odom.pose.pose.position.y = t_w_base(1, 3);
    odom.pose.pose.position.z = t_w_base(2, 3);
    odom.pose.pose.orientation.x = quat.x();
    odom.pose.pose.orientation.y = quat.y();
    odom.pose.pose.orientation.z = quat.z();
    odom.pose.pose.orientation.w = quat.w();
    odom_pub_->publish(odom);

    if (publish_tf_) {
      auto tf = matrix_to_transform(t_w_base, odom_frame_id_, base_frame_id_, rgb_msg->header.stamp);
      tf_broadcaster_->sendTransform(tf);
    }
    RCLCPP_INFO(
      get_logger(), "Publish %s -> %s t=(%.3f %.3f %.3f) q=(%.3f %.3f %.3f %.3f)",
      odom_frame_id_.c_str(), base_frame_id_.c_str(),
      t_w_base(0, 3), t_w_base(1, 3), t_w_base(2, 3),
      quat.x(), quat.y(), quat.z(), quat.w());
  }

  void throttle_warn(const std::string & msg)
  {
    const double now_sec = this->now().seconds();
    if (now_sec - last_warn_sec_ < 5.0) {
      return;
    }
    last_warn_sec_ = now_sec;
    RCLCPP_WARN(get_logger(), "%s", msg.c_str());
  }

  void startup_watchdog()
  {
    if (slam_mode_ == "RGBD" || !rgbd_seen_ || imu_seen_ || missing_imu_reported_) {
      return;
    }
    const auto elapsed = std::chrono::steady_clock::now() - startup_wall_time_;
    if (elapsed < std::chrono::duration<double>(imu_timeout_sec_)) {
      return;
    }
    missing_imu_reported_ = true;
    RCLCPP_ERROR(
      get_logger(),
      "No IMU received on %s after %.1f s in %s mode. Falling back — no IMU data available.",
      imu_topic_.c_str(), imu_timeout_sec_, slam_mode_.c_str());
  }

  std::string vocabulary_path_;
  std::string settings_path_;
  std::string generated_settings_path_;
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string imu_topic_;
  std::string camera_info_topic_;
  std::string odom_frame_id_;
  std::string base_frame_id_;
  std::string camera_frame_id_;
  bool publish_tf_ = true;
  double depth_scale_ = 1.0;
  double imu_timeout_sec_ = 15.0;
  std::string slam_mode_ = "IMU_RGBD";

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> info_sub_;
  std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  std::unique_ptr<ORB_SLAM3::System> slam_;
  std::deque<ImuSample> imu_buffer_;
  std::mutex imu_mutex_;
  bool imu_seen_ = false;
  bool rgbd_seen_ = false;
  bool missing_imu_reported_ = false;
  double last_warn_sec_ = 0.0;
  bool pose_initialized_ = false;
  Eigen::Matrix4d last_pose_ = Eigen::Matrix4d::Identity();
  std::chrono::steady_clock::time_point startup_wall_time_;
  rclcpp::TimerBase::SharedPtr startup_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Orbslam3RgbdImuNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
