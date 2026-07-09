#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <System.h>
#include <ImuTypes.h>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Geometry>

class Orbslam3RgbdImuNode final : public rclcpp::Node {
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image, CameraInfo>;

public:
  explicit Orbslam3RgbdImuNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("orbslam3_exo", options),
    rgb_sub_(this, declare_parameter<std::string>("rgb_topic", "/exo/masked/image_raw")),
    depth_sub_(this, declare_parameter<std::string>("depth_topic", "/exo/masked/depth_raw")),
    info_sub_(this, declare_parameter<std::string>("camera_info_topic", "/camera/exo/color/camera_info")) {
    vocabulary_path_ = declare_parameter<std::string>("vocabulary_path", "");
    settings_path_ = declare_parameter<std::string>("settings_path", "");
    mode_ = declare_parameter<std::string>("mode", "rgbd_imu");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/exo/odom");
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    odom_frame_id_ = declare_parameter<std::string>("odom_frame_id", "odom");
    base_frame_id_ = declare_parameter<std::string>("base_frame_id", "exo_link");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    fail_without_imu_ = declare_parameter<bool>("fail_without_imu", true);
    imu_time_offset_s_ = declare_parameter<double>("imu_time_offset_s", 0.0);

    if (vocabulary_path_.empty() || settings_path_.empty()) {
      throw std::runtime_error("vocabulary_path and settings_path are required");
    }

    const bool inertial = mode_ == "rgbd_imu";
    if (mode_ != "rgbd" && !inertial) {
      throw std::runtime_error("mode must be rgbd or rgbd_imu");
    }

    auto sensor = ORB_SLAM3::System::RGBD;
#ifdef ORB_SLAM3_HAS_IMU_RGBD
    if (inertial) {
      sensor = ORB_SLAM3::System::IMU_RGBD;
    }
#else
    if (inertial) {
      throw std::runtime_error("This ORB-SLAM3 build does not expose IMU_RGBD; use mode:=rgbd or an RGB-D-inertial ORB-SLAM3 fork.");
    }
#endif

    slam_ = std::make_unique<ORB_SLAM3::System>(vocabulary_path_, settings_path_, sensor, false);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);

    if (inertial) {
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Imu::SharedPtr msg) { imu_callback(std::move(msg)); });
    }

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(30), rgb_sub_, depth_sub_, info_sub_);
    sync_->registerCallback(
      std::bind(&Orbslam3RgbdImuNode::rgbd_callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));

    RCLCPP_INFO(get_logger(), "ORB-SLAM3 wrapper online in %s mode.", mode_.c_str());
  }

  ~Orbslam3RgbdImuNode() override {
    if (slam_) {
      slam_->Shutdown();
    }
  }

private:
  static double stamp_to_sec(const builtin_interfaces::msg::Time & stamp) {
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
  }

  void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_queue_.push_back(*msg);
    while (imu_queue_.size() > 2000) {
      imu_queue_.pop_front();
    }
  }

  std::vector<ORB_SLAM3::IMU::Point> pop_imu_until(double stamp_s) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    std::vector<ORB_SLAM3::IMU::Point> out;
    while (!imu_queue_.empty()) {
      const auto & msg = imu_queue_.front();
      const double imu_t = stamp_to_sec(msg.header.stamp) + imu_time_offset_s_;
      if (imu_t > stamp_s) {
        break;
      }
      out.emplace_back(
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
        imu_t);
      imu_queue_.pop_front();
    }
    return out;
  }

  void rgbd_callback(
    const Image::ConstSharedPtr & rgb_msg,
    const Image::ConstSharedPtr & depth_msg,
    const CameraInfo::ConstSharedPtr &) {
    const double stamp_s = stamp_to_sec(rgb_msg->header.stamp);
    auto imu = pop_imu_until(stamp_s);
    if (mode_ == "rgbd_imu" && fail_without_imu_ && imu.empty()) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 3000, "Waiting for IMU samples before RGB-D frame.");
      return;
    }

    cv::Mat rgb = cv_bridge::toCvShare(rgb_msg, "bgr8")->image;
    cv::Mat depth = cv_bridge::toCvShare(depth_msg, "32FC1")->image;
    auto t_cw = slam_->TrackRGBD(rgb, depth, stamp_s, imu);
    const Eigen::Matrix3f r_wc = t_cw.rotationMatrix().transpose();
    const Eigen::Vector3f t_wc = -r_wc * t_cw.translation();

    publish_pose(rgb_msg->header.stamp, r_wc, t_wc);
  }

  void publish_pose(
    const builtin_interfaces::msg::Time & stamp,
    const Eigen::Matrix3f & r_wc,
    const Eigen::Vector3f & t_wc) {
    const Eigen::Quaternionf q(r_wc);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = base_frame_id_;
    odom.pose.pose.position.x = t_wc.x();
    odom.pose.pose.position.y = t_wc.y();
    odom.pose.pose.position.z = t_wc.z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom_pub_->publish(odom);

    if (!publish_tf_) {
      return;
    }
    geometry_msgs::msg::TransformStamped tf;
    tf.header = odom.header;
    tf.child_frame_id = base_frame_id_;
    tf.transform.translation.x = odom.pose.pose.position.x;
    tf.transform.translation.y = odom.pose.pose.position.y;
    tf.transform.translation.z = odom.pose.pose.position.z;
    tf.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);
  }

  std::string vocabulary_path_;
  std::string settings_path_;
  std::string mode_;
  std::string imu_topic_;
  std::string odom_topic_;
  std::string map_frame_id_;
  std::string odom_frame_id_;
  std::string base_frame_id_;
  bool publish_tf_{true};
  bool fail_without_imu_{true};
  double imu_time_offset_s_{0.0};

  std::unique_ptr<ORB_SLAM3::System> slam_;
  message_filters::Subscriber<Image> rgb_sub_;
  message_filters::Subscriber<Image> depth_sub_;
  message_filters::Subscriber<CameraInfo> info_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::mutex imu_mutex_;
  std::deque<sensor_msgs::msg::Imu> imu_queue_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Orbslam3RgbdImuNode>());
  rclcpp::shutdown();
  return 0;
}
