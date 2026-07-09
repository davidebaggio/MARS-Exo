#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("orbslam3_rgbd_imu_stub");
  RCLCPP_FATAL(
    node->get_logger(),
    "ORB_SLAM3_ROOT is not configured or libORB_SLAM3.so is missing. "
    "Build third_party/ORB_SLAM3 and set ORB_SLAM3_ROOT, then rebuild.");
  rclcpp::shutdown();
  return 1;
}
