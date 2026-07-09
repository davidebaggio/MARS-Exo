#include <rclcpp/rclcpp.hpp>

class Orbslam3StubNode final : public rclcpp::Node {
public:
  Orbslam3StubNode() : Node("orbslam3_exo") {
    declare_parameter<std::string>("vocabulary_path", "");
    declare_parameter<std::string>("settings_path", "");
    declare_parameter<std::string>("mode", "rgbd_imu");
    RCLCPP_FATAL(
      get_logger(),
      "ORB-SLAM3 backend not built. Build ORB-SLAM3 from source, export ORB_SLAM3_ROOT, then rebuild this workspace.");
  }
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Orbslam3StubNode>();
  rclcpp::shutdown();
  return 2;
}
