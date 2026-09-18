// A narrow composition of upstream ros1_bridge factories, exclusively ROS 1 -> 2.
#include <iostream>
#include <string>
#include <vector>
#include "ros/ros.h"
#include "rclcpp/rclcpp.hpp"
#include "ros1_bridge/bridge.hpp"

int main(int argc, char **argv) {
  // Each generated entry is: ROS 1 type, absolute topic, volatile|retained.
  // No ROS 1 publisher or service client/server is created here.
  if (argc < 4 || (argc - 1) % 3 != 0) {
    std::cerr << "expected type/topic/durability triples\n";
    return 2;
  }
  ros::init(argc, argv, "a1_observation_bridge", ros::init_options::NoSigintHandler | ros::init_options::NoRosout);
  rclcpp::init(argc, argv);
  ros::NodeHandle ros1;
  auto ros2 = std::make_shared<rclcpp::Node>("a1_observation_bridge",
    rclcpp::NodeOptions().start_parameter_services(false).enable_rosout(false));
  std::vector<ros1_bridge::Bridge1to2Handles> bridges;
  try {
    for (int i = 1; i < argc; i += 3) {
      std::string type1(argv[i]), topic(argv[i + 1]), durability(argv[i + 2]), type2;
      if (topic.empty() || topic.front() != '/' ||
          (durability != "volatile" && durability != "retained") ||
          !ros1_bridge::get_1to2_mapping(type1, type2)) {
        throw std::runtime_error("invalid or unmapped observation topic: " + topic);
      }
      auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
      if (durability == "retained") qos.transient_local();
      bridges.push_back(ros1_bridge::create_bridge_from_1_to_2(
        ros1, ros2, type1, topic, 10, type2, topic, qos));
      std::cout << topic << " " << type1 << " -> " << type2 << std::endl;
    }
    ros::AsyncSpinner spinner(1);
    spinner.start();
    rclcpp::spin(ros2);
    spinner.stop();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    ros::shutdown();
    rclcpp::shutdown();
    return 1;
  }
  ros::shutdown();
  rclcpp::shutdown();
  return 0;
}
