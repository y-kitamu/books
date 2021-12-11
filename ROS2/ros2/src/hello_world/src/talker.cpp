/*
 * talker.cpp
 *
 * Create Date : 2021-10-09 19:02:07
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>


using namespace std::chrono_literals;


class Talker: public rclcpp::Node {
  public:
    explicit Talker(const std::string& topic_name) : Node("Talker") {
        auto publish_message = [this] () -> void {
            auto msg = std::make_unique<std_msgs::msg::String>();
            msg->data = "Hellow World!";
            RCLCPP_INFO(this->get_logger(), "%s", msg->data.c_str());
            pub_->publish(std::move(msg));
        };

        rclcpp::QoS qos(rclcpp::KeepLast(10));
        pub_ = create_publisher<std_msgs::msg::String>(topic_name, qos);
        timer_ = create_wall_timer(100ms, publish_message);
    }

  private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};


int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);

    auto node = std::make_shared<Talker>("chatter");
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
