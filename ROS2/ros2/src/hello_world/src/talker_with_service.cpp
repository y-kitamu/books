/*
 * talker_with_service.cpp
 *
 * Create Date : 2021-10-10 23:14:14
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "hello_world_msgs/srv/set_message.hpp"

using namespace std::chrono_literals;
using hello_world_msgs::srv::SetMessage;


class Talker : public rclcpp::Node {
  public:
    explicit Talker(const std::string& topic_name) : Node("talker"), data_("Hello world!") {
        auto publish_message = [this]() -> void {
            auto msg = std::make_unique<std_msgs::msg::String>();
            msg->data = data_;

            RCLCPP_INFO(this->get_logger(), "%s", msg->data.c_str());
            pub_->publish(std::move(msg));
        };

        rclcpp::QoS qos(rclcpp::KeepLast(10));
        pub_ = create_publisher<std_msgs::msg::String>(topic_name, qos);
        timer_ = create_wall_timer(100ms, publish_message);

        auto handle_set_message = [this](const std::shared_ptr<rmw_request_id_t> request_header,
                                         const std::shared_ptr<SetMessage::Request> request,
                                         std::shared_ptr<SetMessage::Response> response) -> void {
            (void)request_header;
            RCLCPP_INFO(this->get_logger(), "message %s -> %s", this->data_.c_str(),
                        request->message.c_str());
            std::this_thread::sleep_for(1s);
            this->data_ = request->message;
            response->result = true;
        };

        srv_ = create_service<SetMessage>("set_message", handle_set_message);
    }

  private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<SetMessage>::SharedPtr srv_;
    std::string data_;
};


int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);

    auto node = std::make_shared<Talker>("chatter");
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
