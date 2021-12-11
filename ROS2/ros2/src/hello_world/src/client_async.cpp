/*
 * client_async.cpp
 *
 * Create Date : 2021-10-12 07:01:43
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <chrono>
#include <cinttypes>
#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "hello_world_msgs/srv/set_message.hpp"

using namespace std::chrono_literals;
using hello_world_msgs::srv::SetMessage;

class ClientNode : public rclcpp::Node {
  public:
    explicit ClientNode(const std::string& service_name) : Node("client_async") {
        client_ = create_client<SetMessage>(service_name);

        while (!client_->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Service not available.");
        }
        auto request = std::make_shared<SetMessage::Request>();
        request->message = "Hello service!";

        {
            using ServiceResponseFuture = rclcpp::Client<SetMessage>::SharedFuture;
            auto response_recieved_callback = [this](ServiceResponseFuture future) {
                auto response = future.get();
                RCLCPP_INFO(this->get_logger(), "%s", response->result ? "true" : "false");
                rclcpp::shutdown();
            };
            auto future_result = client_->async_send_request(request, response_recieved_callback);
        }
    }

  private:
    rclcpp::Client<SetMessage>::SharedPtr client_;
};

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ClientNode>("set_message");
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
