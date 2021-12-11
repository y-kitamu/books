/*
 * set_and_get_parameters.cpp
 *
 * Create Date : 2021-10-12 19:01:59
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <vector>

using namespace std::chrono_literals;


int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("set_and_get_parameters");

    node->declare_parameter("foo");
    node->declare_parameter("bar");
    node->declare_parameter("baz");

    auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(node);
    while (!parameters_client->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(node->get_logger(), "Interrupted");
        }
        RCLCPP_INFO(node->get_logger(), "Waiting");
    }

    std::stringstream ss;
    for (auto& parameter : parameters_client->get_parameters({"foo", "bar", "baz"})) {
        ss << "\nParameter name: " << parameter.get_name();
        ss << "\nParameter value (" << parameter.get_type_name() << "): " << parameter.value_to_string();
    }
    RCLCPP_INFO(node->get_logger(), ss.str().c_str());

    rclcpp::shutdown();
    return 0;
}
