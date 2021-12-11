/*
 * talker_with_service_param.cpp
 *
 * Create Date : 2021-10-12 19:24:37
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <chrono>
#include <cstdio>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>

#include "hello_world_msgs/srv/set_message.hpp"

using namespace std::chrono_literals;
using hello_world_msgs::srv::SetMessage;


class Talker : public rclcpp::Node {
  public:
    explicit Talker(const std::string& topic_name) : Node("talker"), data_("Hello world!") {
        auto publish_message = [this]() -> void {
            auto msg = std::make_unique<std_msgs::msg::String>();
            msg->data = data_;

            std::string decorated_data = decoration_ + msg->data + decoration_;
            RCLCPP_INFO(this->get_logger(), "%s", decorated_data.c_str());
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
            this->data_ = request->message;
            response->result = true;
        };

        srv_ = create_service<SetMessage>("set_message", handle_set_message);

        decoration_ = declare_parameter("decoration", "");
        auto parameter_callback = [this](const std::vector<rclcpp::Parameter> params)
            -> rcl_interfaces::msg::SetParametersResult {
            auto result = rcl_interfaces::msg::SetParametersResult();
            result.successful = false;
            for (auto param : params) {
                if (param.get_name() == "decoration") {
                    decoration_ = param.as_string();
                    result.successful = true;
                }
            }
            if (!result.successful) {
                RCLCPP_INFO(this->get_logger(), "Failed to update param");
            }
            return result;
        };
        param_cb_res_ = add_on_set_parameters_callback(parameter_callback);
    }

  private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<SetMessage>::SharedPtr srv_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_res_;
    std::string data_;
    std::string decoration_;
};


int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);

    auto node = std::make_shared<Talker>("chatter");
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
