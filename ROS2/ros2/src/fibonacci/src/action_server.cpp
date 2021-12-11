/*
 * action_server.cpp
 *
 * Create Date : 2021-10-17 10:30:58
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <inttypes.h>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "hello_world_msgs/action/fibonacci.hpp"


using Fibonacci = hello_world_msgs::action::Fibonacci;
using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;


class MinimalActionServer : public rclcpp::Node {
  public:
    explicit MinimalActionServer() : Node("minimal_action_server") {
        using namespace std::placeholders;

        this->action_server_ = rclcpp_action::create_server<Fibonacci>(
            this->get_node_base_interface(), this->get_node_clock_interface(),
            this->get_node_logging_interface(), this->get_node_waitables_interface(), "fibonacci",
            std::bind(&MinimalActionServer::handle_goal, this, _1, _2),
            std::bind(&MinimalActionServer::handle_cancel, this, _1),
            std::bind(&MinimalActionServer::handle_accepted, this, _1));
    }

  private:
    rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
                                            std::shared_ptr<const Fibonacci::Goal> goal) {
        RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
        (void)uuid;
        if (goal->order > 9000) {
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleFibonacci> goal_handle) {
        RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void execute(const std::shared_ptr<GoalHandleFibonacci> goal_handle) {
        RCLCPP_INFO(this->get_logger(), "Executing goal");
        rclcpp::Rate loop_rate(1);
        const auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<Fibonacci::Feedback>();
        auto& sequence = feedback->sequence;
        sequence.push_back(0);
        sequence.push_back(1);
        auto result = std::make_shared<Fibonacci::Result>();

        for (int i = 1; (i < goal->order) && rclcpp::ok(); ++i) {
            if (goal_handle->is_canceling()) {
                result->sequence = sequence;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal Canceled");
            }

            sequence.push_back(sequence[i] + sequence[i - 1]);
            goal_handle->publish_feedback(feedback);
            RCLCPP_INFO(this->get_logger(), "Publish Feedback");

            loop_rate.sleep();
        }

        if (rclcpp::ok()) {
            result->sequence = sequence;
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal Succeeded");
        }
    }

    void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle) {
        using namespace std::placeholders;

        std::thread{std::bind(&MinimalActionServer::execute, this, _1), goal_handle}.detach();
    }
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto action_server = std::make_shared<MinimalActionServer>();

    rclcpp::spin(action_server);
    rclcpp::shutdown();
    return 0;
}
