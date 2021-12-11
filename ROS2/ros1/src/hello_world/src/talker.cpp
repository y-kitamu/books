/*
 * talker.cpp
 *
 * Create Date : 2021-10-03 18:08:47
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <hello_world/SetMessage.h>

std_msgs::String msg;

bool SetMessage(hello_world::SetMessage::Request &req,
                hello_world::SetMessage::Response &res) {
    ROS_INFO("message %s -> %s", msg.data.c_str(), req.message.c_str());
    msg.data = req.message;
    res.result = true;
    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    ros::ServiceServer service = n.advertiseService("set_message", SetMessage);
    ros::Publisher chatter = n.advertise<std_msgs::String>("chatter", 1000);
    ros::Rate loop_rate(10);

    msg.data = "Hello world!";
    while (ros::ok()) {
        std::string decoration = "";
        n.param<std::string>("decoration", decoration, "");
        std::string decorated_data = decoration + msg.data + decoration;
        // ROS_INFO("%s", msg.data.c_str());
        ROS_INFO("%s", decorated_data.c_str());

        chatter.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
