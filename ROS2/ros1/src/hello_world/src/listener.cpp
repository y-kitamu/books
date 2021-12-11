/*
 * listener.cpp
 *
 * Create Date : 2021-10-03 21:46:04
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <ros/ros.h>
#include <std_msgs/String.h>

void callback(const std_msgs::String::ConstPtr& msg) {
    ROS_INFO("%s", msg->data.c_str());
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;

    ros::Subscriber chatter = n.subscribe("chatter", 1000, callback);
    ros::spin();

    return 0;
}
