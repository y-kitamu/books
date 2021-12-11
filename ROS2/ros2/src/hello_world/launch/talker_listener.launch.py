"""talker_listener.launch.py

Author : Yusuke Kitamura
Create Date : 2021-10-14 21:20:01
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""
from launch import LaunchDescription
import launch_ros.actions


def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(package="hello_world", executable="talker", output="screen"),
        launch_ros.actions.Node(package="hello_world", executable="listener", output="screen"),
    ])
