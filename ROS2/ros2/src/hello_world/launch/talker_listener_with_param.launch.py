"""talker_listener_with_param.launc.py

Author : Yusuke Kitamura
Create Date : 2021-10-14 21:47:57
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""
from launch import LaunchDescription
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    decoration = launch.substitutions.LaunchConfiguration("decoration")

    return LaunchDescription([
        launch.actions.DeclareLaunchArgument("decoration",
                                             default_value="",
                                             description="Message decoration string"),
        launch_ros.actions.Node(package="hello_world",
                                executable="talker_with_service_param",
                                name="talker",
                                output="screen",
                                parameters=[{
                                    'decoration': decoration
                                }]),
        launch_ros.actions.Node(package="hello_world", executable="listener", output="screen"),
    ])
