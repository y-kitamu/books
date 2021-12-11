"""talker_listener_with_param_file.launch.py

Author : Yusuke Kitamura
Create Date : 2021-10-14 22:01:25
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""
from launch import LaunchDescription
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    params_file = launch.substitutions.LaunchConfiguration(
        "params",
        default=[launch.substitutions.ThisLaunchFileDir(), "/params.yaml"],
    )

    return LaunchDescription([
        launch_ros.actions.Node(package='hello_world',
                                executable="talker_with_service_param",
                                name="talker",
                                output="screen",
                                parameters=[params_file]),
        launch_ros.actions.Node(package="hello_world", executable="listener", output="screen")
    ])
