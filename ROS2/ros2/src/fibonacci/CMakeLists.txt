cmake_minimum_required(VERSION 3.8)
project(fibonacci)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(hello_world_msgs REQUIRED)

function (custom_executable target)
  add_executable(${target} src/${target}.cpp)
  ament_target_dependencies(${target} "rclcpp" "rclcpp_action" "hello_world_msgs")
  install(TARGETS ${target} DESTINATION lib/${PROJECT_NAME})
endfunction()

custom_executable(action_server)
custom_executable(action_client)

ament_package()
