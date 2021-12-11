cmake_minimum_required(VERSION 3.20)

project(BinaryAnalysis)

# set(CMAKE_C_COMPILER /usr/bin/clang)
# set(CMAKE_CXX_COMPILER /usr/bin/clang++)
include(cmake/utility.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS "-fno-stack-protector")

ADD_EXES(${CMAKE_CURRENT_SOURCE_DIR}/src)
