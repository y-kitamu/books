cmake_minimum_required(VERSION 3.20)

# set(CMAKE_C_COMPILER /usr/bin/clang)
# set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(CUDA_Tutorial VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "-arch=compute_52")

file(GLOB file_list "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cu")
foreach(fname IN LISTS file_list)
  get_filename_component(bname ${fname} NAME_WE)
  add_executable(${bname} ${fname})
endforeach()
