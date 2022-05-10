cmake_minimum_required(VERSION 3.17)
project(HW_VISION_TASKS)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED )

set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
add_subdirectory(tools/eigen)
add_subdirectory(tools/gtest)
enable_testing()

add_subdirectory(1_task)
add_subdirectory(2_task)
add_subdirectory(3_task)
add_subdirectory(4_task)
