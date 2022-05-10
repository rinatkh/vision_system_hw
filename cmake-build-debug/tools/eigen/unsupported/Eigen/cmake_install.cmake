# Install script for directory: /home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/AdolcForward"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/AlignedVector3"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/ArpackSupport"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/AutoDiff"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/BVH"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/EulerAngles"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/FFT"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/IterativeSolvers"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/KroneckerProduct"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/MatrixFunctions"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/MoreVectorization"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/MPRealSupport"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/NNLS"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/NonLinearOptimization"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/NumericalDiff"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/OpenGLSupport"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/Polynomials"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/Skyline"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/SparseExtra"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/SpecialFunctions"
    "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/home/rinat/CLionProjects/vision_system_hw/tools/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/rinat/CLionProjects/vision_system_hw/cmake-build-debug/tools/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

