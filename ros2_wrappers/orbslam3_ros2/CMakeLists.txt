cmake_minimum_required(VERSION 3.16)
project(orbslam3_ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)

set(ORB_SLAM3_ROOT "" CACHE PATH "ORB-SLAM3 source/build root")
if(DEFINED ENV{ORB_SLAM3_ROOT} AND NOT "$ENV{ORB_SLAM3_ROOT}" STREQUAL "")
  set(ORB_SLAM3_ROOT "$ENV{ORB_SLAM3_ROOT}" CACHE PATH "ORB-SLAM3 source/build root" FORCE)
elseif(NOT ORB_SLAM3_ROOT)
  set(_local_orb_root "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ORB_SLAM3")
  if(EXISTS "${_local_orb_root}/include/System.h" AND EXISTS "${_local_orb_root}/lib/libORB_SLAM3.so")
    set(ORB_SLAM3_ROOT "${_local_orb_root}" CACHE PATH "ORB-SLAM3 source/build root" FORCE)
  endif()
endif()
set(ORB_SLAM3_INCLUDE "${ORB_SLAM3_ROOT}/include")
set(ORB_SLAM3_LIBRARY "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so")

if(ORB_SLAM3_ROOT AND EXISTS "${ORB_SLAM3_INCLUDE}/System.h" AND EXISTS "${ORB_SLAM3_LIBRARY}")
  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  if(NOT DEFINED Pangolin_DIR AND NOT DEFINED ENV{Pangolin_DIR})
    set(_local_pangolin_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/Pangolin/build/src")
    if(EXISTS "${_local_pangolin_dir}/PangolinConfig.cmake")
      set(Pangolin_DIR "${_local_pangolin_dir}" CACHE PATH "Pangolin build directory" FORCE)
    endif()
  endif()
  find_package(Pangolin REQUIRED CONFIG)
  file(READ "${ORB_SLAM3_INCLUDE}/System.h" ORB_SLAM3_SYSTEM_H)

  add_executable(orbslam3_rgbd_imu src/orbslam3_rgbd_imu_node.cpp)
  target_include_directories(orbslam3_rgbd_imu PRIVATE
    "${ORB_SLAM3_INCLUDE}"
    "${ORB_SLAM3_ROOT}"
    "${ORB_SLAM3_ROOT}/include/CameraModels"
    "${ORB_SLAM3_ROOT}/Thirdparty/Sophus"
    "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2"
    "${ORB_SLAM3_ROOT}/Thirdparty/g2o"
    ${Pangolin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(orbslam3_rgbd_imu "${ORB_SLAM3_LIBRARY}" ${OpenCV_LIBS} pangolin)
  target_compile_definitions(orbslam3_rgbd_imu PRIVATE ORB_SLAM3_ENABLED=1)
  if(ORB_SLAM3_SYSTEM_H MATCHES "IMU_RGBD")
    target_compile_definitions(orbslam3_rgbd_imu PRIVATE ORB_SLAM3_HAS_IMU_RGBD=1)
  endif()
else()
  add_executable(orbslam3_rgbd_imu src/orbslam3_stub_node.cpp)
endif()

ament_target_dependencies(orbslam3_rgbd_imu
  rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs tf2_ros message_filters cv_bridge
)

install(TARGETS orbslam3_rgbd_imu DESTINATION lib/${PROJECT_NAME})

ament_package()
