cmake_minimum_required(VERSION 2.8.3)
project(poseEstimation3D)

## Standard-message-only ROS package: no custom .msg / message_generation, to
## keep the build minimal and robust.  Heavy pose estimation is remote.
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
)
find_package(OpenCV REQUIRED)

catkin_package()

## AmmClient (minimal HTTP client) is vendored into dependencies/AmmClient/ by
## initialize.sh (real file copies, never symlinks) and compiled straight into
## the node — no prebuilt .so, no LD_LIBRARY_PATH, fully self-contained/portable.
set(AMMCLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/AmmClient")
if(NOT EXISTS "${AMMCLIENT_DIR}/AmmClient.c")
  message(FATAL_ERROR
    "AmmClient sources not found at ${AMMCLIENT_DIR}\n"
    "  Run ./initialize.sh in this package first to vendor AmmClient.")
endif()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${AMMCLIENT_DIR}
)

add_executable(poseEstimation3D
  src/main.cpp
  ${AMMCLIENT_DIR}/AmmClient.c
  ${AMMCLIENT_DIR}/network.c
  ${AMMCLIENT_DIR}/protocol.c
  ${AMMCLIENT_DIR}/tools.c
)
target_link_libraries(poseEstimation3D
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  pthread
  rt
)

install(TARGETS poseEstimation3D RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
