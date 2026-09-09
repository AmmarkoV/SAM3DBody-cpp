#!/bin/bash
# run_it.sh — launch the node.  Any extra args are forwarded to roslaunch, e.g.:
#   ./run_it.sh serverHost:=192.168.1.50 serverPort:=8080 fromRGBTopic:=/xtion/rgb/image_raw
#
# Make sure the catkin workspace is built and sourced first:
#   cd <catkin_ws> && catkin_make && source devel/setup.bash

roslaunch poseEstimation3D poseEstimation3D.launch "$@"

exit 0
