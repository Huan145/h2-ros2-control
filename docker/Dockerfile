# FoundationPose perception container for the H2 pipeline.
#
# Base: NVIDIA's pre-built Isaac ROS dev image, pulled from NGC as part of the
# Isaac ROS getting-started flow:
#   https://nvidia-isaac-ros.github.io/getting_started/index.html
#
# This image adds the FoundationPose stack and a few utilities on top. Every
# other package in the running container (CUDA 13.0, TensorRT 10.13,
# the isaac_ros/gxf packages, OpenCV, rviz2) arrives as an apt dependency of
# ros-jazzy-isaac-ros-foundationpose.
#
# Build:
#   docker build -t isaac_ros_foundationpose_custom:v2 -f docker/Dockerfile .
#
# Run: see docker/README.md

FROM nvcr.io/nvidia/isaac/ros:isaac_ros_28556f8bc78a98822bd08b2d7c6fcf9b-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-isaac-ros-foundationpose \
        ros-jazzy-domain-bridge \
        ros-jazzy-image-transport-plugins \
        ros-jazzy-compressed-image-transport \
        vim \
        nload \
    && rm -rf /var/lib/apt/lists/*
