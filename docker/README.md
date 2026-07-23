# Perception container (FoundationPose)

The FoundationPose perception pipeline runs inside a container on the
perception machine, not on the host.

## Image

`isaac_ros_foundationpose_custom:v1` — a customised NVIDIA Isaac ROS image
with FoundationPose. Built from NVIDIA's Isaac ROS dev image; the current
image was produced with `docker commit` rather than a Dockerfile, so it is
not yet reproducible from source. Writing a Dockerfile is a TODO.

## Run

On the perception machine:

    docker run -it \
      --name isaac_ros_dev_container \
      --gpus all \
      --network host \
      --ipc host \
      --privileged \
      -v /home/<user>/workspaces/isaac_ros-dev:/workspaces/isaac_ros-dev \
      -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev \
      isaac_ros_foundationpose_custom:v1 \
      bash

Key points:
- The host workspace is mounted at `/workspaces/isaac_ros-dev` inside the
  container. All paths in the launch file are relative to this.
- `ISAAC_ROS_WS` is passed in so the launch file picks it up.
- `--network host` is required so ROS 2 topics reach the other machine.

## Launching the pipeline

Once inside the container:

    ros2 launch foundationpose_klt_tracking.launch.py

(The launch file lives in `launch/` in this repo — copy it into the mounted
workspace, or mount this repo so it is reachable from inside the container.)

## Required assets in the mounted workspace

These are not committed to this repo (large, downloadable, or NVIDIA-provided):

- FoundationPose models (`refine_model.onnx`, `score_model.onnx`, and the
  corresponding TensorRT `.plan` engines) under
  `isaac_ros_assets/models/foundationpose/` — download from NVIDIA NGC.
- The KLT box mesh — committed here in `assets/KLT_box/`; copy it into the
  mounted workspace so it appears at `/workspaces/isaac_ros-dev/KLT_box/`.

## TODO

- Write a Dockerfile capturing the customisations on top of NVIDIA's base
  image, so the environment is reproducible.
