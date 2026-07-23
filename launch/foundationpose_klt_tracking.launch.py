#!/usr/bin/env python3
"""
FoundationPose + Tracking launch file for KLT box in Isaac Sim.

RUNS INSIDE the Isaac ROS container (isaac_ros_foundationpose_custom) on the
perception machine — not on the host. All paths here are container-internal;
ISAAC_ROS_WS defaults to /workspaces/isaac_ros-dev (the mounted workspace).
See docker/README.md for the image and run command.

Pipeline:
  Isaac Sim → /rgb, /depth, /camera_info    ]
  semantic_mask_node.py → /segmentation_mask ]
        ↓
  Selector Node  (decides: full estimation or tracking?)
        ├──► FoundationPose Node  (full 6-DoF estimation, runs every reset_period ms)
        └──► Tracking Node        (fast frame-to-frame tracking, runs every frame)
        ↓
  /output  (Detection3DArray, stable ~20Hz)

Topics expected:
  /rgb                 - sensor_msgs/Image (rgb8,  1280x720)
  /depth               - sensor_msgs/Image (32FC1, 1280x720)
  /camera_info         - sensor_msgs/CameraInfo
  /segmentation_mask   - sensor_msgs/Image (mono8, 1280x720)
"""

import os
import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# ── Paths ────────────────────────────────────────────────────────────────────
ISAAC_ROS_WS          = os.environ.get('ISAAC_ROS_WS', '/workspaces/isaac_ros-dev')
FOUNDATIONPOSE_MODELS = os.path.join(ISAAC_ROS_WS, 'isaac_ros_assets', 'models', 'foundationpose')
REFINE_ENGINE_PATH    = os.path.join(FOUNDATIONPOSE_MODELS, 'refine_trt_engine.plan')
SCORE_ENGINE_PATH     = os.path.join(FOUNDATIONPOSE_MODELS, 'score_trt_engine.plan')
MESH_FILE_PATH        = os.path.join(ISAAC_ROS_WS, 'KLT_box', 'KLT_box_metres.obj')
TEXTURE_PATH          = ''


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            'mesh_file_path',
            default_value=MESH_FILE_PATH,
            description='Absolute path to the .obj mesh file'),
        DeclareLaunchArgument(
            'texture_path',
            default_value=TEXTURE_PATH,
            description='Absolute path to texture file (leave empty if none)'),
        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value=REFINE_ENGINE_PATH,
            description='Absolute path to refine TensorRT engine'),
        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value=SCORE_ENGINE_PATH,
            description='Absolute path to score TensorRT engine'),
        DeclareLaunchArgument(
            'reset_period',
            default_value='5000',
            description='How often (ms) to run full estimation instead of tracking. '
                        'Lower = more accurate but more compute. Recommended: 3000-10000'),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='False',
            description='Whether to launch RViz2 (disable inside container)'),
    ]

    mesh_file_path     = LaunchConfiguration('mesh_file_path')
    texture_path       = LaunchConfiguration('texture_path')
    refine_engine_path = LaunchConfiguration('refine_engine_file_path')
    score_engine_path  = LaunchConfiguration('score_engine_file_path')
    reset_period       = LaunchConfiguration('reset_period')
    launch_rviz        = LaunchConfiguration('launch_rviz')

    # ── Selector Node ────────────────────────────────────────────────────────
    # Sits between the inputs and the two FoundationPose nodes.
    # Routes each incoming frame to either:
    #   - FoundationPose Node (full estimation) every reset_period ms
    #   - Tracking Node (fast) every other frame
    # Also receives the output pose to know the current tracked state.
    selector_node = ComposableNode(
        name='selector_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Selector',
        parameters=[{
            # How often to trigger full re-estimation (milliseconds)
            # 5000ms = re-estimate every 5 seconds, track in between
            'reset_period': reset_period,
        }],
        remappings=[
            # Inputs from Isaac Sim / mask node
            ('depth_image',   '/depth'),
            ('image',         '/rgb'),
            ('camera_info',   '/camera_info'),
            ('segmentation',  '/segmentation_mask'),
            # Selector outputs go to FoundationPose and Tracking nodes
            # via their default topic names (no remapping needed)
        ]
    )

    # ── FoundationPose Node (full estimation) ────────────────────────────────
    # Runs infrequently (every reset_period ms).
    # Takes segmentation mask + depth + RGB → outputs initial/corrected pose.
    # Score model ranks N candidate poses and picks the best one.
    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path':   texture_path,

            'refine_engine_file_path':     refine_engine_path,
            'refine_input_tensor_names':   ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names':  ['input1', 'input2'],
            'refine_output_tensor_names':  ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],

            'score_engine_file_path':      score_engine_path,
            'score_input_tensor_names':    ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names':   ['input1', 'input2'],
            'score_output_tensor_names':   ['output_tensor'],
            'score_output_binding_names':  ['output1'],
        }],
        # Note: when used with Selector, the Selector handles input remapping.
        # FoundationPose node reads from Selector's output topics directly.
    )

    # ── Tracking Node (fast frame-to-frame) ──────────────────────────────────
    # Runs every frame (~20Hz).
    # Uses the previous frame's pose as starting point → smooth, stable output.
    # Only uses the refine model (no score model needed for tracking).
    foundationpose_tracking_node = ComposableNode(
        name='foundationpose_tracking_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path':   texture_path,

            # Tracking only uses the refine model, not the score model
            'refine_engine_file_path':     refine_engine_path,
            'refine_input_tensor_names':   ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names':  ['input1', 'input2'],
            'refine_output_tensor_names':  ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
        }],
    )

    # ── Container ────────────────────────────────────────────────────────────
    # All 3 nodes share the same process and GPU memory (zero-copy via NITROS)
    foundationpose_container = ComposableNodeContainer(
        name='foundationpose_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            selector_node,
            foundationpose_node,
            foundationpose_tracking_node,
        ],
        output='screen'
    )

    # ── RViz2 (disabled by default — run on host instead) ────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', '/opt/ros/jazzy/share/isaac_ros_foundationpose/rviz/foundationpose_isaac_sim.rviz'],
        condition=IfCondition(launch_rviz)
    )

    return launch.LaunchDescription(launch_args + [
        foundationpose_container,
        rviz_node,
    ])

