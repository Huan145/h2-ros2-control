# h2-ros2-control

ROS 2 stack for controlling a Unitree H2 humanoid in Isaac Sim: RL locomotion,
arm IK, and FoundationPose-based 6-DoF box detection for grasping.

Trained policies come from the companion training repo
(`UnitreeH2-isaaclab-training`); this repo consumes the exported policy.

## Architecture

Perception chain (KLT box pose):

    Isaac Sim  ->  /rgb, /depth, /camera_info, /semantic_segmentation
                       |
    semantic_mask_node ->  /segmentation_mask   (binary mask of the box)
                       |
    FoundationPose (container) ->  /tracking/output   (Detection3DArray, ~20Hz)
                       |
    pose_overlay_visualizer_tracking_grasp
                       |->  /tracking_visualization  (RGB + wireframe overlay)
                       |->  /target_pose_left        (grasp pose, pelvis frame)
                       |->  /target_pose_right

Control chain:

    /target_pose_left|right  ->  H2_ik_node  ->  /arm_joints  ->  Isaac Sim
    H2_lift_node             ->  hardcoded box-lift sequence
    h2_fullbody_controller   ->  RL locomotion policy (joint_states + IMU)

## Machines

| Machine | ROS_DOMAIN_ID | Runs |
|---|---|---|
| Sim / control | 99 | Isaac Sim, control nodes, semantic mask, visualizer |
| Perception | 0 | FoundationPose container |

The domains are isolated by design; topics are forwarded with the standard
`domain_bridge` package.

    sudo apt-get install -y ros-jazzy-domain-bridge
    ros2 run domain_bridge domain_bridge config/bridge_config.yaml

Run it on both machines. `config/bridge_config.yaml` forwards camera topics
(/rgb, /depth, /camera_info, /segmentation_mask) from domain 99 to domain 0, and
results (/tracking/output, /tracking_visualization) back from domain 0 to 99.


## Nodes and environments

| Node | Env | Purpose |
|---|---|---|
| `nodes/control/h2_fullbody_controller.py` | `h2_rl` | RL locomotion policy |
| `nodes/control/H2_ik_node.py` | `ik_env` | arm IK to grasp poses |
| `nodes/control/H2_lift_node.py` | `ik_env` | hardcoded box-lift sequence |
| `nodes/perception/semantic_mask_node.py` | `yolo-env` | Isaac Sim semantics -> binary mask |
| `nodes/perception/pose_overlay_visualizer_tracking_grasp.py` | `yolo-env` | overlay + grasp poses |
| `launch/foundationpose_klt_tracking.launch.py` | container | FoundationPose pipeline |

Create the environments:

    conda env create -f envs/h2_rl.yml
    conda env create -f envs/ik_env.yml

Every terminal needs the env activated, ROS 2 sourced, and the domain set:

    conda activate <env>
    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=<domain>      # sim/control machine: 99, perception machine: 0

## Running

Nodes are currently run individually for testing.

    # RL locomotion (h2_rl env)
    python3 nodes/control/h2_fullbody_controller.py [--policy <file>]

    # Arm IK, consumes grasp poses (ik_env)
    python3 nodes/control/H2_ik_node.py

    # Box lift sequence (ik_env)
    python3 nodes/control/H2_lift_node.py

    # Semantic mask; target label set by TARGET_LABEL in the file (ik_env)
    python3 nodes/perception/semantic_mask_node.py

FoundationPose pipeline: see `docker/README.md` (runs inside the container).

## Assets and models

- `assets/h2_description/` - H2 URDF + STL meshes (used by the IK nodes)
- `assets/KLT_box/` - KLT box mesh for FoundationPose; copy into the container's
  mounted workspace so it lands at `/workspaces/isaac_ros-dev/KLT_box/`
- `assets/scenes/` - Isaac Sim scene USDs (flattened, self-contained)
- `models/` - trained policies, not committed; see `models/README.md`
- FoundationPose weights are not committed; download from NVIDIA NGC into the
  container workspace (see `docker/README.md`)

## Notes

- USD scenes are flattened snapshots; re-flatten and re-commit if you modify one.
- Binary assets (STL, USD, PNG) are tracked with Git LFS.
