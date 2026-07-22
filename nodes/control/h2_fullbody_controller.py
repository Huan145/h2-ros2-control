#!/usr/bin/env python3
"""
h2_fullbody_controller.py

H2 humanoid locomotion policy controller, adapted from NVIDIA's reference
h1_fullbody_controller.py (IsaacSim-ros_workspaces repo) to H2's specific
joint configuration, confirmed against env.yaml from the h2_flat training run.

Key architectural differences from the earlier h2_rl_policy_node.py:
  - Message-driven control (TimeSynchronizer on joint_states + imu), not a
    wall-clock ROS2 timer. Control cadence is tied to the sim's own physics
    step rate via a decimation counter, matching Isaac Lab's decimation=4
    exactly, regardless of wall-clock/rendering load.
  - use_sim_time=True -- all timing uses Isaac Sim's /clock, not wall time.
  - Uses an IMU sensor on the pelvis for angular velocity + orientation
    (gravity projection), and integrates linear acceleration for linear
    velocity -- NOT an odometry node. Requires an IMU sensor + ROS2 Publish
    IMU ActionGraph on the pelvis link (see setup notes at bottom of file).

Confirmed against env.yaml (h2_flat, 2026-05-25_09-21-25 run):
  - sim.dt = 0.005, decimation = 4  -> policy runs at 50Hz, physics at 200Hz
  - actions.joint_pos: scale=0.5, offset=0.0, use_default_offset=True, joints=".*"
  - observations.policy: base_lin_vel, base_ang_vel, projected_gravity,
    # velocity_commands, joint_pos_rel, joint_vel_rel, last_action
    -- ALL have scale=null, clip=null (no hidden scaling anywhere)
  - init_state.joint_pos confirms defaults for ALL non-zero joints, including
    arm joints that were WRONG (left at 0.0) in the previous node version:
      hip_pitch: -0.2, knee: 0.42, ankle_pitch: -0.23,
      shoulder_pitch: 0.28, shoulder_roll: 0.16, elbow: 0.52
  - usd_path confirmed: H2.usd (not H2_box.usd)

NOT yet applied anywhere (env.yaml confirms armature=0.01 on every actuator
group; this is a separate PhysX joint attribute, not part of UsdPhysics
DriveAPI, and still needs to be set via Script Editor if not already).

Run:
  conda activate <ros2 + torch env>
  source /opt/ros/jazzy/setup.bash
  python3 h2_fullbody_controller.py
"""

import io
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from message_filters import Subscriber, TimeSynchronizer


# ============================================================================
# CONFIG
# ============================================================================

# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-05-25_09-21-25/exported/policy.pt"
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-09_15-21-41/exported/policy.pt"
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-13_14-50-31/exported/policy.pt"
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-14_11-25-11/exported/policy.pt"
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-14_13-16-16/exported/policy.pt" #model_3000.pt - stable
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-14_16-08-43/exported/policy.pt"
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-15_11-33-47/exported/policy.pt"
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-15_15-36-01/exported/policy.pt"
# POLICY_PATH = "/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-15_16-07-25/exported/policy.pt" #model_4000.pt - stable 
# POLICY_PATH="/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-16_09-22-21/exported/policy.pt"
# POLICY_PATH="/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-16_11-57-56/exported/policy.pt"
# POLICY_PATH="/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-16_14-18-01/exported/policy.pt"
POLICY_PATH="/home/apt-ipc/IsaacLab/logs/rsl_rl/h2_flat/2026-07-16_16-56-57/exported/policy.pt" #best stable model

ACTION_SCALE = 0.5   # confirmed: actions.joint_pos.scale in env.yaml
DECIMATION = 4       # confirmed: env.yaml decimation (policy runs every 4th tick)
NUM_JOINTS = 31

# Joint order -- confirmed via robot.data.joint_names AND action_term._joint_names
# (both identical, per earlier verification).
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "head_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# Default joint positions -- confirmed directly from env.yaml's
# scene.robot.init_state.joint_pos. NOTE: shoulder_pitch/shoulder_roll/elbow
# are NON-ZERO -- this was wrong (left at 0.0) in the previous node version.
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2, "waist_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0, "waist_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_knee_joint": 0.42, "right_knee_joint": 0.42, "head_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.28, "right_shoulder_pitch_joint": 0.28,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0, "head_yaw_joint": 0.0,
    "left_shoulder_roll_joint": 0.16, "right_shoulder_roll_joint": -0.16,
    "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.52, "right_elbow_joint": 0.52,
    "left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}

OBS_DIM = 3 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS  # 105


# ============================================================================
# Node
# ============================================================================

class H2FullbodyController(Node):
    """Fullbody controller for the H2 humanoid, adapted from NVIDIA's H1 example."""

    def __init__(self):
        super().__init__('h2_fullbody_controller')

        self.declare_parameter('policy_path', POLICY_PATH)
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)]
        )

        self._logger = self.get_logger()

        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        self._cmd_vel_subscription = self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_callback, qos_profile=10)

        self._joint_publisher = self.create_publisher(
            JointState, 'joint_command', qos_profile=sim_qos_profile)

        self._imu_sub_filter = Subscriber(self, Imu, 'imu', qos_profile=sim_qos_profile)
        self._joint_states_sub_filter = Subscriber(self, JointState, 'joint_states', qos_profile=sim_qos_profile)

        queue_size = 10
        subscribers = [self._joint_states_sub_filter, self._imu_sub_filter]
        self.sync = TimeSynchronizer(subscribers, queue_size)
        self.sync.registerCallback(self._tick)

        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._action_scale = ACTION_SCALE
        self._previous_action = np.zeros(NUM_JOINTS)
        self._policy_counter = 0
        self._decimation = DECIMATION
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._lin_vel_b = np.zeros(3)
        self._dt = 0.0
        self.action = np.zeros(NUM_JOINTS)

        self.joint_names = JOINT_NAMES
        self.default_pos = np.array([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], dtype=np.float64)

        self._logger.info(
            f"Initializing H2FullbodyController: obs_dim={OBS_DIM}, "
            f"action_dim={NUM_JOINTS}, decimation={self._decimation}"
        )

    def _cmd_vel_callback(self, msg):
        self._cmd_vel = msg

    def _tick(self, joint_state: JointState, imu: Imu):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.')

        self._dt = (now - self._last_tick_time)
        self._last_tick_time = now

        self.forward(joint_state, imu)

        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names

        action_pos = self.default_pos + self.action * self._action_scale
        self._joint_command.position = action_pos.tolist()
        self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        R_BI = self.quat_to_rot_matrix(quat_array).T

        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ])
        # Naive integration -- drifts over long runs, matches the reference
        # H1 implementation. Consider replacing with an odometry-based twist
        # if drift becomes a problem over longer test durations.
        self._lin_vel_b = lin_acc_b * self._dt + self._lin_vel_b

        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
        ])

        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(OBS_DIM)
        obs[0:3] = self._lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b

        cmd_vel = [self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z]
        obs[9:12] = np.array(cmd_vel)

        current_joint_pos = np.zeros(NUM_JOINTS)
        current_joint_vel = np.zeros(NUM_JOINTS)
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        obs[12:12 + NUM_JOINTS] = current_joint_pos - self.default_pos
        obs[12 + NUM_JOINTS:12 + 2 * NUM_JOINTS] = current_joint_vel
        obs[12 + 2 * NUM_JOINTS:12 + 3 * NUM_JOINTS] = self._previous_action

        return obs

    def _compute_action(self, obs):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_t).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        obs = self._compute_observation(joint_state, imu)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def load_policy(self):
        with open(self.policy_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.policy = torch.jit.load(buffer)

    def _get_stamp_prefix(self) -> str:
        import time
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'


def main(args=None):
    rclpy.init(args=args)
    node = H2FullbodyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# ============================================================================
# SETUP NOTES -- what changes vs. the odometry-based setup from before:
#
# 1. ADD an IMU sensor: right-click /World/H2/pelvis (or wherever your
#    pelvis prim lives) -> Create > Isaac > Sensors > Imu Sensor.
#
# 2. ADD a new ActionGraph ("ROS_Imu"), pipelineStage = pipelineStageOnDemand:
#    On Physics Step -> Isaac Read IMU Node (IMU Prim = the sensor above,
#    uncheck "Read Gravity" -- confirmed done) -> ROS2 Publish IMU (topic: "imu")
#    Also wire Isaac Read Simulation Time -> ROS2 Publish IMU, with
#    "Reset on Stop" checked on the Read Simulation Time node.
#
# 3. Your EXISTING /joint_states publisher and /joint_command -> Articulation
#    Controller graph stay as they are -- this node still uses those same
#    topic names.
#
# 4. You can remove/disable the Isaac Compute Odometry Node + ROS2 Publish
#    Odometry graph from before -- this node no longer uses /odom at all.
#
# 5. CRITICAL, per NVIDIA's own docs: start this node BEFORE hitting Play,
#    not after. e.g.:
#      conda activate h2_rl && source /opt/ros/jazzy/setup.bash
#      python3 h2_fullbody_controller.py
#    ...THEN press Play in Isaac Sim. This node will just sit idle waiting
#    for the first synchronized joint_states/imu message pair, so there's
#    no harm in it running before physics starts -- but starting Play first
#    (our old order) left the robot with no active balancing policy for the
#    first several seconds, which is very likely why the settle-phase saga
#    was ever necessary in the first place.
# ============================================================================