#!/usr/bin/env python3
"""
H2 Box Lift Node
-----------------
Two-stage sequence with two DIFFERENT kinds of targets:

  STAGE 1 — Approach (box-relative, dynamic):
    Subscribes to /target_pose_left and /target_pose_right (published
    live by the grasp visualizer, based on wherever FoundationPose
    currently detects the box). Takes the latest grasp pose as a
    baseline, then pinches each hand inward by APPROACH_DIST_M along
    world Y to press into the box:
        Left hand:  y -= APPROACH_DIST_M
        Right hand: y += APPROACH_DIST_M
    Orientation is carried over unchanged from the grasp pose. Since
    the box can be anywhere, the ACHIEVED hand position after this
    stage varies run-to-run — it is NOT the fixed lift point below.

  STAGE 2 — Lift (fixed, box-independent):
    Regardless of where Stage 1 ended up, both hands are driven to a
    SECOND, fixed, pre-verified-reachable point in space:
        Left hand:  (LIFT_X, +LIFT_HALF_WIDTH_Y, LIFT_Z)
        Right hand: (LIFT_X, -LIFT_HALF_WIDTH_Y, LIFT_Z)
    with identity (unity) orientation. This is a known carry pose,
    confirmed reachable by manually publishing PoseStamped to
    /target_pose_left/right against H2_ik_node.py beforehand. The
    target is NOT derived from Stage 1's achieved position in any way.

Both stages use the same reachability handling: IK never rejects, it
just converges to the closest achievable pose and logs a warning if
the gap exceeds REACHABILITY_THRESH_M.

Usage:
  conda activate ik_env
  source /opt/ros/jazzy/setup.bash
  python3 H2_lift_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import time
import os

# ── Constants (mirrors H2_ik_node.py) ───────────────────────────────────────

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH      = os.path.join(_REPO_ROOT, "assets", "h2_description", "H2.urdf")
LEFT_EE_FRAME  = "left_hand_link"
RIGHT_EE_FRAME = "right_hand_link"

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_NAMES  = ARM_JOINT_NAMES[:7]
RIGHT_ARM_NAMES = ARM_JOINT_NAMES[7:]

PUBLISH_HZ      = 50
MAX_JOINT_VEL   = 8.0    # degrees per second — slow, contact-safe (was 30 in H2_ik_node.py)
MIN_INTERP_TIME = 1.5    # seconds — minimum move duration even for tiny deltas

SETTLE_TIME_S   = 1.0    # pause after approach, before lift, to let grip stabilize

IK_MAX_ITER             = 1000
IK_EPS_POS              = 1e-3
IK_EPS_ROT              = 1e-2
IK_DT                   = 0.5
IK_DAMP                 = 1e-6
REACHABILITY_THRESH_M   = 0.03   # 30mm — matches H2_ik_node.py
WAYPOINT_STEP_M         = 0.02   # 20mm Cartesian steps

APPROACH_DIST_M = 0.15   # pinch-inward distance for Stage 1 (box-relative)

LIFT_X            = 0.20    # fixed Stage 2 carry-pose target — confirmed reachable
LIFT_HALF_WIDTH_Y = 0.20    # 40cm apart total, ±20cm from centerline
LIFT_Z             = 0.275  # confirmed reachable

# Index of wrist_pitch within each arm's 7-joint list:
# [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
WRIST_PITCH_LOCAL_IDX = 5

WRIST_TILT_DEG = -9.0   # applied to wrist_pitch AFTER lift completes — sign is a
                         # best guess (URDF axis direction unknown). If the wrist
                         # tilts DOWN instead of up on first test, just flip this
                         # to -20.0.

POSE_WAIT_TIMEOUT_S = 5.0   # how long to wait for /target_pose_* messages


class ArmIK:
    """Self-contained IK solver for one arm — used for one-shot moves."""

    def __init__(self, model, ee_id, joint_names):
        self.model = model
        self.ee_id = ee_id
        self.v_indices = []
        self.q_indices = []
        for name in joint_names:
            jid = model.getJointId(name)
            self.v_indices.append(model.joints[jid].idx_v)
            self.q_indices.append(model.joints[jid].idx_q)

    def current_ee_pose(self, q_full):
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q_full)
        pin.updateFramePlacements(self.model, data)
        oMf = data.oMf[self.ee_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def solve(self, q_full_start, target: pin.SE3, logger, arm_name, lock_joints=None):
        """
        Waypoint-seeded IK, identical approach to H2_ik_node.py's
        _plan_and_solve, but blocking (no thread/cancel machinery).

        lock_joints: optional dict {joint_index_in_q_indices: fixed_angle_rad}.
        After each IK iteration, locked joints are forced back to their fixed
        angle and excluded from the velocity update — this prevents the
        solver's null-space from swinging that joint through a large arc
        to satisfy a small Cartesian position change (e.g. wrist_pitch
        absorbing most of the motion near a kinematic singularity).

        Returns (new_arm_q, achieved_pos).
        """
        data = self.model.createData()
        q = q_full_start.copy()

        # Apply initial lock values and figure out their v_indices so we can
        # zero their contribution to dq each iteration.
        locked_v_indices = set()
        if lock_joints:
            for local_idx, fixed_angle in lock_joints.items():
                q[self.q_indices[local_idx]] = fixed_angle
                locked_v_indices.add(self.v_indices[local_idx])

        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        current_pos = data.oMf[self.ee_id].translation.copy()
        current_rot = data.oMf[self.ee_id].rotation.copy()

        target_pos = target.translation
        target_rot = target.rotation

        dist = np.linalg.norm(target_pos - current_pos)
        n_waypoints = max(int(np.ceil(dist / WAYPOINT_STEP_M)), 1)

        logger.info(
            f"{arm_name}: planning {n_waypoints} waypoints over {dist*1000:.0f}mm"
        )

        positions = [
            current_pos + (target_pos - current_pos) * (i / n_waypoints)
            for i in range(1, n_waypoints + 1)
        ]

        r_current = Rotation.from_matrix(current_rot)
        r_target  = Rotation.from_matrix(target_rot)
        key_rots  = Rotation.concatenate([r_current, r_target])
        slerp     = Slerp([0, 1], key_rots)
        rotations = [
            slerp(i / n_waypoints).as_matrix()
            for i in range(1, n_waypoints + 1)
        ]

        for pos, rot in zip(positions, rotations):
            waypoint = pin.SE3(rot, pos)

            for _ in range(IK_MAX_ITER):
                pin.forwardKinematics(self.model, data, q)
                pin.updateFramePlacements(self.model, data)

                oMf = data.oMf[self.ee_id]
                err = pin.log6(oMf.inverse() * waypoint).vector
                J = pin.computeFrameJacobian(
                    self.model, data, q, self.ee_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )

                if np.linalg.norm(err[:3]) < IK_EPS_POS and np.linalg.norm(err[3:]) < IK_EPS_ROT:
                    break

                JJT  = J @ J.T
                damp = IK_DAMP * np.eye(JJT.shape[0])
                dq   = J.T @ np.linalg.solve(JJT + damp, err)

                dq_masked = np.zeros(self.model.nv)
                for idx in self.v_indices:
                    if idx not in locked_v_indices:
                        dq_masked[idx] = dq[idx]

                q = pin.integrate(self.model, q, IK_DT * dq_masked)
                q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

                # Re-assert locked joint angles in case integration drifted them
                # (shouldn't happen since dq is zeroed, but guards against
                # any cross-coupling in the integration step)
                if lock_joints:
                    for local_idx, fixed_angle in lock_joints.items():
                        q[self.q_indices[local_idx]] = fixed_angle

        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        achieved = data.oMf[self.ee_id].translation.copy()
        err_m = np.linalg.norm(achieved - target_pos)

        logger.info(
            f"{arm_name} final error: {err_m*1000:.1f}mm | "
            f"achieved: {achieved.round(3)} | desired: {target_pos.round(3)}"
        )

        if err_m > REACHABILITY_THRESH_M:
            logger.warn(
                f"{arm_name} target only partially reachable — "
                f"{err_m*1000:.1f}mm exceeds {REACHABILITY_THRESH_M*1000:.0f}mm threshold. "
                f"Using closest achievable pose instead (hand will NOT travel full distance)."
            )
            # NOTE: unlike H2_ik_node.py we do NOT reject here — for grasping,
            # "as close as physically possible" is exactly what we want, since
            # the box surface is expected to stop the hand short of the
            # nominal target. We still report it loudly above.

        new_arm_q = np.array([q[idx] for idx in self.q_indices])
        return new_arm_q, achieved


class H2LiftNode(Node):

    def __init__(self):
        super().__init__("h2_lift_node")

        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.get_logger().info(f"Loaded H2 URDF — DoFs: {self.model.nq}")

        left_ee_id  = self.model.getFrameId(LEFT_EE_FRAME)
        right_ee_id = self.model.getFrameId(RIGHT_EE_FRAME)

        self.left_ik  = ArmIK(self.model, left_ee_id,  LEFT_ARM_NAMES)
        self.right_ik = ArmIK(self.model, right_ee_id, RIGHT_ARM_NAMES)

        self.q_full = pin.neutral(self.model)

        self.current_joint_state = None
        self.last_left_pose  = None
        self.last_right_pose = None

        self.create_subscription(JointState, "/joint_states", self._cb_joint_state, 10)
        self.create_subscription(PoseStamped, "/target_pose_left",  self._cb_left,  10)
        self.create_subscription(PoseStamped, "/target_pose_right", self._cb_right, 10)
        self.pub = self.create_publisher(JointState, "/arm_joints", 10)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _cb_joint_state(self, msg: JointState):
        self.current_joint_state = msg

    def _cb_left(self, msg: PoseStamped):
        self.last_left_pose = msg

    def _cb_right(self, msg: PoseStamped):
        self.last_right_pose = msg

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pose_msg_to_se3(msg: PoseStamped) -> pin.SE3:
        p = msg.pose.position
        o = msg.pose.orientation
        t = np.array([p.x, p.y, p.z])
        R = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        return pin.SE3(R, t)

    def _sync_q_full_from_joint_states(self):
        """Seed q_full from Isaac Sim's actual current joint angles."""
        if self.current_joint_state is None:
            self.get_logger().warn(
                "No /joint_states received yet — using neutral pose as IK seed."
            )
            return

        name_to_pos = dict(zip(self.current_joint_state.name, self.current_joint_state.position))
        for names, ik in ((LEFT_ARM_NAMES, self.left_ik), (RIGHT_ARM_NAMES, self.right_ik)):
            for name, idx in zip(names, ik.q_indices):
                if name in name_to_pos:
                    self.q_full[idx] = name_to_pos[name]

    def _publish_and_wait(self, left_arm_q, right_arm_q, left_start, right_start):
        """
        Velocity-limited smooth interpolation to the new joint targets,
        published at PUBLISH_HZ, blocking until motion completes.
        """
        delta_deg_l = np.abs(np.degrees(left_arm_q)  - np.degrees(left_start))
        delta_deg_r = np.abs(np.degrees(right_arm_q) - np.degrees(right_start))
        max_delta   = max(np.max(delta_deg_l), np.max(delta_deg_r))
        duration    = max(max_delta / MAX_JOINT_VEL, MIN_INTERP_TIME)

        self.get_logger().info(
            f"Interpolating over {duration:.2f}s (max joint delta {max_delta:.1f} deg)"
        )

        start_time = time.time()
        period = 1.0 / PUBLISH_HZ

        while rclpy.ok():
            elapsed = time.time() - start_time
            alpha = min(elapsed / duration, 1.0)
            alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep

            q_left  = left_start  + alpha_s * (left_arm_q  - left_start)
            q_right = right_start + alpha_s * (right_arm_q - right_start)

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ARM_JOINT_NAMES
            msg.position = q_left.tolist() + q_right.tolist()
            self.pub.publish(msg)

            rclpy.spin_once(self, timeout_sec=0.0)

            if alpha >= 1.0:
                break
            time.sleep(period)

        return q_left, q_right

    def _wait_for_grasp_poses(self):
        self.get_logger().info("Waiting for /target_pose_left and /target_pose_right ...")
        start = time.time()
        while rclpy.ok() and (self.last_left_pose is None or self.last_right_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > POSE_WAIT_TIMEOUT_S:
                raise RuntimeError(
                    "Timed out waiting for /target_pose_left and /target_pose_right. "
                    "Is the grasp visualizer node running?"
                )
        # Grab a couple of /joint_states messages too
        start = time.time()
        while rclpy.ok() and self.current_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > POSE_WAIT_TIMEOUT_S:
                self.get_logger().warn(
                    "No /joint_states received — proceeding with neutral IK seed."
                )
                break

    # ── Main sequence ────────────────────────────────────────────────────────

    def run_sequence(self):
        self._wait_for_grasp_poses()
        self._sync_q_full_from_joint_states()

        left_grasp_se3  = self._pose_msg_to_se3(self.last_left_pose)
        right_grasp_se3 = self._pose_msg_to_se3(self.last_right_pose)

        self.get_logger().info(
            f"Baseline grasp poses (box-relative) — "
            f"Left: {left_grasp_se3.translation.round(3)} | "
            f"Right: {right_grasp_se3.translation.round(3)}"
        )

        # Capture starting joint angles for interpolation
        left_start  = np.array([self.q_full[idx] for idx in self.left_ik.q_indices])
        right_start = np.array([self.q_full[idx] for idx in self.right_ik.q_indices])

        # ── Stage 1: Approach — pinch inward toward the box, wherever it is ──
        left_approach_pos  = left_grasp_se3.translation.copy()
        left_approach_pos[1] -= APPROACH_DIST_M     # left hand: -Y
        right_approach_pos = right_grasp_se3.translation.copy()
        right_approach_pos[1] += APPROACH_DIST_M    # right hand: +Y

        left_approach_se3  = pin.SE3(left_grasp_se3.rotation,  left_approach_pos)
        right_approach_se3 = pin.SE3(right_grasp_se3.rotation, right_approach_pos)

        self.get_logger().info("=== Stage 1: Approach (box-relative, dynamic) ===")
        left_arm_q, left_achieved = self.left_ik.solve(
            self.q_full, left_approach_se3, self.get_logger(), "Left"
        )
        right_arm_q, right_achieved = self.right_ik.solve(
            self.q_full, right_approach_se3, self.get_logger(), "Right"
        )

        # Commit solved q back into q_full before next solve
        for idx, val in zip(self.left_ik.q_indices, left_arm_q):
            self.q_full[idx] = val
        for idx, val in zip(self.right_ik.q_indices, right_arm_q):
            self.q_full[idx] = val

        left_start, right_start = self._publish_and_wait(
            left_arm_q, right_arm_q, left_start, right_start
        )

        self.get_logger().info(
            f"Approach complete — Left achieved: {left_achieved.round(3)} | "
            f"Right achieved: {right_achieved.round(3)} "
            f"(this position is box-dependent and will differ run to run)"
        )

        self.get_logger().info(
            f"Settling for {SETTLE_TIME_S:.1f}s to let grip/contact stabilize before lift..."
        )
        settle_start = time.time()
        while rclpy.ok() and (time.time() - settle_start) < SETTLE_TIME_S:
            rclpy.spin_once(self, timeout_sec=0.05)

        # ── Stage 2: Lift — fixed carry-pose target, independent of Stage 1 ──
        identity_rot = np.eye(3)
        left_lift_pos  = np.array([LIFT_X,  LIFT_HALF_WIDTH_Y, LIFT_Z])
        right_lift_pos = np.array([LIFT_X, -LIFT_HALF_WIDTH_Y, LIFT_Z])

        left_lift_se3  = pin.SE3(identity_rot, left_lift_pos)
        right_lift_se3 = pin.SE3(identity_rot, right_lift_pos)

        self.get_logger().info(
            f"=== Stage 2: Lift (fixed target — Left: {left_lift_pos.round(3)} | "
            f"Right: {right_lift_pos.round(3)}) ==="
        )
        left_arm_q2, left_achieved2 = self.left_ik.solve(
            self.q_full, left_lift_se3, self.get_logger(), "Left"
        )
        right_arm_q2, right_achieved2 = self.right_ik.solve(
            self.q_full, right_lift_se3, self.get_logger(), "Right"
        )

        for idx, val in zip(self.left_ik.q_indices, left_arm_q2):
            self.q_full[idx] = val
        for idx, val in zip(self.right_ik.q_indices, right_arm_q2):
            self.q_full[idx] = val

        left_start, right_start = self._publish_and_wait(
            left_arm_q2, right_arm_q2, left_start, right_start
        )

        self.get_logger().info(
            f"Lift complete — Left achieved: {left_achieved2.round(3)} | "
            f"Right achieved: {right_achieved2.round(3)}"
        )

        # ── Stage 3: Wrist tilt — pure joint-space delta, no IK needed ──────
        # Only wrist_pitch changes; every other joint stays exactly where
        # the lift step left it.
        tilt_rad = np.radians(WRIST_TILT_DEG)

        left_arm_q3  = left_arm_q2.copy()
        right_arm_q3 = right_arm_q2.copy()
        left_arm_q3[WRIST_PITCH_LOCAL_IDX]  += tilt_rad
        right_arm_q3[WRIST_PITCH_LOCAL_IDX] += tilt_rad

        # Respect joint limits so we don't command past hardware range
        left_lo  = self.model.lowerPositionLimit[self.left_ik.q_indices[WRIST_PITCH_LOCAL_IDX]]
        left_hi  = self.model.upperPositionLimit[self.left_ik.q_indices[WRIST_PITCH_LOCAL_IDX]]
        right_lo = self.model.lowerPositionLimit[self.right_ik.q_indices[WRIST_PITCH_LOCAL_IDX]]
        right_hi = self.model.upperPositionLimit[self.right_ik.q_indices[WRIST_PITCH_LOCAL_IDX]]

        left_arm_q3[WRIST_PITCH_LOCAL_IDX]  = np.clip(left_arm_q3[WRIST_PITCH_LOCAL_IDX],  left_lo,  left_hi)
        right_arm_q3[WRIST_PITCH_LOCAL_IDX] = np.clip(right_arm_q3[WRIST_PITCH_LOCAL_IDX], right_lo, right_hi)

        self.get_logger().info(
            f"=== Stage 3: Wrist tilt ({WRIST_TILT_DEG:+.1f}deg) — "
            f"Left wrist_pitch: {np.degrees(left_arm_q2[WRIST_PITCH_LOCAL_IDX]):.1f} -> "
            f"{np.degrees(left_arm_q3[WRIST_PITCH_LOCAL_IDX]):.1f}deg | "
            f"Right wrist_pitch: {np.degrees(right_arm_q2[WRIST_PITCH_LOCAL_IDX]):.1f} -> "
            f"{np.degrees(right_arm_q3[WRIST_PITCH_LOCAL_IDX]):.1f}deg ==="
        )

        for idx, val in zip(self.left_ik.q_indices, left_arm_q3):
            self.q_full[idx] = val
        for idx, val in zip(self.right_ik.q_indices, right_arm_q3):
            self.q_full[idx] = val

        self._publish_and_wait(left_arm_q3, right_arm_q3, left_start, right_start)

        self.get_logger().info("=== Sequence finished ===")


def main():
    rclpy.init()
    node = H2LiftNode()
    try:
        node.run_sequence()
    except RuntimeError as e:
        node.get_logger().error(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()