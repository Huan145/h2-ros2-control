#!/usr/bin/env python3
"""
H2 Local Navigator Node
-----------------------
Walks the H2 to a grasp-ready standoff pose in front of the KLT box, then
hands over to the IK/lift pipeline (which the orchestrator starts only AFTER
the locomotion policy is stopped — no simultaneous arm arbitration).

Terminal state (dictated by the grasp convention in
pose_overlay_visualizer_tracking_grasp.py — hands grasp the ±box-Y faces,
fingers +X pelvis):
    - box Y axis (horizontal projection) parallel to pelvis Y (either sign)
    - box center at (STANDOFF_X, 0) in the pelvis frame
      STANDOFF_X default 0.45 m — derived from the IK workspace:
      grasp hand-x ~= box_x - 0.09 (GRASP_PELVIS_OFFSET) must land near the
      0.31 m neutral reach, inside H2_ik_node's 30 mm reject threshold.

Inputs:
    /tracking/output   vision_msgs/Detection3DArray   box pose, CAMERA frame
    TF                 pelvis <- Camera               same chain the grasp
                                                      visualizer uses
    (deliberately NOT /target_pose_left/right — those contain grasp tuning
    offsets: GRASP_REACH, GRASP_PELVIS_OFFSET)

Outputs:
    cmd_vel                  geometry_msgs/Twist   consumed by
                                                   h2_fullbody_controller
    ~/status   (latched)     std_msgs/String       idle|running|succeeded|failed
    ~/result   (latched)     std_msgs/String       JSON: final dx/dy/dyaw or
                                                   failure reason
    ~/target_point           geometry_msgs/PointStamped  standoff point (debug)

Services:
    ~/start    std_srvs/Trigger
    ~/cancel   std_srvs/Trigger

Velocity constraints:
    linear  |v| in [0.17, v_max] m/s   (v_max default 0.30 — tracking
                                        fragility while walking; spec allows
                                        up to 0.50)
    angular |w| in [0.25, 0.50] rad/s

Timing: use_sim_time=True, staleness measured from message header stamps
against /clock — matches h2_fullbody_controller. A lost track first HOLDS
(zero cmd) for lost_grace_s so FoundationPose's periodic full re-estimation
(reset_period, default 5000 ms) can rescue the run, and only then fails.

State machine:
    IDLE -> ROTATE_TO_GOAL -> APPROACH -> FINAL_ALIGN -> DONE
    Any active state -> HOLD (pose stale) -> resume | FAILED

Run (same env as the controller):
    conda activate <ros2 env>
    source /opt/ros/jazzy/setup.bash
    python3 h2_local_navigator.py
"""

import json
import math
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

import tf2_ros
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray


# ── Helpers ──────────────────────────────────────────────────────────────────

def quat_to_rot(q):
    """Quaternion msg (x,y,z,w fields) -> 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_to_matrix(ts):
    tr, ro = ts.transform.translation, ts.transform.rotation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(ro)
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def wrap_half(a):
    """Wrap to [-pi/2, pi/2] — grasp-axis alignment is pi-symmetric because
    either sign of box Y is acceptable (left/right assignment handles it)."""
    a = wrap(a)
    if a > math.pi / 2:
        a -= math.pi
    elif a < -math.pi / 2:
        a += math.pi
    return a


class Phase(Enum):
    IDLE = "idle"
    ROTATE_TO_GOAL = "rotate_to_goal"
    APPROACH = "approach"
    FINAL_ALIGN = "final_align"
    HOLD = "hold"
    DONE = "succeeded"
    FAILED = "failed"


# ── Node ─────────────────────────────────────────────────────────────────────

class H2LocalNavigator(Node):

    def __init__(self):
        super().__init__("h2_local_navigator")
        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time",
                                       rclpy.Parameter.Type.BOOL, True)]
        )

        # ── parameters ──────────────────────────────────────────────────────
        p = self.declare_parameter
        p("tracking_topic", "/tracking/output")
        p("cmd_vel_topic", "cmd_vel")
        p("pelvis_frame", "pelvis")
        p("camera_frame", "Camera")
        p("control_rate_hz", 20.0)

        p("standoff_x", 0.45)          # [m] box center x at handover (see header)
        p("v_min", 0.17)               # [m/s] policy can't track slower
        p("v_max", 0.30)               # [m/s] capped below the 0.5 spec limit
                                       #       to protect FoundationPose tracking
        p("w_min", 0.25)               # [rad/s]
        p("w_max", 0.50)

        p("kp_lin", 1.0)
        p("kp_ang", 1.5)

        p("pos_tol", 0.05)             # [m]  compatible with IK workspace slack
        p("yaw_tol", math.radians(4))  # [rad] grasp-axis alignment tolerance
        p("capture_radius", 0.15)      # [m]  APPROACH -> FINAL_ALIGN
        p("bearing_gate", math.radians(30))
        p("settle_cycles", 20)         # 1 s at 20 Hz inside tolerance, cmd=0

        p("nav_timeout_s", 60.0)       # [s] sim time; total budget per start —
                                       #     guarantees every start terminates
        p("pose_timeout", 0.8)         # [s] sim time; stale -> HOLD (zero cmd)
        p("lost_grace_s", 8.0)         # [s] HOLD duration before FAILED —
                                       #     covers a 5 s reset_period rescue
        p("pose_lpf_alpha", 0.35)      # EMA on (x, y, grasp-axis vector)
        p("min_horizontal_y", 0.20)    # reject frames where box Y is near-vertical

        g = lambda n: self.get_parameter(n).value
        self.standoff_x = g("standoff_x")
        self.v_min, self.v_max = g("v_min"), g("v_max")
        self.w_min, self.w_max = g("w_min"), g("w_max")
        self.kp_lin, self.kp_ang = g("kp_lin"), g("kp_ang")
        self.pos_tol, self.yaw_tol = g("pos_tol"), g("yaw_tol")
        self.capture_radius = g("capture_radius")
        self.bearing_gate = g("bearing_gate")
        self.settle_cycles = g("settle_cycles")
        self.nav_timeout_s = g("nav_timeout_s")
        self.pose_timeout = g("pose_timeout")
        self.lost_grace_s = g("lost_grace_s")
        self.alpha = g("pose_lpf_alpha")
        self.min_hy = g("min_horizontal_y")
        self.pelvis_frame = g("pelvis_frame")
        self.camera_frame = g("camera_frame")

        # ── state ───────────────────────────────────────────────────────────
        self.phase = Phase.IDLE
        self.phase_before_hold = None
        self.hold_since = None
        self.start_time = None
        self.settle_count = 0

        # Filtered box state in pelvis frame:
        #   bx, by      box center (horizontal)
        #   gy          unit 2-vector, horizontal projection of box Y axis
        self.bx = self.by = None
        self.gy = None
        self.last_stamp = None         # rclpy.time.Time of last accepted pose

        # ── TF ──────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── I/O ─────────────────────────────────────────────────────────────
        self.create_subscription(Detection3DArray, g("tracking_topic"),
                                 self.on_tracking, 10)
        self.pub_cmd = self.create_publisher(Twist, g("cmd_vel_topic"), 10)
        self.pub_point = self.create_publisher(PointStamped, "~/target_point", 10)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_status = self.create_publisher(String, "~/status", latched)
        self.pub_result = self.create_publisher(String, "~/result", latched)

        self.create_service(Trigger, "~/start", self.on_start)
        self.create_service(Trigger, "~/cancel", self.on_cancel)

        self.dt = 1.0 / g("control_rate_hz")
        self.create_timer(self.dt, self.control_step)

        self.publish_status()
        self.get_logger().info(
            f"H2 local navigator ready | standoff_x={self.standoff_x:.2f} m | "
            f"v in [{self.v_min},{self.v_max}] m/s | "
            f"w in [{self.w_min},{self.w_max}] rad/s"
        )

    # ── perception input ────────────────────────────────────────────────────

    def on_tracking(self, msg: Detection3DArray):
        if not msg.detections:
            return
        det = msg.detections[0]
        pose = det.results[0].pose.pose

        T_pc = self.lookup_pelvis_cam(msg.header.stamp)
        if T_pc is None:
            return

        # Box center: camera -> pelvis
        p_cam = np.array([pose.position.x, pose.position.y, pose.position.z, 1.0])
        p_pel = (T_pc @ p_cam)[:3]

        # Box Y axis (grasp axis): camera -> pelvis, horizontal projection
        R_box_cam = quat_to_rot(pose.orientation)
        y_pel = T_pc[:3, :3] @ R_box_cam[:, 1]
        y_h = y_pel[:2]
        n = np.linalg.norm(y_h)
        if n < self.min_hy:
            self.get_logger().warn(
                "Box Y axis near-vertical — pose rejected (box on its side?)",
                throttle_duration_sec=2.0)
            return
        y_h = y_h / n

        # EMA filter. Canonicalize the grasp-axis sign first: FoundationPose
        # can flip a symmetric mesh by 180 deg between re-estimations, which
        # would otherwise cancel the average. Either sign is grasp-equivalent.
        if self.gy is not None and float(np.dot(y_h, self.gy)) < 0.0:
            y_h = -y_h

        if self.bx is None:
            self.bx, self.by = float(p_pel[0]), float(p_pel[1])
            self.gy = y_h
        else:
            a = self.alpha
            self.bx += a * (float(p_pel[0]) - self.bx)
            self.by += a * (float(p_pel[1]) - self.by)
            v = (1 - a) * self.gy + a * y_h
            self.gy = v / np.linalg.norm(v)

        # Staleness clock: header stamp if sane, else now (both sim time)
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        self.last_stamp = stamp if stamp.nanoseconds > 0 else self.get_clock().now()

    def lookup_pelvis_cam(self, stamp=None):
        """Look up pelvis <- Camera AT THE IMAGE CAPTURE TIME. FoundationPose
        adds 50-150 ms of pipeline latency; using the latest TF instead would
        bias the box position by 1.5-4.5 cm at 0.3 m/s and the bearing by
        1-3 deg at 0.4 rad/s — same order as the tolerances. Falls back to
        the latest transform only if the buffer can't serve the stamp."""
        query_time = rclpy.time.Time()
        if stamp is not None:
            t = rclpy.time.Time.from_msg(stamp)
            if t.nanoseconds > 0:
                query_time = t
        try:
            ts = self.tf_buffer.lookup_transform(
                self.pelvis_frame, self.camera_frame, query_time,
                timeout=rclpy.duration.Duration(seconds=0.05))
            return transform_to_matrix(ts)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass
        try:
            ts = self.tf_buffer.lookup_transform(
                self.pelvis_frame, self.camera_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            self.get_logger().warn(
                "TF at image stamp unavailable — using latest transform "
                "(expect small bias while moving)",
                throttle_duration_sec=5.0)
            return transform_to_matrix(ts)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF {self.pelvis_frame}<-{self.camera_frame} failed: {e}",
                throttle_duration_sec=2.0)
            return None

    def pose_age(self):
        if self.last_stamp is None:
            return float("inf")
        return (self.get_clock().now() - self.last_stamp).nanoseconds * 1e-9

    # ── goal geometry ───────────────────────────────────────────────────────

    def compute_errors(self):
        """Returns (tx, ty, eyaw):
        (tx, ty) — standoff point in pelvis frame (where the PELVIS should
                   be is the origin, so this is the remaining translation),
        eyaw     — rotation (rad, +ccw) that makes the grasp axis lateral.
        """
        # Approach direction: horizontal unit vector perpendicular to the
        # grasp axis, pointing from the box toward the robot (origin).
        n1 = np.array([self.gy[1], -self.gy[0]])
        b = np.array([self.bx, self.by])
        n = n1 if float(np.dot(n1, -b)) > 0.0 else -n1

        t = b + self.standoff_x * n      # standoff point in pelvis frame
        # Rotating the robot by dtheta rotates pelvis-frame vectors by
        # -dtheta, so to move the grasp axis from angle phi to ±pi/2:
        phi = math.atan2(self.gy[1], self.gy[0])
        eyaw = wrap_half(phi - math.pi / 2)
        return float(t[0]), float(t[1]), eyaw

    # ── services ────────────────────────────────────────────────────────────

    def on_start(self, req, resp):
        if self.pose_age() > self.pose_timeout:
            resp.success = False
            resp.message = "No fresh box track — refusing to start."
            return resp
        self.settle_count = 0
        self.hold_since = None
        self.start_time = self.get_clock().now()
        self.set_phase(Phase.ROTATE_TO_GOAL)
        tx, ty, eyaw = self.compute_errors()
        resp.success = True
        resp.message = (f"Approach started: standoff at ({tx:.2f},{ty:.2f}) m, "
                        f"grasp-axis error {math.degrees(eyaw):.1f} deg")
        return resp

    def on_cancel(self, req, resp):
        self.stop()
        self.set_phase(Phase.IDLE)
        resp.success = True
        resp.message = "Cancelled."
        return resp

    # ── velocity shaping ────────────────────────────────────────────────────

    def shape(self, err, kp, vmin, vmax, tol):
        """P-control clamped into [vmin, vmax]; zero inside tolerance.
        The floor exists because the policy can't track slow commands —
        approach is bang-bang-ish near the goal, absorbed by settle check."""
        if abs(err) < tol:
            return 0.0
        v = kp * err
        return math.copysign(min(max(abs(v), vmin), vmax), v)

    # ── control loop ────────────────────────────────────────────────────────

    def control_step(self):
        if self.phase in (Phase.IDLE, Phase.DONE, Phase.FAILED):
            return

        # ── global budget: every start terminates, orchestrator never hangs ─
        # Covers rotate<->approach oscillation and final_align limit cycles;
        # HOLD time counts too (the lost_grace_s check usually fires first).
        if self.start_time is not None:
            elapsed = (self.get_clock().now() -
                       self.start_time).nanoseconds * 1e-9
            if elapsed > self.nav_timeout_s:
                self.fail(f"timeout: {elapsed:.1f}s > "
                          f"nav_timeout_s={self.nav_timeout_s}s "
                          f"(last phase: {self.phase.value})")
                return

        age = self.pose_age()

        # ── lost-track handling: HOLD first, FAILED only after grace ──────
        if age > self.pose_timeout:
            if self.phase != Phase.HOLD:
                self.phase_before_hold = self.phase
                self.hold_since = self.get_clock().now()
                self.stop()
                self.set_phase(Phase.HOLD)
                self.get_logger().warn(
                    "Box track stale — holding (zero cmd), waiting for "
                    "FoundationPose re-estimation...")
            else:
                held = (self.get_clock().now() -
                        self.hold_since).nanoseconds * 1e-9
                if held > self.lost_grace_s:
                    self.fail(f"Track lost for {held:.1f}s "
                              f"(> {self.lost_grace_s}s grace)")
            self.stop()
            return

        if self.phase == Phase.HOLD:
            self.get_logger().info("Track recovered — resuming.")
            self.set_phase(self.phase_before_hold or Phase.ROTATE_TO_GOAL)

        tx, ty, eyaw = self.compute_errors()
        self.publish_point(tx, ty)

        dist = math.hypot(tx, ty)
        bearing = math.atan2(ty, tx)
        cmd = Twist()

        if self.phase == Phase.ROTATE_TO_GOAL:
            # Aim at the standoff point before walking (policy tracks vx
            # best). NOTE: large turns risk pushing the box out of the head
            # camera FOV — precondition for v1 is box roughly X-face-on.
            if dist < self.capture_radius:
                self.set_phase(Phase.FINAL_ALIGN)
            elif abs(bearing) < self.bearing_gate * 0.5:
                self.set_phase(Phase.APPROACH)
            else:
                cmd.angular.z = self.shape(bearing, self.kp_ang, self.w_min,
                                           self.w_max, self.bearing_gate * 0.4)

        elif self.phase == Phase.APPROACH:
            if dist < self.capture_radius:
                self.set_phase(Phase.FINAL_ALIGN)
            elif abs(bearing) > self.bearing_gate:
                self.set_phase(Phase.ROTATE_TO_GOAL)   # overshoot: re-aim
            else:
                cmd.linear.x = self.shape(tx, self.kp_lin, self.v_min,
                                          self.v_max, self.pos_tol)
                cmd.linear.y = self.shape(ty, self.kp_lin, self.v_min,
                                          self.v_max, self.pos_tol * 2)
                cmd.angular.z = self.shape(bearing, self.kp_ang, self.w_min,
                                           self.w_max, self.yaw_tol * 2)

        elif self.phase == Phase.FINAL_ALIGN:
            # Decoupled trim: heading (grasp axis lateral) first — position
            # errors are meaningless while the pelvis frame is rotating —
            # then fore/aft (backward walking if too close: tx < 0) and
            # lateral onto the box centerline.
            ex_ok = abs(tx) < self.pos_tol
            ey_ok = abs(ty) < self.pos_tol
            eyaw_ok = abs(eyaw) < self.yaw_tol

            if ex_ok and ey_ok and eyaw_ok:
                self.stop()
                self.settle_count += 1
                if self.settle_count >= self.settle_cycles:
                    self.succeed(tx, ty, eyaw)
                return
            self.settle_count = 0
            if not eyaw_ok:
                cmd.angular.z = self.shape(eyaw, self.kp_ang, self.w_min,
                                           self.w_max, self.yaw_tol)
            else:
                cmd.linear.x = self.shape(tx, self.kp_lin, self.v_min,
                                          self.v_max, self.pos_tol)
                cmd.linear.y = self.shape(ty, self.kp_lin, self.v_min,
                                          self.v_max, self.pos_tol)

        self.pub_cmd.publish(cmd)

    # ── terminal states ─────────────────────────────────────────────────────

    def succeed(self, tx, ty, eyaw):
        self.stop()
        self.set_phase(Phase.DONE)
        result = {
            "outcome": "succeeded",
            "dx_m": round(tx, 4),
            "dy_m": round(ty, 4),
            "dyaw_deg": round(math.degrees(eyaw), 2),
            "box_x_m": round(self.bx, 4),
            "box_y_m": round(self.by, 4),
        }
        self.publish_result(result)
        self.get_logger().info(f"Standoff reached: {result}")

    def fail(self, reason):
        self.stop()
        self.set_phase(Phase.FAILED)
        self.publish_result({"outcome": "failed", "reason": reason})
        self.get_logger().error(f"Navigation failed: {reason}")

    # ── plumbing ────────────────────────────────────────────────────────────

    def stop(self):
        self.pub_cmd.publish(Twist())

    def set_phase(self, phase: Phase):
        if phase != self.phase:
            self.phase = phase
            self.settle_count = 0
            self.get_logger().info(f"Phase -> {phase.value}")
            self.publish_status()

    def publish_status(self):
        running = self.phase in (Phase.ROTATE_TO_GOAL, Phase.APPROACH,
                                 Phase.FINAL_ALIGN, Phase.HOLD)
        msg = String()
        msg.data = "running" if running else self.phase.value
        self.pub_status.publish(msg)

    def publish_result(self, d):
        msg = String()
        msg.data = json.dumps(d)
        self.pub_result.publish(msg)

    def publish_point(self, tx, ty):
        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.pelvis_frame
        m.point.x, m.point.y = tx, ty
        self.pub_point.publish(m)


def main():
    rclpy.init()
    node = H2LocalNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
